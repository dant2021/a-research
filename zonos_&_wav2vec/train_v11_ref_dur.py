import torch
import torch.nn as nn
from kokoro.pipeline import KPipeline
from kokoro.model import KModel, KModel_modified
import os
import torchaudio
import torch.nn.functional as F     
from losses import MultiResolutionSTFTLoss
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gc
import random
import re
import wandb

# Initialize wandb - replace with your project name
wandb.init(project="temporal-alignment_v2")

"""
##################

ZONOS SPEAKER EMBEDDING

##################
"""

import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device("cpu")


DEFAULT_DEVICE = get_device()


class logFbankCal(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 80,
    ):
        super().__init__()
        self.fbankCal = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length * sample_rate),
            hop_length=int(hop_length * sample_rate),
            n_mels=n_mels,
        )

    def forward(self, x):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out


class ASP(nn.Module):
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2

        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        return x


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, in_ch=1, feat_dim="2d", **kwargs):
        super(ResNet, self).__init__()
        if feat_dim == "1d":
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        elif feat_dim == "2d":
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d
        elif feat_dim == "3d":
            self.NormLayer = nn.BatchNorm3d
            self.ConvLayer = nn.Conv3d
        else:
            print("error")

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, block_id=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, block_id=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, block_id=3)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, block_id=4)

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride, block_id))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def ResNet293(in_planes: int, **kwargs):
    return ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs)


class ResNet293_based(nn.Module):
    def __init__(
        self,
        in_planes: int = 64,
        embd_dim: int = 256,
        acoustic_dim: int = 80,
        featCal=None,
        dropout: float = 0,
        **kwargs,
    ):
        super(ResNet293_based, self).__init__()
        self.featCal = featCal
        self.front = ResNet293(in_planes)
        block_expansion = SimAMBasicBlock.expansion
        self.pooling = ASP(in_planes * block_expansion, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # Removed
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    def __init__(self, C, featCal):
        super(ECAPA_TDNN, self).__init__()
        self.featCal = featCal
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # Added
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        x = self.featCal(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


class SpeakerEmbedding(nn.Module):
    def __init__(self, ckpt_path: str = "ResNet293_SimAM_ASP_base.pt", device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        with torch.device(device):
            self.model = ResNet293_based()
            state_dict = torch.load(ckpt_path, weights_only=True, mmap=True, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.featCal = logFbankCal()

        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, 16_000).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = self._get_resampler(sample_rate)(wav)
        return wav

    def forward(self, wav: torch.Tensor, sample_rate: int):
        wav = self.prepare_input(wav, sample_rate).to(self.device, self.dtype)
        return self.model(wav).to(wav.device)


class SpeakerEmbeddingLDA(nn.Module):
    def __init__(self, device: str = DEFAULT_DEVICE):
        super().__init__()
        spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base.pt",
        )
        lda_spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base_LDA-128.pt",
        )

        self.device = device
        with torch.device(device):
            self.model = SpeakerEmbedding(spk_model_path, device)
            lda_sd = torch.load(lda_spk_model_path, weights_only=True)
            out_features, in_features = lda_sd["weight"].shape
            self.lda = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
            self.lda.load_state_dict(lda_sd)

        self.requires_grad_(False).eval()

    def forward(self, wav: torch.Tensor, sample_rate: int):
        emb = self.model(wav, sample_rate).to(torch.float32)
        return emb, self.lda(emb)


"""
##################

ZONOS SPEAKER EMBEDDING END

##################
"""


# Load whisper model
model_id = "openai/whisper-large-v3-turbo"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, use_safetensors=True
)
whisper_model.to('cuda')
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device='cuda',
    return_timestamps=True
)

class StyleEncoder(nn.Module):
    def __init__(self, input_dim=256, style_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, style_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


model = KModel_modified().cuda()

model.train()  # Ensure model is in training mode
pipeline = KPipeline(lang_code='en-us', model=model)

style_encoder = StyleEncoder().cuda()

# Training setup
optimizer = torch.optim.Adam(style_encoder.parameters(), lr=1e-4)

class StyleSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_style, ref_style):
        # Compute MSE loss
        mse = F.mse_loss(pred_style, ref_style)
        
        if pred_style.dim() == 1:
            pred_style = pred_style.unsqueeze(0)
            ref_style = ref_style.unsqueeze(0)
            
        cos_sim = F.cosine_similarity(pred_style, ref_style, dim=0).mean()
        
        return 1 - cos_sim + mse

criterion = nn.ModuleList([
    MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_lengths=[512, 1024, 2048]
    ).cuda(),
    StyleSimilarityLoss().cuda()
])

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

spk = SpeakerEmbeddingLDA().to('cuda')

def create_multi_voice_audio(text, voice_styles):
    """
    Creates a multi-voice audio sample by splitting text at sentence boundaries
    and generating each segment with a different voice style.
    """
    # Split text at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Generate audio for each segment
    audio_segments = []
    segments_info = []
    current_sample = 0
    
    for i, sentence in enumerate(sentences):
        # Select a random voice style for this segment
        style = random.choice(voice_styles)
        
        # Load voice style
        reference_style = pipeline.load_voice(style)
        
        # Convert text to phonemes
        phonemes, _ = pipeline.g2p(sentence)
        
        # Generate audio
        with torch.no_grad():
            output = KPipeline.infer(model, phonemes, reference_style, speed=1)
            audio = output.audio
        
        # Track segment information
        start_sample = current_sample
        end_sample = start_sample + len(audio)
        
        segments_info.append({
            'text': sentence,
            'style': style,
            'start_sample': start_sample,
            'end_sample': end_sample,
            'start_sec': start_sample / 24000,
            'end_sec': end_sample / 24000,
            'reference_style': reference_style.detach().clone(),
        })
        
        audio_segments.append(audio)
        current_sample = end_sample
    
    # Concatenate all segments
    combined_audio = torch.cat(audio_segments)
    
    return combined_audio, segments_info

def extract_zonos_features(audio, segments_info):
    """
    Extracts Zonos speaker embedding features for each segment in the audio.
    """
    # Convert to appropriate format for Zonos (currently at 24kHz)
    device = next(spk.parameters()).device
    audio = audio.to(device)

    print(f"Extracting Zonos features for {len(segments_info)} segments")   
    # Extract features for each segment
    segment_features = []
    
    for segment in segments_info:
        # Extract the segment audio
        start_sample = segment['start_sample']
        end_sample = segment['end_sample']
        
        # Handle very short segments
        segment_audio = audio[start_sample:end_sample]
        
        # Get speaker embedding
        with torch.no_grad():
            embedding, _ = spk(segment_audio.unsqueeze(0), sample_rate=24000)
            segment_feat = embedding.squeeze()

        segment_features.append(segment_feat)
    
    return segment_features

def generate_test_samples(epoch, style_encoder, test_voice_styles):
    """
    Generate test samples using voices not seen during training.
    """
    print(f"Generating test samples for epoch {epoch}...")
    test_outputs = []
    test_info = []
    
    # Use a shorter text for testing to save time
    for voice_style in test_voice_styles:
        print(f"  Testing with voice {voice_style}...")
        
        # Load voice style
        reference_style = pipeline.load_voice(voice_style)
        
        # Convert text to phonemes
        phonemes, _ = pipeline.g2p("these are the voices of the 3 storytellers in the Evermore Valley story")
        
        # Generate reference audio with original style
        with torch.no_grad():
            reference_output = KPipeline.infer(model, phonemes, reference_style, speed=1)
            reference_audio = reference_output.audio
            reference_dur = reference_output.pred_dur

            print(f"Reference audio shape: {reference_audio.shape}")

            # Save reference audio
            torchaudio.save(
                os.path.join("outputs", "test_samples", f"epoch_{epoch}_{voice_style}_reference.wav"),
                reference_audio.unsqueeze(0).cpu(),
                sample_rate=24000
            )

            reference_audio = reference_audio.cuda()
            
            # Extract Zonos features from reference audio
            embedding, _ = spk(reference_audio.unsqueeze(0), sample_rate=24000)
            zonos_features = embedding.squeeze().cuda()
            
            # Generate style vector from Zonos features using current model
            style_vector = style_encoder(zonos_features)
            style_vector = style_vector.unsqueeze(0)
            ref_dur = reference_dur.cuda()
            
            # Generate audio with predicted style
            predicted_output = model(phonemes, style_vector, speed=1, return_output=True, ref_dur=ref_dur)
            #print(f"Predicted output: {predicted_output}")
            predicted_audio = predicted_output.audio
            #print(f"Predicted audio: {predicted_audio}")
            #print(f"Predicted audio shape: {predicted_audio.shape}")
            # Save predicted audio
            torchaudio.save(
                os.path.join("outputs", "test_samples", f"epoch_{epoch}_{voice_style}_predicted.wav"),
                predicted_audio.unsqueeze(0).cpu(),
                sample_rate=24000
            )
            
            # Compute metrics
            style_vector = style_vector.cpu()
            reference_style = reference_style.cpu()
            predicted_audio = predicted_audio.cuda()
            reference_audio = reference_audio.cuda()

            ref_dur = ref_dur.cpu()
            pred_dur = predicted_output.pred_dur.cpu()
            direct_dur_loss = F.l1_loss(pred_dur.float(), ref_dur.float())  # L1 for individual durations


            style_loss = criterion[1](style_vector, reference_style)

            print(f"Style loss: {style_loss}")
            print(f"Direct dur loss: {direct_dur_loss}")

            test_info.append({
                "epoch": epoch,
                "voice": voice_style,
                "style_loss": style_loss,
                "direct_dur_loss": direct_dur_loss
            })
    
    # Log metrics to wandb
    for info in test_info:
        wandb.log({
            f"test_{info['voice']}_style_loss": info["style_loss"],
            f"test_{info['voice']}_direct_dur_loss": info["direct_dur_loss"],
            "epoch": epoch
        })
    
    print(f"Test samples generation complete for epoch {epoch}")
    torch.cuda.empty_cache()
    gc.collect()

    return test_info

def train_multi_voice(text, voice_styles, test_voice_styles, num_epochs=50):
    """
    Trains the StyleEncoder on multi-voice audio.
    """
    print(f"Creating multi-voice audio with {len(voice_styles)} different voices...")
    combined_audio, segments_info = create_multi_voice_audio(text, voice_styles)

    # Initialize the saved_audio and saved_ref structures
    saved_audio = {epoch: {} for epoch in range(0, num_epochs, 5)}
    saved_ref = {epoch: {} for epoch in range(0, num_epochs, 5)}
    
    segment_features = extract_zonos_features(combined_audio, segments_info)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_loss_stft = 0
        epoch_loss_style = 0
        epoch_loss_dur = 0
        epoch_loss_dur_cumsum = 0

        combined_audio, segments_info = create_multi_voice_audio(text, voice_styles)
        segment_features = extract_zonos_features(combined_audio, segments_info)
        
        # Process each segment
        for i, (zonos_features, segment) in enumerate(zip(segment_features, segments_info)):
            # Train on this segment
            optimizer.zero_grad()
            zonos_features = zonos_features.cuda()
            # Generate trainable style vector from zonos features
            style_vector = style_encoder(zonos_features)
            style_vector = style_vector.unsqueeze(0)
            style_vector = style_vector.requires_grad_(True)
            
            # Get reference style for comparison
            reference_style = segment['reference_style'].to(style_vector.device)
            
            # Convert text to phonemes
            phonemes, _ = pipeline.g2p(segment['text'])
            
            # Generate reference audio
            with torch.no_grad():
                reference_output = KPipeline.infer(model, phonemes, reference_style, speed=1)
                ref_audio = reference_output.audio
                ref_dur = reference_output.pred_dur
                ref_dur_raw = reference_output.pred_dur_raw

            predicted_output = model(phonemes, style_vector, speed=1, return_output=True)#, ref_dur=ref_dur)
            pred_dur_raw = predicted_output.pred_dur_raw
            pred_dur = predicted_output.pred_dur
            # Get audio lengths
            pred_len = predicted_output.audio.size(0)
            ref_len = ref_audio.size(0)
            duration_ratio = sum(ref_dur)/sum(predicted_output.pred_dur)
            max_len = max(pred_len, ref_len)

            # pad both audios to the max length
            pred_audio = F.pad(predicted_output.audio, (0, max_len - pred_len))
            ref_audio = F.pad(ref_audio, (0, max_len - ref_len))


            # Ensure both are on the CUDA device
            pred_audio = pred_audio.cuda()
            ref_audio = ref_audio.cuda()


            cumsum_ref_dur = torch.cumsum(ref_dur, dim=0)
            cumsum_pred_dur = torch.cumsum(pred_dur, dim=0)
            
            #ce_dur_loss = F.binary_cross_entropy_with_logits(pred_dur.float().flatten(), ref_dur.float().flatten()) 
            loss_dur_cumsum = F.l1_loss(cumsum_pred_dur.float(), cumsum_ref_dur.float())
            loss_dur_raw = F.l1_loss(pred_dur_raw.float(), ref_dur_raw.float())

            loss_dur = loss_dur_raw 
            loss_dur = loss_dur/len(phonemes)

            # Compute losses
            loss_stft = criterion[0](pred_audio, ref_audio)
            loss_style = criterion[1](style_vector, reference_style)
            loss = loss_stft + loss_dur * 100  

            # Backpropagate
            loss.backward()
            
            model.zero_grad()
            torch.nn.utils.clip_grad_norm_(style_encoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_loss_stft += loss_stft.item()
            epoch_loss_style += loss_style.item()
            epoch_loss_dur += loss_dur.item()
            epoch_loss_dur_cumsum += loss_dur_cumsum.item()
            
            # Save audio at specific epochs
            if epoch % 5 == 0:
                saved_audio[epoch][i] = pred_audio.detach().cpu()
                saved_ref[epoch][i] = ref_audio.detach().cpu()
        
        # Average losses over segments
        num_segments = len(segments_info)
        epoch_loss /= num_segments
        epoch_loss_stft /= num_segments
        epoch_loss_style /= num_segments
        epoch_loss_dur /= num_segments
        epoch_loss_dur_cumsum /= num_segments

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_stft": epoch_loss_stft,
            "loss_style": epoch_loss_style,
            "loss_dur": epoch_loss_dur,
            "duration_ratio": duration_ratio,
            "loss_dur_cumsum": epoch_loss_dur_cumsum
        })

        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, STFT: {epoch_loss_stft:.4f}, Style: {epoch_loss_style:.4f}, Dur: {epoch_loss_dur:.4f}")

        # Generate test samples every 25 epochs or at the end
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                style_encoder.eval()  # Set to evaluation mode
                generate_test_samples(epoch, style_encoder, test_voice_styles)
                style_encoder.train()  # Set back to training mode

        # Save audio at specific epochs
        if epoch % 25 == 0:
            # Create combined audio files with all segments
            all_pred_segments = []
            all_ref_segments = []
            
            for i in range(len(segments_info)):
                if i in saved_audio[epoch]:  # Check if this segment exists in saved_audio
                    all_pred_segments.append(saved_audio[epoch][i])
                    all_ref_segments.append(saved_ref[epoch][i])
            
            if all_pred_segments:
                # Save combined files
                torchaudio.save(
                    os.path.join("outputs", f"epoch_{epoch}_all_segments_predicted.wav"),
                    torch.cat(all_pred_segments).cpu().unsqueeze(0),
                    sample_rate=24000
                )
                torchaudio.save(
                    os.path.join("outputs", f"epoch_{epoch}_all_segments_reference.wav"),
                    torch.cat(all_ref_segments).cpu().unsqueeze(0),
                    sample_rate=24000
                )
        
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save the final model
    save_path = os.path.join("models", "style_encoder_multi_voice.pth")
    torch.save(style_encoder.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Return the audio from the last saved epoch
    last_epoch = (num_epochs - 1) // 5 * 5  # Get the last epoch that's a multiple of 5
    return saved_audio[last_epoch], saved_ref[last_epoch]

# Test with multi-voice training
if __name__ == "__main__":
    # The Five Voices of Evermore Valley story
    training_text = (
        "Once upon a time, in the small town of Evermore Valley, there lived five unique storytellers who shared a magical book. Each could only read certain parts, their voices changing the very nature of the tale. "
        "Maria, with her gentle voice like flowing water, began every story. In Evermore Valley, nestled between ancient mountains and whispering forests, there stood a curious shop called 'Fables & Fantasies.' The owner, an elderly man named Theodore, claimed his books contained real magic. Nobody believed him, of course. But young Lily, with her bright curious eyes and untamed imagination, felt drawn to the dusty shelves and forgotten tomes. "
        "Jackson, whose deep voice rumbled like distant thunder, always took over when adventure called. Lily discovered a leather-bound book with silver clasps that seemed to hum beneath her fingertips! The moment she opened it, the shop disappeared. She found herself standing on a cliff overlooking a vast ocean where impossible ships with rainbow sails glided through violet waters. A map materialized in her hand, marking a path to the Crystal Caves where, legends said, one could speak to the stars themselves. "
        "Elena narrated all moments of mystery, her voice sharp and precise like the tick of a clockwork puzzle. The journey wasn't as straightforward as Lily expected. Strange symbols appeared on the map that changed with the phases of the moon. Messages written in ancient script materialized then vanished from the edges of the pages. Most troubling was the shadowy figure that seemed to follow her, always vanishing when she turned to look. Was it Theodore? Or something else entirely? "
        "Cameron spoke with theatrical flair whenever magic and wonder filled the pages. The Crystal Caves sparkled with impossible light! Every surface reflected not Lily's face but different versions of her—older, younger, sometimes wearing strange clothes or speaking languages she didn't understand. Crystalline formations played melodies when touched, orchestrating symphonies that told stories of other worlds. Time behaved differently here; minutes stretched into hours while days compressed into heartbeats. "
        "Sophia, whose voice carried the warmth of a hearth fire, always brought the stories home. When Lily finally returned to the shop, Theodore was waiting with knowing eyes. 'Every reader finds a different adventure,' he explained, 'because every heart seeks a different truth.' Lily understood then that magic existed not just in faraway realms but in the spaces between words, in the quiet moments between heartbeats, and in the connections we forge with stories. She would return to Evermore Valley many times throughout her life, each visit revealing new chapters in both the magical book and her own unfolding story. "
        "And so the tale of Evermore Valley was told, passing from voice to voice, each bringing their own magic to the telling."
    )
    
    # List of voice styles to use - using the five specified voices
    voice_styles_small = ["am_michael", "am_fenrir", "af_nicole", "af_bella", "af_heart"]
    
    # Expanded list with all available voice styles
    voice_styles = [
        # American female voices
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", 
        "af_kore", "af_nicole", "af_nova", "af_sarah", "af_sky",
        
        # American male voices
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
        "am_michael", "am_onyx", "am_puck", "am_santa",
        
        # British female voices
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        
        # British male voices
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        
        # English voices (possibly other dialects)
        "ef_dora", "em_alex",
        
        # French female voice
        "ff_siwis",
        
        # Hindi voices
        "hf_alpha", "hf_beta", "hm_psi",
        
        # Italian voices
        "if_sara", "im_nicola",
    ]

    test_voice_styles = ["hm_omega", "em_santa","af_river"]
    
    print(f"Training StyleEncoder with {len(voice_styles)} different voices...")
    best_pred, best_ref = train_multi_voice(training_text, voice_styles, test_voice_styles, num_epochs=2000)
    
    # Save the final audio outputs
    torchaudio.save(
        os.path.join("outputs", "final_predicted_audio.wav"),
        best_pred.cpu().unsqueeze(0),
        sample_rate=24000
    )
    torchaudio.save(
        os.path.join("outputs", "final_reference_audio.wav"),
        best_ref.cpu().unsqueeze(0),
        sample_rate=24000
    )
    print("Training complete. Final audio files saved in outputs/")
