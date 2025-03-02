import torch
import torch.nn as nn
from kokoro.pipeline import KPipeline
from kokoro.model import KModel
import os
import torchaudio
import torch.nn.functional as F     
from losses import MultiResolutionSTFTLoss
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import gc
import random
import re
import wandb

# Initialize wandb - replace with your project name
wandb.init(project="style-encoder")

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

# Load validation dataset
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

class StyleEncoder(nn.Module):
    def __init__(self, input_dim=1280, style_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, style_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Initialize models and pipeline
model = KModel().cuda()
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
        
        # Ensure tensors have proper dimensions for cosine similarity
        if pred_style.dim() == 1:
            pred_style = pred_style.unsqueeze(0)
            ref_style = ref_style.unsqueeze(0)
            
        # Compute cosine similarity (higher = more similar)
        cos_sim = F.cosine_similarity(pred_style, ref_style, dim=0).mean()
        
        # Return combined loss (1 - cosine_similarity ensures lower is better)
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
            'reference_style': reference_style.detach().clone()
        })
        
        audio_segments.append(audio)
        current_sample = end_sample
    
    # Concatenate all segments
    combined_audio = torch.cat(audio_segments)
    
    return combined_audio, segments_info

def extract_whisper_features(audio, segments_info):
    """
    Extracts Whisper features for each segment in the audio.
    """
    # Convert to 16kHz for Whisper
    audio_16k = torchaudio.functional.resample(
        audio.unsqueeze(0), 
        orig_freq=24000, 
        new_freq=16000
    ).squeeze(0)
    
    # Process through Whisper
    inputs = processor(
        audio_16k.cpu().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to('cuda')
    
    # Get encoder features
    with torch.no_grad():
        encoder_output = whisper_model.model.encoder(inputs)
        features = encoder_output.last_hidden_state.squeeze(0)
    
    # Extract features for each segment
    segment_features = []
    feature_rate = features.size(0) / (audio_16k.size(0) / 16000)  # features per second
    
    for segment in segments_info:
        # Convert time to feature indices
        start_idx = int(segment['start_sec'] * feature_rate)
        end_idx = int(segment['end_sec'] * feature_rate)
        
        # Ensure indices are valid
        start_idx = max(0, start_idx)
        end_idx = min(features.size(0), end_idx)
        
        if start_idx >= end_idx:
            # Handle edge case for very short segments
            mid_idx = (start_idx + min(features.size(0), start_idx + 1)) // 2
            segment_feat = features[mid_idx:mid_idx+1].mean(dim=0)
        else:
            # Average features over the segment
            segment_feat = features[start_idx:end_idx].mean(dim=0)
        
        segment_features.append(segment_feat)
    
    return segment_features

def train_multi_voice(text, voice_styles, num_epochs=50):
    """
    Trains the StyleEncoder on multi-voice audio.
    """
    print(f"Creating multi-voice audio with {len(voice_styles)} different voices...")
    combined_audio, segments_info = create_multi_voice_audio(text, voice_styles)
    
    # Save the training audio
    os.makedirs("outputs", exist_ok=True)
    torchaudio.save(
        os.path.join("outputs", "training_multi_voice.wav"),
        combined_audio.cpu().unsqueeze(0),
        sample_rate=24000
    )
    
    segment_features = extract_whisper_features(combined_audio, segments_info)
    
    # Initialize dictionaries to store audio at specific epochs
    save_epochs = [10, 20, 30, 40, 50]
    saved_audio = {epoch: {i: None for i in range(len(segments_info))} for epoch in save_epochs}
    saved_ref = {epoch: {i: None for i in range(len(segments_info))} for epoch in save_epochs}
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_loss_stft = 0
        epoch_loss_style = 0
        
        # Process each segment
        for i, (whisper_features, segment) in enumerate(zip(segment_features, segments_info)):
            # Train on this segment
            optimizer.zero_grad()
            
            # Generate trainable style vector from whisper features
            style_vector = style_encoder(whisper_features)
            
            # Get reference style for comparison
            reference_style = segment['reference_style'].to(style_vector.device)
            
            # Convert text to phonemes
            phonemes, _ = pipeline.g2p(segment['text'])
            
            # Expand style vector to match reference style shape
            style_vector = style_vector.expand_as(reference_style)
            style_vector = style_vector.requires_grad_(True)
            
            # Generate audio with predicted style
            with torch.set_grad_enabled(True):
                predicted_output = KPipeline.infer(model, phonemes, style_vector, speed=1)
                predicted_audio = predicted_output.audio
            
            # Generate reference audio
            with torch.no_grad():
                reference_output = KPipeline.infer(model, phonemes, reference_style, speed=1)
                ref_audio = reference_output.audio
            
            # Get audio lengths
            pred_len = predicted_audio.size(0)
            ref_len = ref_audio.size(0)
            min_len = min(pred_len, ref_len)
            
            # Trim both audios to the shorter length
            pred_audio = predicted_audio[:min_len]
            ref_audio = ref_audio[:min_len]
            
            # Compute losses
            loss_stft = criterion[0](pred_audio, ref_audio)
            loss_style = criterion[1](style_vector, reference_style)
            loss = 0.2*loss_stft + 10*loss_style
            
            # Backpropagate
            loss.backward()
            
            model.zero_grad()
            torch.nn.utils.clip_grad_norm_(style_encoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_loss_stft += loss_stft.item()
            epoch_loss_style += loss_style.item()
            
            # Save audio at specific epochs
            if epoch in save_epochs:
                saved_audio[epoch][i] = pred_audio.detach().clone()
                saved_ref[epoch][i] = ref_audio.detach().clone()
        
        # Average losses over segments
        num_segments = len(segments_info)
        epoch_loss /= num_segments
        epoch_loss_stft /= num_segments
        epoch_loss_style /= num_segments
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_stft": epoch_loss_stft,
            "loss_style": epoch_loss_style
        })

        # Save audio at specific epochs
        if epoch in save_epochs:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, STFT: {epoch_loss_stft:.4f}, Style: {epoch_loss_style:.4f}")
            
            # Create combined audio files with all segments
            all_pred_segments = []
            all_ref_segments = []
            
            for i in range(len(segments_info)):
                if saved_audio[epoch][i] is not None:
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
    last_epoch = max(save_epochs)
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
    voice_styles = ["am_michael", "am_fenrir", "af_nicole", "af_bella", "af_heart"]
    
    print(f"Training StyleEncoder with {len(voice_styles)} different voices...")
    best_pred, best_ref = train_multi_voice(training_text, voice_styles, num_epochs=1000)
    
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

