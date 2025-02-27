import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast
from src.models.synthesis import KokoroSynthesizer
from src.models.style_voice import StyleVoice
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src.training.dataset import AudioDataset
from configs.default_config import config
from src.models.losses import MultiResolutionSTFTLoss

class Trainer:
    def __init__(self, config):
        print("\n=== Initializing Models ===")
        
        # Initialize Whisper
        print("Loading Whisper...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 

        model_id = "openai/whisper-large-v3-turbo"
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.whisper_processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"return_timestamps": True}  # Force timestamp generation
        )
        
        print(f"Whisper model loaded: large-v3-turbo")
        
        # Initialize Bypass Network
        print("\nInitializing Bypass Network...")
        self.style_vector_network = StyleVoice(
            whisper_hidden_dim=config['whisper_hidden_dim'], #1280
            style_dim=256  # Fixed for Kokoro
        )
        
        # Initialize Kokoro
        print("\nInitializing Kokoro...")
        self.speech_synthesis = KokoroSynthesizer(
            voice=config['kokoro_voice'],
            lang_code=config['kokoro_lang_code']
        )
        
        # Setup devices
        self.whisper_device = torch.device("cuda:0")
        self.training_device = torch.device("cuda:0")
        
        # Move models to devices
        self.whisper_model.to(self.whisper_device)
        self.style_vector_network.to(self.training_device)
        self.speech_synthesis.to(self.training_device)
        
        # Initialize loss
        self.stft_loss = MultiResolutionSTFTLoss().to(self.training_device)
        
        # Initialize optimizer with gradient accumulation
        self.optimizer = AdamW(
            self.style_vector_network.parameters(), 
            lr=config['learning_rate']
        )
        self.scaler = torch.amp.GradScaler('cuda')
        self.config = config

    def print_tensor_stats(self, tensor, name):
        """Helper to print tensor statistics"""
        if isinstance(tensor, torch.Tensor):
            print(f"\n{name}:")
            print(f"Shape: {tensor.shape}")
            print(f"Mean: {tensor.mean().item():.4f}")
            print(f"Std: {tensor.std().item():.4f}")
            print(f"Min: {tensor.min().item():.4f}")
            print(f"Max: {tensor.max().item():.4f}")
    
    def train_step(self, batch):
        # Get audio sequence and target
        audio_seq = batch['audio'].to(self.whisper_device)
        target_audio = batch['target_audio'].to(self.training_device)
        
        # Get text from Whisper
        with torch.no_grad():
            # Ensure audio is in the correct format for Whisper (1D array)
            audio_np = audio_seq[0].cpu().numpy()  # Get first item from batch
            
            # Get text transcription
            result = self.whisper_pipe(audio_np, return_timestamps=True)
            text = result["text"]
            word_boundaries = result["chunks"]  # (start/end times in sec)
            
            # Get whisper features by processing the audio through the encoder
            input_features = self.whisper_processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.whisper_device)
            
            # Get encoder outputs directly from the encoder
            encoder_outputs = self.whisper_model.get_encoder()(input_features)
            whisper_features = encoder_outputs.last_hidden_state
            
            # Calculate frame rate properly
            total_frames = whisper_features.shape[1]
            frame_rate = total_frames / 30  # 30s audio
            
            # Convert word boundaries to frame indices
            word_frames = []
            for chunk in result['chunks']:
                start = chunk['timestamp'][0]
                end = chunk['timestamp'][1]
                
                if start is None or end is None:
                    print(f"Skipping chunk with missing timestamps: {chunk['text']}")
                    continue
                
                start_frame = int(start * frame_rate)
                end_frame = int(end * frame_rate)
                word_frames.append((start_frame, end_frame))
        
        # Now predict word-level style features
        style_features = self.style_vector_network(
            whisper_features.to(self.training_device),
            word_frames
        )
        
        # Generate final audio with style features
        with torch.set_grad_enabled(True):
            with autocast():
                generated_audio, _ = self.speech_synthesis(
                    text=text,
                    style_features=style_features
                )
                loss = self.stft_loss(generated_audio, target_audio)
        
        return loss
    
    def train(self, train_dataloader):
        self.whisper_model.eval()  # Freeze whisper
        self.speech_synthesis.eval()  # Freeze synthesis but keep grad
        self.style_vector_network.train()
        
        for epoch in range(self.config['num_epochs']):
            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                
                # Compute loss with gradient scaling
                with autocast():
                    loss = self.train_step(batch)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

def main():
    # Create trainer with config
    trainer = Trainer(config)
    
    # Initialize dataset with desired batch size
    dataset = AudioDataset(audio_files="your audio files as a list", chunk_duration=30.0, batch_size=32, num_workers=4)
    # Get the dataloader
    dataloader = dataset.get_dataloader()
    
    # Start training
    trainer.train(dataloader)

if __name__ == "__main__":
    main() 