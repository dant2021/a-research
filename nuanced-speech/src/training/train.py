import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from src.models.synthesis import KokoroSynthesizer
from src.models.bypass import BypassNetwork
from src.utils.audio import compute_spectrogram
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from src.training.dataset import AudioDataset
from configs.default_config import config
import torch.nn.functional as F
from src.models.losses import MultiResolutionSTFTLoss
import os

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
        )
        
        print(f"Whisper model loaded: large-v3-turbo")
        
        # Initialize Bypass Network
        print("\nInitializing Bypass Network...")
        self.bypass_network = BypassNetwork(
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
        self.training_device = torch.device("cuda:1")
        
        # Move models to devices
        self.whisper_model.to(self.whisper_device)
        self.bypass_network.to(self.training_device)
        self.speech_synthesis.to(self.training_device)
        
        # Initialize loss
        self.stft_loss = MultiResolutionSTFTLoss().to(self.training_device)
        
        # Initialize optimizer with gradient accumulation
        self.optimizer = AdamW(
            self.bypass_network.parameters(), 
            lr=config['learning_rate']
        )
        self.scaler = torch.amp.GradScaler('cuda')
        self.config = config
        """
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['sample_dir'], exist_ok=True)
        """
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
            # Process single audio sample - ensure it's 1D mono
            audio_np = audio_seq[0].cpu().numpy()  # Get first (and only) item from batch
            
            # Convert stereo to mono by averaging channels if needed
            if len(audio_np.shape) > 1 and audio_np.shape[0] == 2:
                audio_np = audio_np.mean(axis=0)  # Average the channels
            elif len(audio_np.shape) > 1:
                audio_np = audio_np.squeeze()
            
            # Ensure we have a 1D array
            if len(audio_np.shape) != 1:
                raise ValueError(f"Failed to convert to 1D array, got shape {audio_np.shape}")
            
            # Get text transcription
            result = self.whisper_pipe(audio_np, return_timestamps=True)
            text = result["text"]
            
            # Get whisper features by processing the audio through the encoder
            input_features = self.whisper_processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.whisper_device)
            
            # Get encoder outputs directly from the encoder
            encoder_outputs = self.whisper_model.get_encoder()(input_features)
            whisper_features = encoder_outputs.last_hidden_state
            whisper_features = whisper_features.to(self.training_device)
        
        # Now predict style features aligned with phoneme length
        style_features = self.bypass_network(whisper_features)  # [batch, seq_len, 256]
        
        # Generate final audio with style features
        with torch.set_grad_enabled(True):
            with autocast():
                generated_audio, _ = self.speech_synthesis(
                    text=text,
                    bypass_features=style_features  # Will be used as voice parameter
                )
                loss = self.stft_loss(generated_audio, target_audio)
        
        return loss
    
    def train(self, train_dataloader):
        self.whisper_model.eval()  # Freeze whisper
        self.speech_synthesis.eval()  # Freeze synthesis but keep grad
        self.bypass_network.train()
        
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
                
                if step % self.config['log_steps'] == 0:
                    print(f"Epoch {epoch}, Step {step}: Loss {loss.item():.4f}")
                
                # Save checkpoint
                if step % self.config['save_steps'] == 0:
                    self.save_checkpoint(epoch, step)
                
                # Generate validation sample
                if step % self.config['validate_steps'] == 0:
                    self.generate_sample(batch, f"validation_e{epoch}_s{step}.wav")
"""
    def save_checkpoint(self, epoch, step):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'bypass_state': self.bypass_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'config': self.config
        }
        path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_e{epoch}_s{step}.pt')
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.bypass_network.load_state_dict(checkpoint['bypass_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        return checkpoint['epoch'], checkpoint['step']
"""
def main():
    # Create trainer with config
    trainer = Trainer(config)
    
    # Create dataset and dataloader
    dataset = AudioDataset(
        audio_files="your audio files as a list",
        chunk_duration=30.0
    )

    def collate_fn(batch):
        # Filter out None values
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
            
        # Assuming each item in batch is a dictionary with 'audio' and 'target_audio' keys
        return {
            'audio': torch.stack([item['audio'] for item in batch]),
            'target_audio': torch.stack([item['target_audio'] for item in batch])
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Explicitly set batch size to 1
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Start training
    trainer.train(dataloader)

if __name__ == "__main__":
    main() 