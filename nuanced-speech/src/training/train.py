import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from src.models.synthesis import KokoroSynthesizer
from src.models.bypass import BypassNetwork
from src.utils.audio import compute_spectrogram
import whisper
import numpy as np
from src.training.dataset import AudioDataset



class Trainer:
    def __init__(self, config):
        print("\n=== Initializing Models ===")
        
        # Initialize Whisper
        print("Loading Whisper...")
        self.whisper_model = whisper.load_model(config['whisper_model'])
        print(f"Whisper model loaded: {config['whisper_model']}")
        
        # Initialize Bypass Network
        self.bypass_network = BypassNetwork(
            whisper_hidden_dim=config['whisper_hidden_dim']
        )
        
        # Initialize Kokoro Synthesizer
        print("\nInitializing Kokoro...")
        self.speech_synthesis = KokoroSynthesizer(
            voice=config['kokoro_voice'],
            lang_code=config['kokoro_lang_code']
        )
        print(f"Kokoro initialized with voice: {config['kokoro_voice']}")
        
        # Setup devices
        self.whisper_device = torch.device("cuda:0")
        self.training_device = torch.device("cuda:1")
        
        # Move models to devices
        self.whisper_model.to(self.whisper_device)
        self.bypass_network.to(self.training_device)
        self.speech_synthesis.to(self.training_device)
    
    def print_tensor_stats(self, tensor, name):
        """Helper to print tensor statistics"""
        if isinstance(tensor, torch.Tensor):
            print(f"\n{name}:")
            print(f"Shape: {tensor.shape}")
            print(f"Mean: {tensor.mean().item():.4f}")
            print(f"Std: {tensor.std().item():.4f}")
            print(f"Min: {tensor.min().item():.4f}")
            print(f"Max: {tensor.max().item():.4f}")
    
    def diagnostic_run(self, batch):
        """Run models without bypass to check basic functionality"""
        print("\n=== Starting Diagnostic Run ===")
        
        # 1. Input audio stats
        audio = batch['audio']
        mel_spec = batch['mel_spec']
        self.print_tensor_stats(audio, "Input Audio")
        self.print_tensor_stats(mel_spec, "Input Mel Spectrogram")
        
        # 2. Whisper processing
        print("\n--- Whisper Processing ---")
        with torch.no_grad():
            try:
                # Get features from Whisper's encoder
                features = self.whisper_model.encoder(mel_spec.to(self.whisper_device))
                
                # Get text from Whisper's decoder
                result = self.whisper_model.decode(features)
                text = result.text
                print(f"\nWhisper Text Output: {text}")
                
                self.print_tensor_stats(features, "Whisper Encoder Output")
                
            except Exception as e:
                print(f"Error in Whisper processing: {str(e)}")
                raise
        
        # 3. Kokoro processing
        print("\n--- Kokoro Processing ---")
        try:
            generated_speech, phonemes = self.speech_synthesis(
                text=text,
                bypass_features=None  # No bypass features yet
            )
            
            print("\nPhonemes:", phonemes)
            self.print_tensor_stats(generated_speech, "Generated Speech")
            
        except Exception as e:
            print(f"Error in Kokoro processing: {str(e)}")
            raise
        
        return {
            'text': text,
            'generated_speech': generated_speech,
            'phonemes': phonemes
        }

    def save_diagnostic_audio(self, audio_tensor, filename):
        """Save audio tensor to file for manual inspection"""
        import torchaudio
        if isinstance(audio_tensor, torch.Tensor):
            audio_tensor = audio_tensor.cpu().detach()
        torchaudio.save(filename, audio_tensor, 16000)
    
    def training_step(self, batch):
        audio = batch['audio']
        mel_spec = batch['mel_spec']
        
        # Process through Whisper
        with torch.no_grad():
            # Get features from encoder
            whisper_features = self.whisper_model.encoder(
                mel_spec.to(self.whisper_device)
            )
            # Get text from decoder
            result = self.whisper_model.decode(whisper_features)
            text = result.text
            whisper_features = whisper_features.to(self.training_device)
        
        # Process through bypass network
        with autocast():
            bypass_features = self.bypass_network(whisper_features)
            
            # Generate speech with Kokoro using bypass features
            generated_speech, phonemes = self.speech_synthesis(
                text=text,
                bypass_features=bypass_features
            )
            
            # Compute loss
            recon_loss = nn.MSELoss()(generated_speech, audio.to(self.training_device))
            
            # Add feature matching loss
            feature_loss = self.compute_feature_loss(
                generated_speech, 
                audio.to(self.training_device)
            )
            
            loss = recon_loss + 0.1 * feature_loss
        
        return loss
    
    def compute_feature_loss(self, generated, target):
        """
        Compute feature matching loss between generated and target audio
        """
        # Extract mel spectrograms
        gen_mel = compute_spectrogram(generated)
        target_mel = compute_spectrogram(target)
        
        return nn.MSELoss()(gen_mel, target_mel)
    
    def train(self, train_dataloader):
        self.bypass_network.train()
        self.speech_synthesis.train()
        
        for step, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            
            loss = self.training_step(batch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if step % self.config['eval_steps'] == 0:
                print(f"Step {step}: Loss {loss.item():.4f}") 