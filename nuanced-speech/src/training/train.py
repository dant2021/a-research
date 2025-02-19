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
        )
        
        print(f"Whisper model loaded: large-v3-turbo")
        
        # Initialize Bypass Network
        print("\nInitializing Bypass Network...")
        self.bypass_network = StyleVoice(
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
            audio_np = audio_seq.cpu().numpy()
            
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