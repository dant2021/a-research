import torch
from torch.optim import AdamW
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src_simple.models.style_encoder import StyleEncoder
from src_simple.models.synthesizer import KokoroWrapper
from src_simple.dataset import AudioDataset

class Trainer:
    def __init__(self, config):
        # Initialize Whisper
        print("Loading Whisper...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "openai/whisper-large-v3-turbo"
        
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.bfloat16,
            device=self.device,
            generate_kwargs={"return_timestamps": True}
        )
        
        # Initialize Style Encoder
        self.style_encoder = StyleEncoder(
            whisper_dim=1280,  # Whisper's hidden dimension
            style_dim=256      # Output style dimension
        ).to(self.device)
        
        # Initialize Kokoro wrapper
        self.synthesizer = KokoroWrapper(
            voice=config['kokoro_voice'],
            lang_code=config['kokoro_lang_code']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.style_encoder.parameters(),
            lr=config['learning_rate']
        )
        
    def train_step(self, batch):
        audio = batch['audio'].to(self.device)
        
        # Get Whisper features and text
        with torch.no_grad():
            # Process audio through Whisper
            input_features = self.processor(
                audio.cpu().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Get encoder outputs
            encoder_outputs = self.whisper_model.get_encoder()(input_features)
            whisper_features = encoder_outputs.last_hidden_state
            
            # Get text transcription with timestamps
            result = self.whisper_pipe(audio.cpu().numpy(), return_timestamps=True)
            text = result["text"]
            word_timestamps = result["chunks"]
        
        # Extract style from Whisper features
        style_features = self.style_encoder(whisper_features, word_timestamps)
        
        # Generate speech with style
        generated_audio = self.synthesizer(text, style_features)
        
        # Compute reconstruction loss
        loss = torch.nn.functional.mse_loss(generated_audio, audio)
        
        return loss
    
    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

def main():
    config = {
        'learning_rate': 3e-4,
        'kokoro_voice': 'af_heart',
        'kokoro_lang_code': 'a',
        'num_epochs': 10
    }
    
    trainer = Trainer(config)
    dataset = AudioDataset("path/to/audio/files")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    trainer.train(dataloader, config['num_epochs'])

if __name__ == "__main__":
    main() 