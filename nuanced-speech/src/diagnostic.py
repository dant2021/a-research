import torch
from pathlib import Path
from src.training.train import Trainer
from src.training.dataset import AudioDataset
from torch.utils.data import DataLoader
from configs.default_config import config

def run_diagnostic():
    # Setup data
    data_dir = Path("data/")
    audio_files = list(data_dir.glob("*.mp3"))
    
    if len(audio_files) == 0:
        raise ValueError(f"No mp3 files found in {data_dir}")
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Create dataset and dataloader
    dataset = AudioDataset(audio_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Get first batch
    batch = next(iter(dataloader))
    
    # Run diagnostic
    results = trainer.diagnostic_run(batch)
    
    # Save input and output audio for comparison
    trainer.save_diagnostic_audio(
        batch['audio'], 
        "diagnostic_input.wav"
    )
    trainer.save_diagnostic_audio(
        results['generated_speech'], 
        "diagnostic_output.wav"
    )
    
    print("\n=== Diagnostic Complete ===")
    print("Saved audio files:")
    print("- diagnostic_input.wav")
    print("- diagnostic_output.wav")
    print("\nPlease check these files to verify audio quality")

if __name__ == "__main__":
    run_diagnostic() 