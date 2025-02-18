import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_files, chunk_duration=30.0, sample_rate=24000):
        self.audio_files = audio_files
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Get total chunks
        total_chunks = waveform.size(-1) // self.chunk_samples
        
        # Randomly select a chunk
        chunk_idx = torch.randint(0, total_chunks, (1,))
        start_idx = chunk_idx * self.chunk_samples
        chunk = waveform[..., start_idx:start_idx + self.chunk_samples]
        
        return {
            'audio': chunk,
            'target_audio': chunk,  # Same audio for now
            'text': None  # We'll get this from Whisper
        }