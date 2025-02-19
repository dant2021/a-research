import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_files, chunk_duration=30.0, sample_rate=16000):
        self.audio_files = audio_files
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.audio_files[idx])
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Get 30-second chunk
        if waveform.size(1) > self.chunk_samples:
            start = torch.randint(0, waveform.size(1) - self.chunk_samples, (1,))
            waveform = waveform[:, start:start + self.chunk_samples]
        else:
            waveform = torch.nn.functional.pad(
                waveform, 
                (0, self.chunk_samples - waveform.size(1))
            )
        
        return {'audio': waveform} 