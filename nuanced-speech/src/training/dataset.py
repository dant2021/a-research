import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_files):
        self.audio_files = audio_files
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        return {
            'audio': waveform,
            'path': str(audio_path)
        }