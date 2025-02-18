import torch
from torch.utils.data import Dataset
from src.utils.audio import load_audio, process_length, compute_spectrogram
import torchaudio
   
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
        
        # Compute mel spectrogram
        mel_spec = compute_spectrogram(waveform)
        
        return {
            'audio': waveform,
            'mel_spec': mel_spec,
            'path': str(audio_path)
        }