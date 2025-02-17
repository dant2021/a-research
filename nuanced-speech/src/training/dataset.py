import torch
from torch.utils.data import Dataset
from ..utils.audio import load_audio, process_length, compute_spectrogram

class SpeechDataset(Dataset):
    def __init__(self, audio_files, max_duration=30.0):
        self.audio_files = audio_files
        self.max_samples = int(max_duration * 16000)  # 16kHz sampling rate
        
    def __len__(self):
        return len(self.audio_files)
        
    def __getitem__(self, idx):
        # Load and preprocess audio
        audio = load_audio(self.audio_files[idx])
        audio = process_length(audio, self.max_samples)
        mel_spec = compute_spectrogram(audio)
        
        return {
            'audio': audio,
            'mel_spec': mel_spec,
            'audio_path': self.audio_files[idx]
        } 