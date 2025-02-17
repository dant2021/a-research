import torch
import torchaudio
import torchaudio.transforms as T

def load_audio(audio_path, target_sr=16000):
    """Load and resample audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform

def process_length(audio, target_length):
    """Pad or truncate audio to target length."""
    current_length = audio.size(-1)
    if current_length > target_length:
        return audio[..., :target_length]
    elif current_length < target_length:
        padding = torch.zeros(audio.size(0), target_length - current_length)
        return torch.cat([audio, padding], dim=-1)
    return audio

def compute_spectrogram(audio, n_mels=80):
    """Compute log-mel spectrogram."""
    mel_spec = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=2048,
        win_length=400,  # 25ms at 16kHz
        hop_length=200,  # 12.5ms at 16kHz
    )
    
    spec = mel_spec(audio)
    return torch.log(spec + 1e-9) 