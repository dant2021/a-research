import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, audio_files, chunk_duration=30.0, sample_rate=16000, batch_size=32, num_workers=4):
        self.audio_files = audio_files
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert stereo to mono by averaging channels if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Get total chunks
        total_chunks = waveform.size(-1) // self.chunk_samples
        if total_chunks == 0:
            waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - waveform.size(-1)))
            total_chunks = 1
        
        # Randomly select a chunk
        chunk_idx = torch.randint(0, total_chunks, (1,))
        start_idx = chunk_idx * self.chunk_samples
        chunk = waveform[..., start_idx:start_idx + self.chunk_samples]
        
        # Ensure shape is [samples] for Whisper (1D array)
        chunk = chunk.squeeze()  # Remove all singleton dimensions
        
        return {
            'audio': chunk,
            'target_audio': chunk,
            'text': None
        }
    
    @staticmethod
    def collate_fn(batch):
        # Filter out None values
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
            
        # Assuming each item in batch is a dictionary with 'audio' and 'target_audio' keys
        return {
            'audio': torch.stack([item['audio'] for item in batch]),
            'target_audio': torch.stack([item['target_audio'] for item in batch])
        }
    
    def get_dataloader(self, shuffle=True):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )