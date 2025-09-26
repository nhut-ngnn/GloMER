
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pickle
from transformers import Wav2Vec2Model
from torch.utils.data import Dataset

class AudioEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        projected = self.projection(pooled)
        return pooled, projected


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z_all = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z_all, z_all.t()) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool).to(sim.device)
        sim.masked_fill_(mask, -float('inf'))
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(sim.device)
        return F.cross_entropy(sim, labels)


class AudioDataset(Dataset):
    def __init__(self, pkl_path, processor, segment_length=16000):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f) 
        self.processor = processor
        self.segment_length = segment_length

    def __len__(self):
        return len(self.data)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze()

    def _get_segment(self, waveform):
        if waveform.size(0) > self.segment_length:
            start = torch.randint(0, waveform.size(0) - self.segment_length, (1,))
            waveform = waveform[start:start + self.segment_length]
        else:
            waveform = F.pad(waveform, (0, self.segment_length - waveform.size(0)))
        return waveform

    def __getitem__(self, idx):
        audio_path, _, _ = self.data[idx]
        waveform = self._load_audio(audio_path)
        seg1 = self._get_segment(waveform)
        seg2 = self._get_segment(waveform)

        proc1 = self.processor(seg1.numpy(), sampling_rate=16000, return_tensors="pt")["input_values"].squeeze()
        proc2 = self.processor(seg2.numpy(), sampling_rate=16000, return_tensors="pt")["input_values"].squeeze()

        return {
            "input_values1": proc1,
            "input_values2": proc2
        }