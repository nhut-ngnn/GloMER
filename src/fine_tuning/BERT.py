import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertModel
import pickle
import numpy as np

class BERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return pooled, self.project(pooled)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2*N, dtype=torch.bool).to(sim.device)
        sim.masked_fill_(mask, -float('inf'))
        targets = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(sim.device)
        return F.cross_entropy(sim, targets)

class TextDataset(Dataset):
    def __init__(self, pkl_path, tokenizer, max_length=128):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def _augment(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
        keep_prob = 0.85
        words = [w for w in words if np.random.rand() < keep_prob]
        return ' '.join(words) if words else text

    def __getitem__(self, idx):
        _, text, _ = self.data[idx]
        aug1 = self._augment(text)
        aug2 = self._augment(text)

        enc1 = self._tokenize(aug1)
        enc2 = self._tokenize(aug2)

        return {
            'input_ids1': enc1['input_ids'].squeeze(),
            'attention_mask1': enc1['attention_mask'].squeeze(),
            'input_ids2': enc2['input_ids'].squeeze(),
            'attention_mask2': enc2['attention_mask'].squeeze(),
        }