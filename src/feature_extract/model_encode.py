import torch
from torch import nn
from transformers import BertModel, Wav2Vec2Model

class BERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.project = nn.Sequential(  # Fine-tuned model expects 'project'
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        projected = self.project(pooled_output)
        return pooled_output, projected


class AudioEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.projection = nn.Sequential(  # Keep this as 'projection' to match checkpoint
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values=input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        projected = self.projection(pooled_output)
        return pooled_output, projected
