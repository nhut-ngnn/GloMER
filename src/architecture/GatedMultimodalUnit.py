import torch
import torch.nn as nn

class GatedMultimodalUnit(nn.Module):
    def __init__(self, text_dim, audio_dim, fusion_dim, dropout_p=0.2):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.gate_proj = nn.Linear(audio_dim + text_dim, fusion_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, text_feat, audio_feat):
        h_t = self.tanh(self.text_proj(text_feat))
        h_a = self.tanh(self.audio_proj(audio_feat))

        gate_input = torch.cat([text_feat, audio_feat], dim=1)
        z = self.sigmoid(self.gate_proj(gate_input))

        fused = (1-z) * h_t + z * h_a
        return fused
