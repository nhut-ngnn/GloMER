import torch
import torch.nn as nn
import numpy as np
from .Projection_head import ProjectionHead
from .Cross_model import CrossModalEncoders
from .Classifier import MLPClassifier
from .GatedMultimodalUnit import GatedMultimodalUnit

class GloMER(nn.Module):
    def __init__(
        self,
        text_input_dim=768,
        audio_input_dim=768,
        fusion_dim=512,
        projection_dim=512,
        num_heads=4,
        dropout=0.3,
        linear_layer_dims=[512, 256],
        num_classes=4
    ):
        super().__init__()

        self.encoders = CrossModalEncoders(
            text_input_dim, audio_input_dim, fusion_dim, dropout, num_heads
        )

        self.gmu = GatedMultimodalUnit(
            text_dim=fusion_dim,
            audio_dim=fusion_dim,
            fusion_dim=fusion_dim
        )

        self.shared_proj = ProjectionHead(input_dim=fusion_dim, projection_dim=projection_dim)

        self.classifier = MLPClassifier(
            input_dim=fusion_dim,
            layer_dims=linear_layer_dims,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, text_feat, audio_feat, return_cls=False, return_all=False):
        text_attn, audio_attn = self.encoders(text_feat, audio_feat)

        text_pooled = text_attn.mean(dim=1)
        audio_pooled = audio_attn.mean(dim=1)
        
        fusion_vec = self.gmu(text_pooled, audio_pooled)

        text_proj_vec = self.shared_proj(text_pooled)
        audio_proj_vec = self.shared_proj(audio_pooled)

        if return_cls:
            return self.classifier(fusion_vec)

        if return_all:
            return {
                "text_pool": text_pooled,
                "audio_pool": audio_pooled,
                "text_proj": text_proj_vec,
                "audio_proj": audio_proj_vec,
                "fusion": fusion_vec,
                "logits": self.classifier(fusion_vec)
            }

        return text_proj_vec, audio_proj_vec
