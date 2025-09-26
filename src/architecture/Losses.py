import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2, group=None):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        if group is not None:
            mask = group.unsqueeze(1) == group.unsqueeze(0)
        else:
            mask = torch.ones((z1.size(0), z2.size(0)), dtype=torch.bool, device=z1.device)

        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        sim_matrix = sim_matrix.masked_fill(~mask, float('-inf'))

        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class DiversityContrastiveLoss(nn.Module):
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        diff = z1 - z2
        return torch.mean(torch.norm(diff, dim=1))


class ConsistencyContrastiveLoss(nn.Module):
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        cosine_sim = F.cosine_similarity(z1, z2, dim=1)
        return 1.0 - cosine_sim.mean()
    