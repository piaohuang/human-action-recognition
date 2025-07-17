# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:14:45 2025

@author: admin
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class ContrastiveFrameEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # Use ResNet as backbone
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove final classification layer
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
            )
        
    def forward(self, x):
        features = self.encoder(x)
        return self.projection(features)

# Contrastive loss (NT-Xent)
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """Compute loss for positive pair (z_i, z_j)"""
        batch_size = z_i.size(0)
        
        # Concatenate all features
        z_i = F.normalize(z_i, p=2, dim=1)  # L2-normalization
        z_j = F.normalize(z_j, p=2, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
       # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Paires positives (diagonales décalées)
        pos_pairs = torch.cat([
            torch.diag(sim[:batch_size, batch_size:]),
            torch.diag(sim[batch_size:, :batch_size])
        ])
        
        # Paires négatives (masquer les paires positives)
        neg_mask = ~torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
        neg_pairs = sim[neg_mask].view(2*batch_size, -1)
        
        # Calcul des métriques
        avg_pos_sim = pos_pairs.mean().item()
        avg_neg_sim = neg_pairs.mean().item()
        pos_neg_ratio = avg_pos_sim / (avg_neg_sim + 1e-8)
        
        # Logging (à adapter selon votre framework)
        # print(
        #     f"[Debug] Pos: {avg_pos_sim:.3f} | "
        #     f"Neg: {avg_neg_sim:.3f} | "
        #     f"Ratio: {pos_neg_ratio:.1f}x"
        # )
        
        # Create similarity labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(2*batch_size, dtype=torch.bool)
        sim = sim[mask].view(2*batch_size, -1)
        labels = labels[mask].view(2*batch_size, -1)
        
        # Compute log probabilities
        logits = torch.exp(sim)
        log_prob = sim - torch.log(logits.sum(dim=1, keepdim=True))
        
        # Compute loss
        loss = - (labels * log_prob).sum(dim=1).mean()
       # return loss
    # Ajoutez ce terme pour éviter le collapse
        # cov_matrix = torch.mm(z_i.t(), z_j) / batch_size
        # off_diag = cov_matrix.flatten()[:-1].view(cov_matrix.size(0)-1, cov_matrix.size(0)+1)[:-1]
        # diversity_loss = off_diag.pow_(2).sum() / z_i.size(1)
        
        return loss,avg_pos_sim,avg_neg_sim  #+ 0.01 * diversity_loss  # Poids ajustable