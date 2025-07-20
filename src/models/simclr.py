# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:14:45 2025

@author: admin
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
class ContrastiveFrameEncoder(nn.Module):
    def __init__(self, feature_dim=128, pretrained=True):
        super().__init__()
        # Use ResNet as backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()  # Remove final classification layer
        # Freeze all layers
        if pretrained is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
            # Unfreeze only last residual block
            for param in self.encoder.layer4.parameters():
                param.requires_grad = True
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, feature_dim)
            )
        
    def forward(self, x):
        features = self.encoder(x)
        projected = self.projection(features)
        return F.normalize(projected, dim=1)

# Contrastive loss (NT-Xent)
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """Compute loss for positive pair (z_i, z_j)"""
        batch_size = z_i.size(0)
        
        # Concatenate all features
        z_i = F.normalize(z_i, dim=1)  
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
       # Compute similarity matrix
        sim_raw = torch.mm(z, z.t())
        sim = torch.mm(z, z.t()) / self.temperature
        
        # # Paires positives (diagonales décalées)
        pos_pairs = torch.cat([
            torch.diag(sim_raw[:batch_size, batch_size:]),
            torch.diag(sim_raw[batch_size:, :batch_size])
        ])
        
        # Paires négatives (masquer les paires positives)
        neg_mask = ~torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
        neg_pairs = sim_raw[neg_mask].view(2*batch_size, -1)
        
        # Calcul des métriques
        avg_pos_sim = pos_pairs.mean().item()
        avg_neg_sim = neg_pairs.mean().item()
        pos_neg_ratio = avg_pos_sim / (avg_neg_sim + 1e-8)
        
        
        
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
        
        return loss,avg_pos_sim,avg_neg_sim