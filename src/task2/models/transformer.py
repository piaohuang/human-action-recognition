# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:57:52 2025

@author: admin
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
class VideoTransformer(nn.Module):
    def __init__(self, feature_dim=128, num_classes=2, 
                 nhead=8, num_layers=3, dim_feedforward=512):
        super().__init__()
        
        # Positional encoding for temporal dimension
        self.positional_encoding = PositionalEncoding(feature_dim)
        
        # Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim_feedforward, num_classes)
            )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
        Returns:
            (batch_size, num_classes)
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer expects (seq_len, batch_size, feature_dim) by default
        # But with batch_first=True, we keep (batch, seq, features)
        transformer_out = self.transformer(x)
        
        # Use [CLS] token or average pooling
        pooled = transformer_out.mean(dim=1)  # Average over time
        
        return self.classifier(pooled)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return x