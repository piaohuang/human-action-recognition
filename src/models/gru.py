# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:51:42 2025

@author: admin
"""
import torch.nn as nn

class SkeletonActionClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=64, num_layers=1):
        """
        Binary classifier for walk vs stand
        Args:
            input_size: 33 keypoints * 2 coordinates = 66
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        gru_out, _ = self.gru(x)  # output shape: (batch_size, seq_length, hidden_size*2)
        
        # Take the last timestep's output
        last_out = gru_out[:, -1, :]
        
        return self.fc(last_out).squeeze(1)  # Output between 0-1