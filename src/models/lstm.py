# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:11 2025

@author: admin
"""
import torch.nn as nn
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # *2 for bidirectional
            nn.LeakyReLU(0.1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            )
        
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = (x - x.mean(dim=(0,1))) / (x.std(dim=(0,1)) + 1e-8)
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep's output
        last_out = lstm_out[:, -1, :]
        output = self.classifier(last_out).squeeze(1)
    
        return output