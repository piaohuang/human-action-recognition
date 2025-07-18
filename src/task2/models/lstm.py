# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:11 2025

@author: admin
"""
import torch.nn as nn
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep's output
        last_out = lstm_out[:, -1, :]
        output = self.classifier(last_out).squeeze(1)
    
        return output