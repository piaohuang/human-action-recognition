# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:26 2025

@author: admin
"""
import torch.nn as nn
import torch
from data.dataset_feature import FeatureDataset, create_dataloaders
from models.lstm import VideoClassifier
from training.train_encoder import evaluate
def train_lstm(features_dir, labels, epochs=30):
    """Train LSTM classifier on extracted features"""
    # 1. Prepare dataset
    dataset = FeatureDataset(features_dir, labels)
    train_loader, val_loader = create_dataloaders(dataset)
    
    # 2. Initialize model
    model = VideoClassifier(
        input_dim=128,  # Match encoder feature_dim
        hidden_dim=256,
        num_classes=len(set(labels))).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), "lstm_classifier.pth")
    