# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:01:43 2025

@author: admin
"""
import torch
from data.dataset_feature import FeatureDataset, create_dataloaders
from data.SimCLR_pretrain import ContrastiveFrameEncoder, ContrastiveLoss
from models.transformer import VideoTransformer
import torch.nn as nn
from train_lstm import evaluate
def train_transformer(features_dir, labels, encoder_path, epochs=50):
    # 1. Prepare dataset
    dataset = FeatureDataset(features_dir, labels)
    train_loader, val_loader = create_dataloaders(dataset)
    
    # 2. Initialize models
    encoder = ContrastiveFrameEncoder().eval()  # Reuse from Task 2
    encoder.load_state_dict(torch.load(encoder_path))
    
    transformer = VideoTransformer(
        feature_dim=128,  # Must match encoder output
        num_classes=len(set(labels))
    ).cuda()
    
    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # 4. Training loop
    for epoch in range(epochs):
        transformer.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.cuda()
            batch_labels = batch_labels.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = transformer(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate(transformer, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.2f}%")
    
    torch.save(transformer.state_dict(), "video_transformer.pth")