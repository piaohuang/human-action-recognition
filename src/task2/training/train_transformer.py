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
from training.train_encoder import evaluate_model
def train_transformer(features_dir, labels, epochs=50, patience=5):
    # 1. Prepare dataset
    dataset = FeatureDataset(features_dir, labels)
    train_loader, val_loader = create_dataloaders(dataset)
    
    # 2. Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoTransformer(
        feature_dim=128,  # Must match encoder output
        num_classes=len(set(dataset.labels))-1
    ).to(device)
    
    # 3. Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_val_loss = float('inf')
    patience_counter = 0
    # 4. Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            loss = criterion(outputs, labels.float())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100 * correct / total
        train_loss /= len(train_loader)
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, 'transformer_classifier_checkpoint.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    