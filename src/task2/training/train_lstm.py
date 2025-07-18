# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:26 2025

@author: admin
"""
import torch.nn as nn
import torch
from data.dataset_feature import FeatureDataset, create_dataloaders
from models.lstm import VideoClassifier
from training.train_encoder import evaluate_model
def train_lstm(features_dir, labels, epochs=30, patience=100):
    """Train LSTM classifier on extracted features"""
    # 1. Prepare dataset
    dataset = FeatureDataset(features_dir, labels)
    train_loader, val_loader = create_dataloaders(dataset)
    
    # 2. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoClassifier(
        input_dim=128,  # Match encoder feature_dim
        hidden_dim=256,
        num_classes=len(set(dataset.labels))).to(device)
    
    criterion = nn.BCELoss()  # Binary cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(features)
         
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
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
        }, 'lstm_classifier_checkpoint.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    