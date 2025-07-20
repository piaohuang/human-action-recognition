# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:26 2025

@author: admin
"""
import torch.nn as nn
import torch
from tqdm import tqdm
from data.dataset_feature import FeatureDataset, create_dataloaders
from models.lstm import VideoClassifier
import os
def train_classifier(model_type, model, train_loader, val_loader, epochs=50, patience=10, save_path = ""):
    """Train LSTM classifier on extracted features"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    if(model_type == 'transformer'): 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"[Frozen] {name}")
  
    # 3. Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels = features.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(features)
         
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient check
            total_grad = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean()
                    total_grad += grad_mean
                    if grad_mean < 1e-8:
                        print(f"Vanishing gradient in {name}")
                  
            print(f"Loss={loss.item():.4f}, Avg Grad={total_grad/len(list(model.parameters())):.6f}")
    
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
            torch.save(model.state_dict(),
            os.path.join(save_path, model_type + '_classifier.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.float().to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total    