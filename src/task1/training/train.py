# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:52:48 2025

@author: admin
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, test_loader, num_epochs=50, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()  # Binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            sequences, labels = sequences.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
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
        val_loss, val_acc = evaluate_model(model, test_loader, device, criterion)
        
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
        }, 'gru_classifier_checkpoint.pth')
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


def predict(model, skeleton_sequence, device='cpu'):
    """Predict whether sequence shows walking (1) or standing (0)"""
    model.eval()
    model.to(device)
    
    # Preprocess sequence
    sequence = torch.FloatTensor(skeleton_sequence)
    sequence = sequence.reshape(sequence.shape[0], -1)

    sequence = (sequence - 0.5) * 2  # Normalize
    # Pad/truncate to match training length
    if len(sequence) > 30:
        sequence = sequence[:30]
    else:
        padding = np.zeros((30 - len(sequence), sequence.shape[1]))
        sequence = np.vstack([sequence, padding])
        
    sequence = sequence.to(device)
    
    with torch.no_grad():
        output = model(sequence)
        prediction = (output > 0.5).float().item()
    
    return "Walking" if prediction == 1 else "Standing"


