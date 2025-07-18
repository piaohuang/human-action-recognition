# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:26 2025

@author: admin
"""

import torch
from data.dataset_pretrain import ContrastiveVideoDataset
from data.SimCLR_pretrain import ContrastiveFrameEncoder, ContrastiveLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
def train_encoder(video_dir, epochs=50):
    """Self-supervised encoder pretraining"""
    # 1. Prepare dataset of video frames with augmentations
    dataset = ContrastiveVideoDataset(video_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveFrameEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ContrastiveLoss()
    
    # 3. Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        pos_sims = []
        neg_sims = []
        
        for batch_idx, batch in enumerate(loader):
            x_i, x_j = batch 
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            z_i = model(x_i)
            z_j = model(x_j)
            loss, pos_sim, neg_sim = criterion(z_i, z_j)
            
            loss.backward()
            optimizer.step()
            # Logging
            total_loss += loss.item()
            pos_sims.append(pos_sim)
            neg_sims.append(neg_sim)
            
            # Gradient debug every 100 batches
            if batch_idx % 10 == 0:
                grads = [p.grad.abs().mean().item() 
                         for p in model.parameters() if p.grad is not None]
                print(f"Batch {batch_idx} | Avg gradient: {np.mean(grads):.2e}")

        
    
        # Epoch statistics
        avg_pos = np.mean(pos_sims)
        avg_neg = np.mean(neg_sims)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(loader):.4f} | "
              f"Pos: {avg_pos:.3f} | "
              f"Neg: {avg_neg:.3f} | "
              f"Ratio: {avg_pos/(avg_neg+1e-8):.1f}x")
        end_time = time.time()
        print(f"Execution time: {(end_time - start_time):.2f} seconds")
    # Save pretrained encoder
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'pretrained_encoder_checkpoint.pth')
    
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