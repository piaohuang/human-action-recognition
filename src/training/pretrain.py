# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:20:26 2025

@author: admin
"""

import torch
from data.dataset_pretrain import ContrastiveVideoDataset
from models.simclr import ContrastiveFrameEncoder, ContrastiveLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
def train_encoder(video_dir, epochs=50):
    """Self-supervised encoder pretraining"""
    # 1. Prepare dataset of video frames with augmentations
    dataset = ContrastiveVideoDataset(video_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=6)
    
    # 2. Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveFrameEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ContrastiveLoss()
    early_stopper = EarlyStopping(patience=10, path='pretrained_encoder.pth')

    
    # 3. Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        pos_sims = []
        neg_sims = []
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
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
            
            # # Gradient debug every 100 batches
            # if batch_idx % 10 == 0:
            #     grads = [p.grad.abs().mean().item() 
            #              for p in model.parameters() if p.grad is not None]
            #     print(f"Batch {batch_idx} | Avg gradient: {np.mean(grads):.2e}")
    
        # Epoch statistics
        avg_pos = np.mean(pos_sims)
        avg_neg = np.mean(neg_sims)
        avg_loss = total_loss/len(loader)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Pos: {avg_pos:.3f} | "
              f"Neg: {avg_neg:.3f} | "
              f"Ratio: {avg_pos/(avg_neg+1e-8):.1f}x")
        end_time = time.time()
        print(f"Execution time: {(end_time - start_time):.2f} seconds")

        early_stopper(avg_loss, model)
    
        if early_stopper.early_stop:
            break

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  # lower val_loss is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at patience = {self.patience}")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)    