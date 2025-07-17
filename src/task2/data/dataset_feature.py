# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:41:38 2025

@author: admin
"""
import os
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import numpy as np

class FeatureDataset(Dataset):
    def __init__(self, features_dir, labels, seq_length=140):
        self.feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) 
                            if f.endswith('.npy')]
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        features = np.load(self.feature_files[idx])  # (seq_len, feature_dim)
        
        # Pad/truncate to fixed length
        if len(features) > self.seq_length:
            features = features[:self.seq_length]
        else:
            padding = np.zeros((self.seq_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])
            
        label = self.labels[idx]
        return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)


def create_dataloaders(dataset, batch_size=16, test_size=0.5):
    """Create train and validation dataloaders"""
    # Split dataset
    train_size = int((1 - test_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False)
    
    return train_loader, val_loader
