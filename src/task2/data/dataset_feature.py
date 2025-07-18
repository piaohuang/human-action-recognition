# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:41:38 2025

@author: admin
"""
import os
from torch.utils.data import DataLoader, Subset, Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class FeatureDataset(Dataset):
    def __init__(self, features_dir, labels, seq_length=140):
        self.feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) 
                            if f.endswith('.npy')]
        print(self.feature_files)
        self.labels = labels if labels is not None else [1 if "walk" in item else 0 for item in os.listdir(features_dir)]
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        features = np.load(self.feature_files[idx])  # (seq_len, feature_dim)
        features = features.reshape(features.shape[0], -1)
        # Pad/truncate to fixed length
        if len(features) > self.seq_length:
            features = features[:self.seq_length]
        else:
            padding = np.zeros((self.seq_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])
            
        label = self.labels[idx]
        return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)


def create_dataloaders(dataset, batch_size=1, test_size=0.5, random_state=1):
    """Create train and validation dataloaders"""
    # Split dataset
    labels = [label for _, label in dataset]
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(labels, train_indices,val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle validation
    )
    
    return train_loader, val_loader
