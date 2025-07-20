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
import torch.nn.functional as F
class FeatureDataset(Dataset):
    def __init__(self, method, npy_files, labels, seq_length=30):
        self.npy_files = npy_files
        self.labels = labels
        self.seq_length = seq_length
        self.method = method
        
    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):
        feature = np.load(self.npy_files[idx]) 
        feature = feature.reshape(feature.shape[0], -1)
        
            
        # Pad/truncate to fixed length
        if len(feature) > self.seq_length:
            feature = feature[:self.seq_length]
        else:
            padding = np.zeros((self.seq_length - len(feature), feature.shape[1]))
            feature = np.vstack([feature, padding])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        feature = torch.FloatTensor(feature)
        
        if(self.method == 'skeleton'):
            # Normalize coordinates to [-1, 1] range
            feature = (feature - 0.5) * 2
        
        return feature, label


def create_dataloaders(method, data_dir, batch_size=1, test_size=0.5, random_state=123):
    """Prepare train/test loaders for walk/stand classification"""
    # Assuming naming convention: walk_*.npy and stand_*.npy
    walk_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.startswith('walk') and f.endswith('.npy')]
    stand_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.startswith('stand') and f.endswith('.npy')]
    
    # Create labels (0 for stand, 1 for walk)
    files = walk_files + stand_files
    labels = [1.]*len(walk_files) + [0.]*len(stand_files)
    
    # Split into train/test (since we have few samples, use 50-50 split)
    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=test_size, random_state=42, stratify=labels
    )
    print(train_files,test_files, train_labels, test_labels)
    # Create datasets
    train_dataset = FeatureDataset(method, train_files, train_labels)
    test_dataset = FeatureDataset(method, test_files, test_labels)
    
    # Since we have very few samples, use batch_size=1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
