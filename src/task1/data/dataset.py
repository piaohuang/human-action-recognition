# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:40:29 2025

@author: admin
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

class SkeletonSequenceDataset(Dataset):
    def __init__(self, npy_files, labels, seq_length=30):
        """
        Args:
            npy_files: List of paths to .npy files
            labels: Corresponding labels (0 for stand, 1 for walk)
            seq_length: Fixed sequence length
        """
        self.npy_files = npy_files
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):
        # Load skeleton sequence (shape: [frames, 33*2 coordinates])
        sequence = np.load(self.npy_files[idx])
        sequence = sequence.reshape(sequence.shape[0], -1)
        # Normalize coordinates to [-1, 1] range
        sequence = (sequence - 0.5) * 2
        
        # Pad/truncate sequence
        if len(sequence) > self.seq_length:
            sequence = sequence[:self.seq_length]
        else:
            padding = np.zeros((self.seq_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
            
        return torch.FloatTensor(sequence), torch.tensor(self.labels[idx], dtype=torch.long)

def prepare_data(data_dir):
    """Prepare train/test loaders for walk/stand classification"""
    # Assuming naming convention: walk_*.npy and stand_*.npy
    walk_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.startswith('walk') and f.endswith('.npy')]
    stand_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.startswith('stand') and f.endswith('.npy')]
    
    # Create labels (0 for stand, 1 for walk)
    files = walk_files + stand_files
    labels = [1]*len(walk_files) + [0]*len(stand_files)
    
    # Split into train/test (since we have few samples, use 75-25 split)
    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=0.5, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SkeletonSequenceDataset(train_files, train_labels)
    test_dataset = SkeletonSequenceDataset(test_files, test_labels)
    
    # Since we have very few samples, use batch_size=1
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    return train_loader, test_loader