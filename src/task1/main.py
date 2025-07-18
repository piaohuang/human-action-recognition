# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:37:10 2025

@author: admin
"""
import torch
from data.dataset import prepare_data
from models.gru import SkeletonActionClassifier
from training.train import train_model, predict
from pathlib import Path 
import os
import numpy as np
if __name__ == "__main__":
    data_dir = Path(r"F:\Piao\human-action-recognition\data\task1\output_keypoints")
    train_loader, test_loader = prepare_data(data_dir)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Initialize model
    model = SkeletonActionClassifier(
        input_size=33*2,  # 33 keypoints * 2 coordinates
        hidden_size=64,
        num_layers=1
    )
    
    # 3. Train (with very few epochs due to small dataset)
    train_model(model, train_loader, test_loader, num_epochs=300, patience=3)
    
    # 4. Load best model for inference
    print(predict(model,  np.load(os.path.join(data_dir, 'test1_skeletons.npy'))))