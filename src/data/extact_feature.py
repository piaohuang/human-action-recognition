# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:18:53 2025

@author: admin
"""
import torch
import cv2
from models.simclr import ContrastiveFrameEncoder
from torchvision.transforms import transforms
import numpy as np
class VideoFeatureExtractor:
    def __init__(self, encoder_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.encoder = ContrastiveFrameEncoder().to(device)
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, video_path, save_path=None):
        """Extract frame features from video
        Args:
        video_path: Path to input video file
        save_path: (Optional) Path to save features as .npy file
        
    Returns:
        torch.Tensor of shape (num_frames, feature_dim)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
    
        features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                feat = self.encoder(frame)
            features.append(feat.cpu())
            
        cap.release()
        
        if save_path:
            np.save(save_path, (torch.stack(features)).numpy())
            print(f"Features saved to {save_path}")
        return torch.stack(features)  # (seq_len, feature_dim)