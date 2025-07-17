# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:02:16 2025

@author: admin
"""
import torch
from ..data.SimCLR_pretrain import ContrastiveFrameEncoder, ContrastiveLoss
from ..models.transformer import VideoTransformer
from torchvision import transforms
import cv2
class TransformerActionRecognizer:
    def __init__(self, encoder_path, transformer_path, class_names):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = class_names
        
        # Load pretrained encoder
        self.encoder = ContrastiveFrameEncoder().to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        
        # Load transformer
        self.transformer = VideoTransformer(
            feature_dim=128,
            num_classes=len(class_names)
        ).to(self.device)
        self.transformer.load_state_dict(torch.load(transformer_path))
        self.transformer.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, video_path):
        # Extract frame features
        features = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = self.transform(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.encoder(frame)
            features.append(feat.cpu())
        
        cap.release()
        
        # Stack features (seq_len, feature_dim) and add batch dim
        features = torch.stack(features).unsqueeze(0).to(self.device)
        
        # Classify sequence
        with torch.no_grad():
            logits = self.transformer(features)
            pred = torch.argmax(logits, dim=1).item()
        
        return self.class_names[pred]