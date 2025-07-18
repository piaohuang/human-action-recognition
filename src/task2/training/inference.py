# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:02:16 2025

@author: admin
"""
import torch
from data.SimCLR_pretrain import ContrastiveFrameEncoder, ContrastiveLoss
from models.transformer import VideoTransformer
from models.lstm import VideoClassifier
from torchvision import transforms
import cv2
class ActionRecognizer:
    def __init__(self, encoder_path, model_type, classifier_path, class_names):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = class_names
        
        # Load pretrained encoder
        self.encoder = ContrastiveFrameEncoder().to(self.device)
        
        self.encoder.load_state_dict(torch.load(encoder_path)['model_state_dict'])
        self.encoder.eval()
        
        # Load model
        if model_type=='transformer':
            self.classifier = VideoTransformer(
                feature_dim=128,
                num_classes=len(class_names)-1
            ).to(self.device)
        elif model_type=='lstm':
            self.classifier = VideoClassifier(
                input_dim=128,
                hidden_dim=256,
                num_classes=len(class_names) -1 ).to(self.device)

        self.classifier.load_state_dict(torch.load(classifier_path)['model_state_dict'])
        self.classifier.eval()
        
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
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")
        else:
            print('predic for file:', video_path)
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
        features = torch.stack(features)
        features = features.reshape(1, features.shape[0], features.shape[1] * features.shape[2]).to(self.device)
        # Classify sequence
        with torch.no_grad():
            logits = self.classifier(features)
            pred = (logits > 0.5).float()
        
        return "Walking" if pred == 1 else "Standing"