# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:02:16 2025

@author: admin
"""
import torch
from models.simclr import ContrastiveFrameEncoder, ContrastiveLoss
from models.transformer import VideoTransformer
from models.lstm import VideoClassifier
from models.gru import SkeletonActionClassifier
from torchvision import transforms
import cv2
import mediapipe as mp
import numpy as np
import torch.nn.functional as F
class ActionRecognizer:
    def __init__(self, model_type, classifier_path, encoder_path = None, class_names = ['walk', 'stand']):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load pretrained encoder
        if encoder_path is not None:
            self.pretrain = True
            self.encoder = ContrastiveFrameEncoder().to(self.device)
            
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.encoder.eval()
            
            # Image transformations
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.pretrain = False
        
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
                ).to(self.device)
        elif model_type=='gru':
            self.classifier = SkeletonActionClassifier(
                input_size=33*2,  # 33 keypoints * 2 coordinates
                hidden_size=64,
                num_layers=1
            )

        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()
        self.classifier.to(self.device)

    
    def predict(self, video_path, seq_length=30):
        if(self.pretrain==False):
            # Initialize MediaPipe Pose
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8)
        # Extract frame features
        features = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"can not open video: {video_path}")
        else:
            print('predict for file:', video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if(self.pretrain):
                frame = self.transform(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.encoder(frame)
                features.append(feat.cpu())
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        keypoints.append([landmark.x * frame.shape[1], landmark.y * frame.shape[0]])
                    features.append(np.array(keypoints))  # Shape: (33, 2)
                
        
        cap.release()
    
        
        if(self.pretrain):
            # Stack features (seq_len, feature_dim) and add batch dim
            features = torch.stack(features)
           # features = features.reshape(1,features.shape[0], features.shape[1] * features.shape[2]).to(self.device)
            features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).to(self.device)

        else:
            features = torch.FloatTensor(features)
            features = features.reshape(features.shape[0], -1)
            features = (features - 0.5) * 2
            
        # Pad/truncate to match training length
        if len(features) > seq_length:
            features = features[:seq_length]
        else:  
            padding = np.zeros((seq_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])
        
        if(self.pretrain):
            features = features.reshape(1,features.shape[0], features.shape[1]).to(self.device)
        features = features.to(self.device)
        # Classify sequence
        with torch.no_grad():
            logits = self.classifier(features)
            pred = (logits > 0.5).float()
        
        return "Walking" if pred == 1 else "Standing"