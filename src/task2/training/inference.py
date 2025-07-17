# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:23:19 2025

@author: admin
"""
class VideoActionRecognizer:
    def __init__(self, encoder_path, classifier_path, class_names):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        self.encoder = ContrastiveFrameEncoder().to(self.device)
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.encoder.eval()
        
        self.classifier = VideoClassifier(
            input_dim=128,
            hidden_dim=256,
            num_classes=len(class_names)).to(self.device)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()
        
        self.class_names = class_names
        self.transform = # ... same as feature extractor ...
    
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
        
        # Stack features and add batch dim
        features = torch.stack(features).unsqueeze(0)  # (1, seq_len, 128)
        
        # Classify sequence
        with torch.no_grad():
            logits = self.classifier(features.to(self.device))
            pred = torch.argmax(logits, dim=1).item()
        
        return self.class_names[pred]