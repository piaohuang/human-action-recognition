# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:04:57 2025

@author: admin
"""
import torch
from data.dataset_feature import create_dataloaders
from models.gru import SkeletonActionClassifier
from models.lstm import VideoClassifier
from models.transformer import VideoTransformer
from training.train import train_classifier
from training.pretrain import train_encoder
from data.extact_feature import VideoFeatureExtractor


# from training.train_encoder import train_encoder
# from data.SimCLR_pretrain import ContrastiveFrameEncoder
# from training.train_transformer import train_transformer
from inference.inference import ActionRecognizer
from pathlib import Path 
import os

pretrain_video_dir = Path(r"F:\Piao\human-action-recognition\data\UCF-101")
video_dir = Path(r"F:\Piao\human-action-recognition\data\videos\train")
skeleton_dir = Path(r"F:\Piao\human-action-recognition\data\keypoints")
feature_dir = Path(r"F:\Piao\human-action-recognition\data\features")
dev_video = Path(r"F:\Piao\human-action-recognition\data\videos\dev\test3.mp4")
checkpoint_dir = Path(r"F:\Piao\human-action-recognition\src\checkpoints")
paths = {
    'pretrain': checkpoint_dir / 'pretrained_encoder.pth',
    'gru': checkpoint_dir / 'gru_classifier.pth',
    'lstm': checkpoint_dir / 'lstm_classifier.pth',
    'transformer': checkpoint_dir / 'transformer_classifier.pth'}

# ---------- Task 1: Skeleton + GRU ----------
def run_skeleton_gru():
    train_loader, val_loader = create_dataloaders('skeleton', skeleton_dir)
    model = SkeletonActionClassifier(input_size=33*2, hidden_size=64)
    train_classifier('gru', model, train_loader, val_loader, epochs=100, patience=10)

    recognizer = ActionRecognizer(model_type='gru', classifier_path=paths['gru'])
    result = recognizer.predict(dev_video)
    print(f"[GRU-Skeleton] Predicted action: {result}")

# ---------- Task 2: Self-supervised Encoder + LSTM ----------
def run_pretrain_encoder():
    train_encoder(video_dir=pretrain_video_dir, epochs=20)
    
def run_feature_extract():
    # Extract features
    feature_extractor = VideoFeatureExtractor(paths['pretrain'])
    for filename in os.listdir(video_dir):
        if filename.lower().endswith('.mp4'):
            name = Path(filename).stem
            save_path = feature_dir / f"{name}.npy"
            video_path = video_dir / filename
            feature_extractor.extract_features(video_path, save_path)
    
def run_pretrain_lstm():
    #  Train LSTM classifier
    train_loader, val_loader = create_dataloaders('pretrain', feature_dir)
    model = VideoClassifier(input_dim=128, hidden_dim=256)
    train_classifier('lstm', model, train_loader, val_loader, epochs=200, patience=10)

    #Inference
    recognizer = ActionRecognizer(
        model_type='lstm',
        classifier_path=paths['lstm'],
        encoder_path=paths['pretrain'],
        class_names=[0, 1]
    )
    result = recognizer.predict(dev_video)
    print(f"[LSTM-Pretrain] Predicted action: {result}")
    
    
# ---------- Task 3: Self-supervised Encoder + Transformer ----------
def run_pretrain_transformer():
    train_loader, val_loader = create_dataloaders('pretrain', feature_dir)
    model = VideoTransformer(feature_dim=128, num_classes=len(set(train_loader.dataset.labels))-1)
    train_classifier('transformer', model, train_loader, val_loader, epochs=200, patience=20)

    recognizer = ActionRecognizer(
        model_type='transformer',
        classifier_path=paths['transformer'],
        encoder_path=paths['pretrain'],
        class_names=[0, 1]
    )
    result = recognizer.predict(dev_video)
    print(f"[Transformer-Pretrain] Predicted action: {result}")

# ---------- Entry ----------
if __name__ == "__main__":
    # run_skeleton_gru()
    run_pretrain_encoder()
    # run_feature_extract()
    #run_pretrain_lstm()
    # run_pretrain_transformer()
    
