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
import torch.nn as nn


# from training.train_encoder import train_encoder
# from data.SimCLR_pretrain import ContrastiveFrameEncoder
# from training.train_transformer import train_transformer
from inference.inference import ActionRecognizer
from pathlib import Path 
import os
import argparse

BASE_DIR = Path(__file__).parent.parent.resolve()

pretrain_video_dir = BASE_DIR / "data" / "UCF-101"
video_dir = BASE_DIR / "data" / "videos" / "train"
skeleton_dir = BASE_DIR / "data" / "keypoints"
feature_dir = BASE_DIR / "data" / "features"
dev_video = BASE_DIR / "data" / "videos" / "dev" / "test1.mp4"
checkpoint_dir = BASE_DIR / "src" / "checkpoints"
paths = {
    'pretrain': checkpoint_dir / 'pretrained_encoder.pth',
    'gru': checkpoint_dir / 'gru_classifier.pth',
    'lstm': checkpoint_dir / 'lstm_classifier.pth',
    'transformer': checkpoint_dir / 'transformer_classifier.pth'
}


def run_skeleton_gru(train, inference):
    if(train):
        train_loader, val_loader = create_dataloaders('skeleton', skeleton_dir)
        model = SkeletonActionClassifier(input_size=33*2, hidden_size=64)
        train_classifier('gru', model, train_loader, val_loader, epochs=100, patience=5, save_path = checkpoint_dir)
    if(inference):
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
    
def run_pretrain_lstm(train, inference):
    if(train):
        #  Train LSTM classifier
        train_loader, val_loader = create_dataloaders('pretrain', feature_dir)
        model = VideoClassifier(input_dim=128, hidden_dim=256)
        train_classifier('lstm', model, train_loader, val_loader, epochs=100, patience=5, save_path = checkpoint_dir )
    if(inference):
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
def run_pretrain_transformer(train, inference):
    if(train):
        train_loader, val_loader = create_dataloaders('pretrain', feature_dir)
        model = VideoTransformer(feature_dim=128, num_classes=len(set(train_loader.dataset.labels))-1)
        train_classifier('transformer', model, train_loader, val_loader, epochs=200, patience=200, save_path = checkpoint_dir)
    if(inference):
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
    parser = argparse.ArgumentParser(description="Human Action Recognition Tasks")
    parser.add_argument(
        "--task",
        type=str,
        choices=["skeleton_gru", "pretrain_encoder", "feature_extract", "pretrain_lstm", "pretrain_transformer"],
        required=True,
        help="Select the task to run"
    )
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--inference', action='store_true', help='Enable inference mode')
    args = parser.parse_args()
    
    if args.task == "skeleton_gru":
        run_skeleton_gru(args.train, args.inference)
    elif args.task == "pretrain_encoder":
        run_pretrain_encoder()
    elif args.task == "feature_extract":
        run_feature_extract()
    elif args.task == "pretrain_lstm":
        run_pretrain_lstm(args.train, args.inference)
    elif args.task == "pretrain_transformer":
        run_pretrain_transformer(args.train, args.inference)
    
