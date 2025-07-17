# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:04:57 2025

@author: admin
"""
from training.train_encoder import train_encoder
from data.SimCLR_pretrain import ContrastiveFrameEncoder
from data.feature_extact import VideoFeatureExtractor
from training.train_lstm import train_lstm
import torch
from pathlib import Path 
# 1. First pretrain the encoder
data_dir = Path(r"F:\Piao\veesion\data\task1\raw\train")
# train_encoder(
#     video_dir=data_dir,
#     epochs=50
# )

# # 2. Then extract features using the trained encoder
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder =  ContrastiveFrameEncoder().to(device)
# encoder.load_state_dict(torch.load("pretrained_encoder.pth"))
# encoder.eval()

encorder_path = Path(r"F:\Piao\veesion\src\task2\output\pretrained_encoder.pth")
feature_extractor = VideoFeatureExtractor(encorder_path)
for filename in os.listdir(data_dir):
        file_path = os.path.join(root_path,'raw', filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in video_extensions:
                keypoints_sequence = extract_skeletons_from_video(file_path, os.path.join(root_path,'output_keypoints'))
      
feature_extractor.extract_features(
    data_dir,
    "extracted_features"
)

# 3. Finally train the LSTM classifier
labels = [0, 1, 0, 1]  # Your labels (0=stand, 1=walk)
train_lstm(
    features_dir="extracted_features",
    labels=labels,
    epochs=30
)
#%%
# 1. Pretrain encoder (from Task 2)
train_encoder(video_dir="your_videos/")

# 2. Train transformer
train_transformer(
    features_dir="extracted_features/",
    labels=[0,1,0,1],  # Dummy labels
    encoder_path="pretrained_encoder.pth"
)

# 3. Run inference
recognizer = TransformerActionRecognizer(
    encoder_path="pretrained_encoder.pth",
    transformer_path="video_transformer.pth",
    class_names=["stand", "walk"]
)

result = recognizer.predict("test_video.mp4")
print(f"Predicted action: {result}")