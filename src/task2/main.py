# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:04:57 2025

@author: admin
"""
import torch
from training.train_encoder import train_encoder
from data.SimCLR_pretrain import ContrastiveFrameEncoder
from data.feature_extact import VideoFeatureExtractor
from training.train_lstm import train_lstm
from training.train_transformer import train_transformer
from training.inference import ActionRecognizer
from pathlib import Path 
import os
data_dir = Path(r"F:\Piao\human-action-recognition\data\task1\raw\train")
encorder_path = Path(r"F:\Piao\human-action-recognition\src\task2\checkpoints\pretrained_encoder_checkpoint.pth")
lstm_path = Path(r"F:\Piao\human-action-recognition\src\task2\checkpoints\lstm_classifier_checkpoint.pth")
transformer_path = Path(r"F:\Piao\human-action-recognition\src\task2\checkpoints\transformer_classifier_checkpoint.pth")

# 1. First pretrain the encoder
# 
# train_encoder(
#     video_dir=data_dir,
#     epochs=200
# )

# 2. extract features from encoder
# feature_extractor = VideoFeatureExtractor(encorder_path)
# for filename in os.listdir(data_dir):
#         file_path = os.path.join(data_dir, filename)
#         if os.path.isfile(file_path):
#             name, ext = os.path.splitext(filename)
#             if ext.lower() in [".mp4"]:
#                 features = feature_extractor.extract_features(
#                     os.path.join(data_dir,filename),
#                     os.path.join(Path(r"F:\Piao\human-action-recognition\data\task1"),
#                                  "extracted_features",
#                                  name+'.npy'
#                 ))


# 3. Finally train the LSTM classifier
train_lstm(
    features_dir=os.path.join(Path(r"F:\Piao\human-action-recognition\data\task1"),
                 "extracted_features"),
    labels=None,
    epochs=200
)

# # 4.  LSTM classifier inference
recognizer = ActionRecognizer(
    encoder_path=encorder_path,
    model_type='lstm',
    classifier_path=lstm_path,
    class_names=[0, 1]
)
result = recognizer.predict(os.path.join(Path(r"F:\Piao\human-action-recognition\data\task1\raw\dev"),
                   "test3.mp4"))
print(f"Predicted action: {result}")
#%%

#2. Train transformer
# train_transformer(
#     features_dir=os.path.join(Path(r"F:\Piao\human-action-recognition\data\task1"),
#                       "extracted_features"),
#          labels=None,
#          epochs=500
# )

# 3. Run inference
recognizer = ActionRecognizer(
    encoder_path=encorder_path,
    model_type='transformer',
    classifier_path=transformer_path,
    class_names=[0, 1]
)

result = recognizer.predict(os.path.join(Path(r"F:\Piao\human-action-recognition\data\task1\raw\dev"),
                  "test2.mp4"))
print(f"Predicted action: {result}")