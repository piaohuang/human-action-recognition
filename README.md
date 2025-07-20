
# humain-action-recognition
This project implements three models for human action recognition using skeleton data and self-supervised learning:
1. **Skeleton-based Video Model**  
   - Extract 2D skeleton keypoints from video frames using an external tool (no training of extractor)  
   - Use LSTM or GRU to model temporal dynamics of keypoint sequences  
   - Train a classifier to predict action classes  
   - Includes training and inference scripts

2. **Self-supervised 2D Encoder + LSTM**  
   - Pretrain a 2D frame-level feature extractor with self-supervised learning (e.g., SimCLR)  
   - Extract per-frame features from videos  
   - Pool features over time using an LSTM classifier  
   - Includes training and inference code

3. **Transformer-based Version**  
   - Replace LSTM with a temporal Transformer for sequence modeling  
   - Reuse the pretrained 2D frame encoder  
   - Includes training and inference code
---

## Data Preparation

Collect 4 short videos (two walking, two standing) for training and a datasets of UCF101 for encoder pretrain

Extract 2D skeleton keypoints using MediaPipe

Use the SimCLR to extract frame-level features

## Dependencies

- Python >= 3.8  
- PyTorch >= 1.10  
- OpenCV  
- Other dependencies listed in requirements.txt

Install dependencies with:

pip install -r requirements.txt

## Run code
Train and Inference with Classifier

`python run_pipeline.py --task [taskname] --[train or inference]`

replace taskname with: `skeleton_gru`, `pretrain_encoder`, `feature_extract`, `pretrain_lstm`, `pretrain_transformer` to launch correspond task.

## Directory structure

├── data/
│   ├── videos/                  # Raw videos
│   ├── keypoints/               # Skeleton data from OpenPose or similar
│   └── features/                # Extracted features from frame encoder

├── src/run_pipeline.py     #excute pipeline

├── src/data/
│   ├── dataset_feature.py         # feature dataset pipeline
│   ├── dataset_pretrain.py          # video dataset pipeline
|   ├── extract_feature.py          # extract feature from pretrained encoder


├── src/models/
│   ├── gru.py         # gru on keypoints
│   ├── simclr.py          # pretrian Encoder model
|   ├── lstm.py          # Encoder + LSTM model
│   └── transformer.py   # Encoder + Transformer model

│
├──src/ training/
│   ├── train.py           # Training script for all classifier model
│   ├── pretrain.py           # Training script for self-supervised
│
├──src/ inference/
│   ├── inference.py           # Inference script

├──src/ checkpoints   #saved model

├── utils/
│   ├──extract_skeletons.py        
│   ├──visualize.py                #show skeletons


├── requirements.txt            # All dependencies
└── README.md                   # Instructions

## Implementation Notes
#### LLM-Generated Components
Basic LSTM/Transformer model skeletons

SimCLR pretraining loop structure

Data loading utilities

#### Custom Components
Keypoint post-processing

Hybrid ConvNet+Transformer architecture

run pipeline

#### Design Decisions
For dataset
- use a custom videos for simplification
- use UCF101 dataset for pretrain encoder 

For Training:

- Used BCEWithLogitsLoss with raw logits from model (no sigmoid inside).

For Temporal Modeling:

- deactivate bidirectional for gru and lstm model due to small dataset to reduce parameters

For Self-Supervision:

- Chose SimCLR over MoCo for simplicity

- Used NT-Xent loss with temperature scaling

Architecture Choices:

- deactivate bidirectional for gru and lstm model due to small dataset to reduce parameters

- Decoupled encoder, classifier, and inference modules: Easy to switch between GRU, LSTM, Transformer. Reusable encoder features → efficient training and testing.

- Transformer with 4 attention heads

- Frame encoder: ResNet18 (pretrained on SimCLR)

#### Future Improvements
Models:

use bidirection for both gru and lstm

use cuda for faster calculate for self supervised learning 

use more head numbers  for transformer

Training:

Add a schelduler learning rate

Implement val loss logging and debug logging

Implement version control of model and dataset