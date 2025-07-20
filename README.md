human-action-recognition


#################
# Directory structure#
#################
human-action-recognition/
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
