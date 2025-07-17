# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:40:24 2025

@author: admin
"""
import os
import random
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image

#debug
def show_images(x_i, x_j, title="Augmentations Contrastives"):
    """
    Affiche côte à côte les deux vues augmentées d'une même image.
    
    Args:
        x_i, x_j (torch.Tensor): Images augmentées (shape [C, H, W])
        title (str): Titre de la figure
    """
    # Convertir les tenseurs -> numpy arrays
    img1 = to_pil_image(x_i)  # Si sur GPU
    img2 = to_pil_image(x_j)
    
    # Créer une figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    # Afficher les images
    axes[0].imshow(img1)
    axes[0].set_title("View 1", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title("View 2", fontsize=12)
    axes[1].axis('off')
    
    # Calculer la différence entre les vues
    diff = np.abs(np.array(img1).astype(float) - np.array(img2).astype(float))
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min())  # Normaliser [0,1]
    
    # Optionnel: Afficher la carte de différence
    if np.max(diff) > 0.1:  # Seulement si différence significative
        fig_diff, ax_diff = plt.subplots(1, 1, figsize=(5, 5))
        ax_diff.imshow(diff_norm)
        ax_diff.set_title("Carte de différence (normalisée)", fontsize=12)
        ax_diff.axis('off')
    
    plt.tight_layout()
    plt.show()
    
class ContrastiveVideoDataset(Dataset):
    def __init__(self, video_dir, frame_size=224):
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                          if f.endswith(('.avi', '.MP4'))]
        print('self.video_files',self.video_files,video_dir)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(frame_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_files[idx])
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Randomly select a frame
        frame = random.choice(frames)
        
        # Return two augmented views of the same frame
        x_i = self.transform(frame)
        x_j = self.transform(frame)
        # Debug: Affichez les deux vues (désactivez en production !)
        # if idx == 10 and random.random() < 0.1:  # 1% des batches
        #     show_images(x_i, x_j)  # Implémentez cette fonction
        return x_i, x_j