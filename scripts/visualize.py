# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:57:46 2025

@author: admin
"""
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt	
import numpy as np
from pathlib import Path 
import os
root_path = Path(r"F:\Piao\veesion\data")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
# MediaPipe pose connections (indices for 33-keypoint model)
connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), 
        (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), 
        (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),
        (16, 22), (11, 23), (12, 24), (23, 24), (23, 25),
        (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32)
    ]		
def animate_skeleton(skeleton_data, output_file='skeleton_animation.mp4',img_width=1500, img_height = 1500,fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (img_width, img_height))
   
    for frame in skeleton_data:
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Convert to pixel coordinates if normalized
        if np.max(frame) <= 1.0:
            frame[:, 0] *= 500
            frame[:, 1] *= 1500
        
        # Draw skeleton
        for (i, j) in connections:
            if i < len(frame) and j < len(frame):
                x1, y1 = int(frame[i, 0]), int(frame[i, 1])
                x2, y2 = int(frame[j, 0]), int(frame[j, 1])
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for joint in frame:
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        
        out.write(img)
        cv2.imshow('Animation', img)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
            break
    
    out.release()
    cv2.destroyAllWindows()
for label in ['test1','test2','test3','walk1', 'walk2', 'stand1','stand2']:
    input_file = label+'_skeletons.npy'
    keypoints_sequence = np.load(os.path.join(root_path,'output_keypoints', input_file))
    print(input_file, keypoints_sequence.shape)
    animate_skeleton(keypoints_sequence, input_file.replace('npy','mp4'))