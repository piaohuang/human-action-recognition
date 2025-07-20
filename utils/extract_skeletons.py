import cv2
import numpy as np
import os
import mediapipe as mp
		

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8)

def extract_skeletons_from_video(video_path, output_folder="skeletons"):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)    
    skeletons = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y])
            skeletons.append(np.array(keypoints))  # Shape: (33, 2)
        

    cap.release()
    
    # Save skeletons as .npy file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, f"{video_name}_skeletons.npy")
    np.save(output_path, np.array(skeletons))
    print(f"Saved skeletons to {output_path}")
    
    return np.array(skeletons)
	
if __name__ == "__main__":
    from pathlib import Path 
    BASE_DIR = Path(__file__).parent.parent.resolve()
    root_path = BASE_DIR / "data" /"videos" / "dev"
    video_extensions = ['.mp4']
    if not os.path.exists(root_path):
        print(f"error: folder path {root_path} not exist")
    for filename in os.listdir(root_path):
            file_path = root_path / filename
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() in video_extensions:
                    keypoints_sequence = extract_skeletons_from_video(file_path, BASE_DIR / "data" / "keypoints")
                  
                    

                    