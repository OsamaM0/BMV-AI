from src.config import SEQ_LEN
import numpy as np
import os
import cv2
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
import mediapipe as mp


actions = []

DATA_PATH = r"W:\NLP\SLD\Video Database\videos"
NPY_PATH =  r"npy_data"



mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

print("Start")
# Loop through each folder and video file
for root, dirs, files in os.walk(DATA_PATH):
    for dir in dirs:

        action = dir
        print("Current Sign: ",action)
        action_path = os.path.join(root, dir)
        video_files = [f for f in os.listdir(action_path) if os.path.isfile(os.path.join(action_path, f))]
        
        # Loop through each video file
        for sequence, video_file in enumerate(video_files):
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            frame_list = []
            previous_frame = None
            
            with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if previous_frame is None:
                        previous_frame = frame
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                                        
                    frame_list.append((frame, results))

                    previous_frame = frame
            
            cap.release()
            
            # Select evenly spaced frames from the collected active frames
            selected_frames = [frame_list[i] for i in np.linspace(0, len(frame_list)-1, SEQ_LEN, dtype=int)]
            
            for frame_num, (frame, results) in enumerate(selected_frames):
                keypoints = extract_coordinates(results)
                npy_path = os.path.join(NPY_PATH, action, str(sequence), str(frame_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, keypoints)
                # cv2.imshow('OpenCV Feed',frame)
                if cv2.waitKey(0)== ord('q'):
                    break
                    
            print(f'Finished processing {action}/{video_file}')


cv2.destroyAllWindows()