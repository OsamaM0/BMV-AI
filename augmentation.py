import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip

def crop_center(video, original_size):
    height, width = original_size
    new_width = int(width * 0.7)
    new_height = int(height * 0.7)
    x_start = (width - new_width) // 2
    y_start = (height - new_height) // 2
    cropped = video[:, y_start:y_start+new_height, x_start:x_start+new_width]
    return np.array([cv2.resize(frame, (width, height)) for frame in cropped])

def crop_top_half(video, original_size):
    height, width = original_size
    cropped = video[:, :2*(height//3), :]
    return np.array([cv2.resize(frame, (width, height)) for frame in cropped])

def expand_video(video, original_size):
    height, width = original_size
    new_width = int(width * 1.5)
    new_height = int(height * 1.5)
    expanded = np.zeros((video.shape[0], new_height, new_width, video.shape[3]), dtype=video.dtype)
    x_start = (new_width - width) // 2
    y_start = (new_height - height) // 2
    expanded[:, y_start:y_start+height, x_start:x_start+width] = video
    return np.array([cv2.resize(frame, (width, height)) for frame in expanded])

def shear_video(video, original_size, shear_factor=0.2):
    height, width = original_size
    M = np.array([[1, shear_factor, 0],
                  [0, 1, 0]])
    sheared = np.array([cv2.warpAffine(frame, M, (width, height)) for frame in video])
    return np.array([cv2.resize(frame, (width, height)) for frame in sheared])

def flip_video(video):
    return np.array([cv2.flip(frame, 1) for frame in video])

def process_videos(input_folder):
    augment_suffixes = ["center_cropped", "top_cropped", "expanded", "sheared", "flipped"]
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            try:
                if file.endswith(".mp4") and not any(suffix in file for suffix in augment_suffixes):
                    video_path = os.path.join(root, file)
                    video_name, ext = os.path.splitext(file)

                    # Read video
                    clip = VideoFileClip(video_path)
                    frames = [frame for frame in clip.iter_frames()]
                    video_array = np.array(frames)
                    # Correct color channel order (MoviePy returns in RGB, but OpenCV expects BGR)
                    video_array = video_array[..., ::-1]

                    original_size = (video_array.shape[1], video_array.shape[2])

                    # Apply augmentations
                    augmented_videos = {
                        "top_cropped": crop_top_half(video_array, original_size),
                        "sheared": shear_video(video_array, original_size),
                        "flipped": flip_video(video_array),
                        "expanded": expand_video(video_array, original_size),
                        "center_cropped": crop_center(video_array, original_size)
                    }

                    # Save augmented videos
                    for aug_name, aug_video in augmented_videos.items():
                        output_path = os.path.join(root, f"{video_name}_{aug_name}{ext}")
                        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), clip.fps, (original_size[1], original_size[0]))
                        for frame in aug_video:
                            out.write(frame)
                        out.release()
                        print(f"{video_name}_{aug_name}{ext}")
            except:
                continue
# Example usage
input_folder = r"W:\NLP\SLD\Video Database\videos"
process_videos(input_folder)
