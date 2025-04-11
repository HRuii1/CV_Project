# video-captioning/data/preprocess.py

import os
import cv2
import torch
import numpy as np
import clip
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

def extract_frames_c3d(video_path, num_frames=16, frame_size=112):
    """
    Extract 'num_frames' from a video, resize them to (frame_size x frame_size).
    Returns a numpy array of shape (num_frames, 3, frame_size, frame_size).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # indices of frames to sample
    interval = max(total_frames // num_frames, 1)
    selected_indices = [i * interval for i in range(num_frames)]
    
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in selected_indices:
            # convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize
            frame_rgb = cv2.resize(frame_rgb, (frame_size, frame_size))
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            
            # move channel dimension first
            frame_rgb = np.transpose(frame_rgb, (2, 0, 1))  # (3, H, W)
            frames.append(frame_rgb)
    
    cap.release()
    
    # If actual # frames < desired, pad with zeros
    while len(frames) < num_frames:
        frames.append(np.zeros((3, frame_size, frame_size), dtype=np.float32))
    
    frames = np.stack(frames, axis=0)  # (num_frames, 3, frame_size, frame_size)
    return frames

def extract_frames_clip(video_path, num_frames=5, frame_size=224):
    """
    Extract 'num_frames' frames for CLIP, 
    but do not embed them yet (if you want to store raw frames).
    Returns a list of PIL Images or numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)
    selected_indices = [i * interval for i in range(num_frames)]
    
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in selected_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (frame_size, frame_size))
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
    
    cap.release()
    return frames

def preprocess_c3d_features(video_dir, out_dir):
    """
    For every video in video_dir, extract 16 frames and save them as a .npy file.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    
    for vf in video_files:
        video_path = os.path.join(video_dir, vf)
        frames = extract_frames_c3d(video_path, Config.NUM_FRAMES_C3D, 112)
        out_path = os.path.join(out_dir, vf + ".npy")
        np.save(out_path, frames)
        print("Saved C3D frames:", out_path)

def preprocess_clip_features(video_dir, out_dir):
    """
    For every video in video_dir, extract frames, feed to CLIP, and save a single .npy
    or a list of .npy files (one embedding per frame).
    """
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device(Config.DEVICE)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    
    for vf in video_files:
        video_path = os.path.join(video_dir, vf)
        frames_pil = extract_frames_clip(video_path, Config.NUM_FRAMES_CLIP, 224)
        
        # preprocess frames using the CLIP transform
        frames_input = [preprocess(img).unsqueeze(0).to(device) for img in frames_pil]
        frames_input = torch.cat(frames_input, dim=0)  # shape: (num_frames, 3, 224, 224)
        
        with torch.no_grad():
            frame_features = clip_model.encode_image(frames_input)  # (num_frames, 512)
        
        frame_features = frame_features.cpu().numpy()
        out_path = os.path.join(out_dir, vf + ".npy")
        np.save(out_path, frame_features)
        print("Saved CLIP features:", out_path)

if __name__ == "__main__":
    # Example usage:
    # python preprocess.py
    # Make sure your config points to correct directories
    preprocess_c3d_features(Config.VIDEO_DIR, Config.PROCESSED_C3D_FEATS)
    preprocess_clip_features(Config.VIDEO_DIR, Config.PROCESSED_CLIP_FEATS)
