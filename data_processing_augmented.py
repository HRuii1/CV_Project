import os
import random
import cv2
import torch
import clip
import numpy as np
import logging
import time

from PIL import Image
from torchvision import transforms as T

# try
TRAIN_LIST = "data/debug_data/debug_train_list.txt"
VAL_LIST = "data/debug_data/debug_val_list.txt"
TEST_LIST = "data/debug_data/debug_test_list.txt"
CAPTIONS_FILE = "data/debug_data/debug_video_captions.txt"
MSVD_FOLDER = "data/debug_data/debug_videos"
OUTPUT_DIR = "data/debug_data/debug_outputs"



###################################################
#                CONFIGURATION
###################################################

# Paths to your list files
# TRAIN_LIST = "train_list.txt"
# VAL_LIST   = "val_list.txt"
# TEST_LIST  = "test_list.txt"

# Path to your video captions file
# CAPTIONS_FILE = "video_captions.txt"

# Folder where the actual MSVD video files are stored
# MSVD_FOLDER = "MSVD"

# Video file extension (e.g., '.avi' or '.mp4')
VIDEO_EXT = ".mp4"

# Base output folder; we will save our processed data as .pt files here.
# OUTPUT_DIR = "/bigtemp/xjj5ys/cv_project/MSVD"

# Output filenames for our datasets
# Base preprocessed data will be stored as a dictionary with keys "train", "val", "test".
VERSION1_PT = os.path.join(OUTPUT_DIR, "version1.pt")       # Each entry: (16, 3, 112, 112)
VERSION2_PT = os.path.join(OUTPUT_DIR, "version2.pt")       # Each entry: (5, 512)
# Augmented data (only for training videos)
VERSION1_AUG_PT = os.path.join(OUTPUT_DIR, "version1_aug.pt")
VERSION2_AUG_PT = os.path.join(OUTPUT_DIR, "version2_aug.pt")

# Checkpoint file
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.pt")
CHECKPOINT_INTERVAL = 150  # save checkpoint every 10 videos

# Log file path
LOG_FILE = os.path.join(OUTPUT_DIR, "preprocess_log.txt")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# Normalization stats (commonly used ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Number of frames for version 1
NUM_FRAMES_V1 = 16
# Resize version1 frames to 112x112
FRAME_SIZE_V1 = (112, 112)

# Number of frames to sample for version 2 (from the 16)
NUM_FRAMES_V2 = 5

# For data augmentation Method 2 (random time-sampling)
NUM_RANDOM_SETS = 5
MIN_INTERVAL = 1
MAX_INTERVAL = 5

###################################################
#              LOGGING SETUP
###################################################
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',  # Append mode
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

###################################################
#              HELPER FUNCTIONS
###################################################

def read_list_file(list_file):
    """Reads a list file (e.g., train_list.txt) and returns a set of video IDs."""
    ids = set()
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            vid = line.strip()
            if vid:
                ids.add(vid)
    return ids

def read_all_lists(train_list, val_list, test_list):
    """Returns a combined set of video IDs from train/val/test lists, and each separately."""
    train_ids = read_list_file(train_list)
    val_ids   = read_list_file(val_list)
    test_ids  = read_list_file(test_list)
    return train_ids.union(val_ids).union(test_ids), train_ids, val_ids, test_ids

def read_captions_file(captions_file):
    """
    Reads 'video captions.txt' and returns a dictionary:
      { video_id: [caption1, caption2, ...], ... }
    Ignores lines starting with '#'.
    """
    video_to_captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            vid, caption = parts[0], parts[1]
            video_to_captions.setdefault(vid, []).append(caption)
    return video_to_captions

def extract_equally_spaced_frames(video_path, num_frames=16):
    """
    Extracts `num_frames` frames from the video at equal intervals.
    Returns a list of PIL Images (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames < num_frames or total_frames <= 0:
        cap.release()
        return frames
    step = total_frames / float(num_frames - 1)
    for i in range(num_frames):
        frame_idx = int(round(i * step))
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def resize_and_normalize_frames(frames, size, mean=MEAN, std=STD):
    """
    Resizes each PIL Image to `size`, converts to a tensor, and normalizes.
    Returns a tensor of shape (num_frames, 3, H, W).
    """
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    tensors = [transform(img) for img in frames]
    return torch.stack(tensors, dim=0)

def random_sample_indices(total, k):
    """Returns a sorted list of k unique indices from [0, total-1]."""
    return sorted(random.sample(range(total), k))

def random_time_sampling(video_path, num_frames, min_interval, max_interval):
    """
    Data augmentation Method 2:
    Randomly samples `num_frames` frames from the video with min/max intervals.
    Returns a list of PIL Images (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    indices = []
    start_idx = random.randint(0, total_frames - 1)
    indices.append(start_idx)
    current_idx = start_idx
    for _ in range(num_frames - 1):
        interval = random.randint(min_interval, max_interval)
        next_idx = current_idx + interval
        if next_idx >= total_frames:
            break
        indices.append(next_idx)
        current_idx = next_idx
    if len(indices) < num_frames:
        cap.release()
        return []
    output_imgs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_imgs.append(Image.fromarray(frame_rgb))
    cap.release()
    return output_imgs if len(output_imgs) == num_frames else []

def save_checkpoint(checkpoint, checkpoint_file):
    """Saves the checkpoint dictionary to disk."""
    torch.save(checkpoint, checkpoint_file)
    logging.info(f"Checkpoint saved with {len(checkpoint['processed_videos'])} videos processed.")

def load_checkpoint(checkpoint_file):
    """Loads the checkpoint dictionary from disk."""
    checkpoint = torch.load(checkpoint_file)
    logging.info(f"Checkpoint loaded with {len(checkpoint['processed_videos'])} videos processed.")
    return checkpoint

###################################################
#                   MAIN
###################################################

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_video_ids, train_ids, val_ids, test_ids = read_all_lists(TRAIN_LIST, VAL_LIST, TEST_LIST)
    logging.info(f"Total unique videos: {len(all_video_ids)}")
    logging.info(f"Total training videos: {len(train_ids)}")
    logging.info(f"Total validation videos: {len(val_ids)}")
    logging.info(f"Total test videos: {len(test_ids)}")

    video_captions = read_captions_file(CAPTIONS_FILE)
    logging.info(f"Total videos with captions: {len(video_captions)}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    logging.info("CLIP model loaded successfully.")

    # Initialize or load checkpoint.
    processed_videos = set()
    version1_data = {"train": {}, "val": {}, "test": {}}
    version2_data = {"train": {}, "val": {}, "test": {}}
    version1_aug_data = {}
    version2_aug_data = {}

    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = load_checkpoint(CHECKPOINT_FILE)
        version1_data = checkpoint.get("version1_data", version1_data)
        version2_data = checkpoint.get("version2_data", version2_data)
        version1_aug_data = checkpoint.get("version1_aug_data", version1_aug_data)
        version2_aug_data = checkpoint.get("version2_aug_data", version2_aug_data)
        processed_videos = set(checkpoint.get("processed_videos", []))
        logging.info(f"Resuming from checkpoint. Videos already processed: {processed_videos}")

    video_list = list(all_video_ids)
    total_videos = len(video_list)
    for idx, vid in enumerate(video_list):
        if vid in processed_videos:
            logging.info(f"Skipping already processed video: {vid}")
            continue

        video_path = os.path.join(MSVD_FOLDER, vid + VIDEO_EXT)
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found: {video_path}")
            processed_videos.add(vid)
            continue

        frames_pil = extract_equally_spaced_frames(video_path, NUM_FRAMES_V1)
        if len(frames_pil) < NUM_FRAMES_V1:
            logging.warning(f"Not enough frames in {vid}. Skipping.")
            processed_videos.add(vid)
            continue

        frames_v1_tensor = resize_and_normalize_frames(frames_pil, FRAME_SIZE_V1, MEAN, STD)
        idx_5 = random_sample_indices(NUM_FRAMES_V1, NUM_FRAMES_V2)
        selected_frames = [frames_pil[i] for i in idx_5]
        processed_frames = [clip_preprocess(img) for img in selected_frames]
        frames_v2_tensor = torch.stack(processed_frames, dim=0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(frames_v2_tensor)
        clip_features = clip_features.cpu()

        if vid in train_ids:
            version1_data["train"][vid] = frames_v1_tensor
            version2_data["train"][vid] = clip_features
        elif vid in val_ids:
            version1_data["val"][vid] = frames_v1_tensor
            version2_data["val"][vid] = clip_features
        elif vid in test_ids:
            version1_data["test"][vid] = frames_v1_tensor
            version2_data["test"][vid] = clip_features

        logging.info(f"Processed base versions for video: {vid}")

        if vid in train_ids:
            version1_aug_data[vid] = []
            version2_aug_data[vid] = []
            if vid in video_captions:
                for cap_text in video_captions[vid]:
                    for s in range(NUM_RANDOM_SETS):
                        frames_rand_pil = random_time_sampling(video_path, NUM_FRAMES_V1, MIN_INTERVAL, MAX_INTERVAL)
                        if len(frames_rand_pil) < NUM_FRAMES_V1:
                            continue
                        frames_rand_v1 = resize_and_normalize_frames(frames_rand_pil, FRAME_SIZE_V1, MEAN, STD)
                        idx_5_rand = random_sample_indices(NUM_FRAMES_V1, NUM_FRAMES_V2)
                        selected_rand_frames = [frames_rand_pil[i] for i in idx_5_rand]
                        processed_rand_frames = [clip_preprocess(img) for img in selected_rand_frames]
                        frames_rand_v2_tensor = torch.stack(processed_rand_frames, dim=0).to(device)
                        with torch.no_grad():
                            clip_features_rand = clip_model.encode_image(frames_rand_v2_tensor)
                        clip_features_rand = clip_features_rand.cpu()

                        aug_entry_v1 = {"type": "random", "set": s, "caption": cap_text, "data": frames_rand_v1}
                        aug_entry_v2 = {"type": "random", "set": s, "caption": cap_text, "data": clip_features_rand}
                        version1_aug_data[vid].append(aug_entry_v1)
                        version2_aug_data[vid].append(aug_entry_v2)
                logging.info(f"Added nested caption and random-sampling augmentations for training video: {vid}")

        processed_videos.add(vid)

        # Optionally, save checkpoint every CHECKPOINT_INTERVAL videos.
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                "version1_data": version1_data,
                "version2_data": version2_data,
                "version1_aug_data": version1_aug_data,
                "version2_aug_data": version2_aug_data,
                "processed_videos": list(processed_videos)
            }
            save_checkpoint(checkpoint, CHECKPOINT_FILE)
            logging.info(f"Checkpoint saved after processing {idx+1}/{total_videos} videos.")

    # Final save.
    torch.save(version1_data, VERSION1_PT)
    torch.save(version2_data, VERSION2_PT)
    torch.save(version1_aug_data, VERSION1_AUG_PT)
    torch.save(version2_aug_data, VERSION2_AUG_PT)
    logging.info("Saved all processed data into final .pt files.")
    logging.info("All done with preprocessing and data augmentation!")
    
if __name__ == "__main__":
    main()
