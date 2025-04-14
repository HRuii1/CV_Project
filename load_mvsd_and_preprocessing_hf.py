import os
import io
import torch
import clip
import random
import av
import logging
import numpy as np
from PIL import Image
from datasets import load_dataset
from torchvision import transforms as T

# =============================================================================
#                          CONFIGURATION
# =============================================================================
OUTPUT_DIR = "data/hf_msvd_outputs"
CACHE_DIR = "/scratch/vmr8pg/hf_cache"

# Path to your local MSVD videos (.avi) after extracting YouTubeClips.tar
MSVD_VIDEO_DIR = "data/msvd_videos/YouTubeClips"

NUM_FRAMES_V1 = 16
FRAME_SIZE_V1 = (112, 112)

NUM_FRAMES_V2 = 5
NUM_RANDOM_SETS = 5
MIN_INTERVAL = 1
MAX_INTERVAL = 5

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

VERSION1_PT     = os.path.join(OUTPUT_DIR, "version1.pt")
VERSION2_PT     = os.path.join(OUTPUT_DIR, "version2.pt")
VERSION1_AUG_PT = os.path.join(OUTPUT_DIR, "version1_aug.pt")
VERSION2_AUG_PT = os.path.join(OUTPUT_DIR, "version2_aug.pt")
LOG_FILE        = os.path.join(OUTPUT_DIR, "preprocess_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
#                   HELPER FUNCTIONS
# =============================================================================
def extract_frames_from_video(video_source, num_frames=16):
    """
    This function opens a local video file path via PyAV and extracts the first `num_frames`.
    """
    container = av.open(video_source)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        img = frame.to_image()
        frames.append(img)
        if len(frames) == num_frames:
            break
    return frames

def resize_and_normalize_frames(frames, size, mean=MEAN, std=STD):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    return torch.stack([transform(img) for img in frames], dim=0)

def random_sample_indices(total, k):
    return sorted(random.sample(range(total), k))

# =============================================================================
#                           MAIN
# =============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # 1) Load each split in a loop
    # -------------------------------------------------------------------------
    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test"
    }

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Prepare data structures keyed by split
    version1_data = {
        "train": {},
        "val": {},
        "test": {}
    }
    version2_data = {
        "train": {},
        "val": {},
        "test": {}
    }
    version1_aug_data = {}
    version2_aug_data = {}

    # We'll store augmented data by split as well if needed
    for short_name in ["train", "val", "test"]:
        version1_aug_data[short_name] = {}
        version2_aug_data[short_name] = {}

    # -------------------------------------------------------------------------
    # For each of train/validation/test, read the dataset
    # -------------------------------------------------------------------------
    for split_name, short_name in split_map.items():
        print(f"\n=== Processing split: {split_name} ===")
        dataset = load_dataset("friedrichor/MSVD", split=split_name, cache_dir=CACHE_DIR)
        print(f"Dataset has {len(dataset)} examples")

        count = 0
        for item in dataset:
            video_id = item["video_id"]
            captions = item["caption"]

            try:
                # item["video"] is just a filename, e.g. "WTf5EgVY5uU_98_104.avi"
                video_filename = item["video"]
                video_source = os.path.join(MSVD_VIDEO_DIR, video_filename)

                # Make sure the file actually exists
                if not os.path.exists(video_source):
                    logging.warning(f"[{split_name}] File not found: {video_source}")
                    continue

                # Extract frames from the local .avi file
                frames = extract_frames_from_video(video_source, NUM_FRAMES_V1)
                if len(frames) < NUM_FRAMES_V1:
                    logging.warning(f"[{split_name}] Too few frames in {video_id}, skipping.")
                    continue

                # -------------- version1 (C3D-like) --------------
                frames_v1 = resize_and_normalize_frames(frames, FRAME_SIZE_V1)
                version1_data[short_name][video_id] = frames_v1

                # -------------- version2 (CLIP) --------------
                # Randomly pick 5 frames out of the 16
                idx_5 = random_sample_indices(NUM_FRAMES_V1, NUM_FRAMES_V2)
                selected_frames = [frames[i] for i in idx_5]
                frames_v2_tensor = torch.stack([clip_preprocess(img) for img in selected_frames]).to(device)

                with torch.no_grad():
                    clip_feats = clip_model.encode_image(frames_v2_tensor).cpu()
                version2_data[short_name][video_id] = clip_feats

                # -------------- Augmentations --------------
                version1_aug_data[short_name][video_id] = []
                version2_aug_data[short_name][video_id] = []

                for cap in captions:
                    for s in range(NUM_RANDOM_SETS):
                        idx_rand = random_sample_indices(NUM_FRAMES_V1, NUM_FRAMES_V2)
                        rand_frames = [frames[i] for i in idx_rand]
                        rand_v1 = resize_and_normalize_frames(rand_frames, FRAME_SIZE_V1)
                        rand_v2_tensor = torch.stack([clip_preprocess(img) for img in rand_frames]).to(device)

                        with torch.no_grad():
                            rand_clip_feats = clip_model.encode_image(rand_v2_tensor).cpu()

                        version1_aug_data[short_name][video_id].append({
                            "caption": cap, 
                            "data": rand_v1
                        })
                        version2_aug_data[short_name][video_id].append({
                            "caption": cap, 
                            "data": rand_clip_feats
                        })

                logging.info(f"[{split_name}] Processed {video_id}")
                count += 1
                if count % 50 == 0:
                    print(f"{count} videos processed in {split_name} split...")

            except Exception as e:
                logging.error(f"[{split_name}] Error processing {video_id}: {str(e)}")

        print(f"=== Finished {split_name} with {count} videos processed ===")

    # -------------------------------------------------------------------------
    # Save final .pt files
    # -------------------------------------------------------------------------
    torch.save(version1_data[short_name], f"{OUTPUT_DIR}/version1_{short_name}.pt")
    torch.save(version2_data[short_name], f"{OUTPUT_DIR}/version2_{short_name}.pt")
    torch.save(version1_aug_data[short_name], f"{OUTPUT_DIR}/version1_aug_{short_name}.pt")
    torch.save(version2_aug_data[short_name], f"{OUTPUT_DIR}/version2_aug_{short_name}.pt")

    torch.save(version1_aug_data, VERSION1_AUG_PT)
    torch.save(version2_aug_data, VERSION2_AUG_PT)
    logging.info("Preprocessing completed and saved for all splits.")
    print("Preprocessing completed and saved for all splits.")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
