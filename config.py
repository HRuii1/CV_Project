# video-captioning/config.py

import torch

class Config:
    ########################
    # General Settings
    ########################
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ########################
    # Training Hyperparameters
    ########################
    EPOCHS = 2
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    
    # If True, freeze CLIP visual backbone
    FREEZE_CLIP = True
    
    # Number of frames to sample per video for the CLIP-based model
    NUM_FRAMES_CLIP = 5
    # Number of frames to stack per video for C3D-based model
    NUM_FRAMES_C3D = 16
    
    ########################
    # GPT-2 Settings
    ########################
    GPT2_MODEL_NAME = "distilgpt2"   # any Hugging Face GPT-2 variant
    MAX_SEQ_LEN = 30          # maximum words in generated caption
    CONTEXT_TOKENS = 20       # number of tokens for visual context
    
    ########################
    # File/Folder Paths
    ########################
    # DATA_DIR = "data"         # root data directory
    # VIDEO_DIR = "raw_videos"  # where raw videos might reside
    OUTPUT_DIR = "checkpoints"
    # CAPTIONS_FILE = "data/video_captions.txt"
    # TRAIN_LIST = "data/train_list.txt"
    VIDEO_DIR = "data/debug_videos"
    # Point to debug outputs
    CAPTIONS_FILE = "data/debug_data/debug_video_captions.txt"
    TRAIN_LIST = "data/debug_data/debug_train_list.txt"
    TEST_LIST = "data/debug_data/debug_test_list.txt"


    # C3D or CLIP feature files (choose based on what you preprocessed)
    PROCESSED_C3D_FEATS = "data/debug_data/debug_outputs/version1.pt"
    PROCESSED_CLIP_FEATS = "data/debug_data/debug_outputs/version2.pt"
    
    # Preprocessed feature paths (CLIP or C3D)
    # PROCESSED_CLIP_FEATS = "data/tures"
    # PROCESSED_C3D_FEATS = "data/c3d_features"


