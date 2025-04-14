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
    EPOCHS = 5
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
    OUTPUT_DIR = "checkpoints"

    # No need for external txt list files anymore
    # We build annotations directly from the .pt files

    # C3D and CLIP features
    PROCESSED_C3D_FEATS = "/bigtemp/vmr8pg/MSVD/version1.pt"
    PROCESSED_C3D_FEATS_AUG = "/bigtemp/vmr8pg/MSVD/version1_aug.pt"
    # PROCESSED_C3D_FEATS_AUG = "mini.pt"
    # PROCESSED_C3D_FEATS = "mini.pt"
    PROCESSED_CLIP_FEATS = "/bigtemp/vmr8pg/MSVD/version2.pt"
    PROCESSED_CLIP_FEATS_AUG = "/bigtemp/vmr8pg/MSVD/version2_aug.pt"


    # Debug mode limit
    DEBUG_LIMIT = None  # set to None or a small number to debug faster
