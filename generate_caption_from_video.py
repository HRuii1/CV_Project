import os
import cv2
import torch
import clip
from PIL import Image
from torchvision import transforms as T
from transformers import GPT2Tokenizer

from config import Config  # or define your own config inline
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder

###############################
# 1) HELPER FUNCTIONS
###############################

def extract_frames_from_video(video_path, num_frames=5):
    """
    Extracts `num_frames` equally spaced frames from the .avi video.
    Returns a list of PIL Images (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total_frames < num_frames or total_frames <= 0:
        cap.release()
        return frames  # or handle error
    step = total_frames / float(num_frames - 1)
    for i in range(num_frames):
        frame_idx = int(round(i * step))
        frame_idx = min(frame_idx, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def preprocess_frames_clip(frames_pil):
    """
    Preprocess each PIL frame for CLIP's input. 
    Returns a tensor of shape (num_frames, 3, 224, 224).
    """
    # If you have your own clip_preprocess, you can use that. 
    # Otherwise, define a transform consistent with your training.
    clip_transform = T.Compose([
        T.Resize((224,224)),  # or the transform used in training
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    tensors = [clip_transform(img) for img in frames_pil]
    return torch.stack(tensors, dim=0)  # (num_frames, 3, 224, 224)

###############################
# 2) MAIN INFERENCE LOGIC
###############################

def main():
    device = torch.device(Config.DEVICE)  # "cuda" or "cpu"
    
    # 1) Load your checkpoint
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "model_epoch_5.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2) Reconstruct your model
    clip_encoder = CLIPEncoder(freeze_clip=True).to(device)
    clip_encoder.load_state_dict(checkpoint["clip_encoder"])
    
    fusion_model = SimpleFusion(
        context_tokens=Config.CONTEXT_TOKENS,
        input_dim=768,  # CLIP dimension
        frames=5        # we are sampling 5 frames
    ).to(device)
    fusion_model.load_state_dict(checkpoint["fusion_model"])
    
    gpt2_decoder = GPT2Decoder().to(device)
    gpt2_decoder.load_state_dict(checkpoint["gpt2_decoder"])

    # 3) Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token is set

    # 4) Provide a path to your .avi video in /videos
    video_path = "videos/kid_poll.avi"  # change to your actual file
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    # 5) Extract frames
    frames_pil = extract_frames_from_video(video_path, num_frames=5)
    if len(frames_pil) < 5:
        print("Not enough frames extracted; skipping inference.")
        return

    # 6) Preprocess frames for CLIP
    frames_tensor = preprocess_frames_clip(frames_pil).to(device)  # (5, 3, 224, 224)
    frames_tensor = frames_tensor.unsqueeze(0)                     # (1, 5, 3, 224, 224)

    # 7) Encode frames with CLIPEncoder
    with torch.no_grad():
        # shape after clip_encoder forward() -> (B, F, 768) = (1, 5, 768)
        enc_out = clip_encoder(frames_tensor)
    
        # 8) Fuse into context embeddings
        # shape -> (1, T_ctx, 768) typically = (1, 20, 768)
        context_embeds = fusion_model(enc_out)
    
        # 9) Generate with GPT-2
        gen_ids = gpt2_decoder.generate(
            context_embeds,
            max_length=Config.MAX_SEQ_LEN,
            num_beams=1
        )
        caption = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    
    print("Generated caption:", caption)

if __name__ == "__main__":
    main()
