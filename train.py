# video-captioning/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from data.dataloader import get_pt_dataloader, read_list_file
from transformers import GPT2Tokenizer
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder

# Example dummy annotations
# Each item: (video_id_without_extension, "caption text")
from data_processing_augmented import read_list_file, read_captions_file

train_ids = read_list_file(Config.TRAIN_LIST)
video_captions_dict = read_captions_file(Config.CAPTIONS_FILE)

train_annotations = []
for vid in train_ids:
    for caption in video_captions_dict.get(vid, []):
        train_annotations.append((vid, caption))

train_ids = read_list_file(Config.TRAIN_LIST)



# def load_annotations(split, captions_file, split_ids):
#     annotations = []
#     with open(captions_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue
#             parts = line.split(maxsplit=1)
#             if len(parts) < 2:
#                 continue
#             vid, caption = parts
#             if vid in split_ids:
#                 annotations.append((vid, caption))
#     return annotations

# train_annotations = load_annotations("train", Config.CAPTIONS_FILE, train_ids)

def train_one_epoch(loader, c3d_encoder, clip_encoder, fusion_model, gpt2_decoder, optimizer, device):
    c3d_encoder.train()
    if clip_encoder is not None:
        clip_encoder.train()  # might be frozen
    fusion_model.train()
    gpt2_decoder.train()
    
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        video_feats, in_ids, att_msks = [x.to(device) for x in batch]
        
        # Forward depending on the approach
        if c3d_encoder is not None:
            # shape: (B, 3, 16, 112, 112)
            # Actually from DataLoader, shape is (B, 16, 3, 112, 112)
            # might need to permute
            video_feats = video_feats.permute(0, 2, 1, 3, 4)  # => (B, 3, 16, 112, 112)
            enc_out = c3d_encoder(video_feats)  # (B, 15360)
            context_embeds = fusion_model(enc_out)  # (B, T_ctx, 768)
        else:
            # CLIP approach
            # shape: (B, F, 512)
            enc_out = clip_encoder(video_feats)  # (B, F, 768) or pass to fusion
            context_embeds = fusion_model(enc_out)  # (B, T_ctx, 768)
        
        # We also need token embeddings for the text portion
        # GPT2 can build them from input_ids, or we can manually embed
        # Easiest is to do: embedding = gpt2_decoder.gpt2.transformer.wte(input_ids)
        B, T = in_ids.shape
        token_embeds = gpt2_decoder.gpt2.transformer.wte(in_ids)  # (B, T, 768)
        
        # Combine context_embeds + token_embeds along time dimension
        inputs_embeds = torch.cat([context_embeds, token_embeds], dim=1)  # (B, T_ctx + T, 768)
        
        # Build an attention mask that allows attending to everything up to current position
        # For simplicity, let's say we allow all context and text tokens
        attention_mask = torch.cat([
            torch.ones((B, context_embeds.size(1)), device=device),
            att_msks
        ], dim=1)
        
        # Our 'labels' for cross-entropy are the text tokens, shifted
        # but in GPT2 usage, we can just feed them with an offset or set 
        # the context positions to -100 so they're ignored
        labels = in_ids.clone()
        
        # We need to shift the labels to align with the text portion ONLY
        # One trick is to pad with -100 for the context positions so they don't affect the loss
        labels_pad = torch.full((B, context_embeds.size(1)), -100, device=device, dtype=torch.long)
        labels = torch.cat([labels_pad, labels], dim=1)
        
        outputs = gpt2_decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask
        )
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    torch.manual_seed(Config.SEED)
    
    device = torch.device(Config.DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    
    # Create your DataLoader
    train_loader = get_pt_dataloader(
        feature_pt_path=Config.PROCESSED_C3D_FEATS,  # or PROCESSED_CLIP_FEATS
        split_name="train",
        annotations=train_annotations,  # e.g., [("video1", "caption1"), ...]
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    # Choose your approach
    USE_C3D = True  # or False for CLIP
    if USE_C3D:
        c3d_encoder = C3DEncoder().to(device)
        clip_encoder = None
    else:
        c3d_encoder = None
        clip_encoder = CLIPEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
    
    fusion_model = SimpleFusion(context_tokens=Config.CONTEXT_TOKENS).to(device)
    gpt2_decoder = GPT2Decoder().to(device)
    
    # Collect all trainable parameters
    params_to_optimize = list(fusion_model.parameters()) + list(gpt2_decoder.parameters())
    if c3d_encoder is not None:
        params_to_optimize += list(c3d_encoder.parameters())
    if clip_encoder is not None and not Config.FREEZE_CLIP:
        params_to_optimize += list(clip_encoder.parameters())
    
    optimizer = optim.Adam(params_to_optimize, lr=Config.LEARNING_RATE)
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        loss_val = train_one_epoch(
            train_loader, 
            c3d_encoder, 
            clip_encoder, 
            fusion_model, 
            gpt2_decoder, 
            optimizer, 
            device
        )
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {loss_val:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "c3d_encoder": c3d_encoder.state_dict() if c3d_encoder else None,
            "clip_encoder": clip_encoder.state_dict() if clip_encoder else None,
            "fusion_model": fusion_model.state_dict(),
            "gpt2_decoder": gpt2_decoder.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
