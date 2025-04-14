# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Import your config or define it inline
from config import Config

# Your model code
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder

# The lazy loader we just created
from lazy_dataset import get_lazy_dataloader

# Utility to read list of IDs and captions
def read_list_file(file_path):
    """Basic function that returns a list of IDs from a text file."""
    ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids

def read_captions_file(captions_file):
    """Returns dict: { video_id: [caption1, caption2, ...], ... }"""
    video_to_captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            vid, cap = parts
            video_to_captions.setdefault(vid, []).append(cap)
    return video_to_captions

def build_annotations(ids, captions_dict):
    """Given a list of video IDs + a dict of captions, build (vid, caption) pairs."""
    annotations = []
    for vid in ids:
        if vid in captions_dict:
            for c in captions_dict[vid]:
                annotations.append((vid, c))
    return annotations


################################################
# Training / Validation loops
################################################

def train_one_epoch(loader, c3d_encoder, clip_encoder, fusion_model, gpt2_decoder, optimizer, device):
    c3d_encoder.train() if c3d_encoder else None
    clip_encoder.train() if clip_encoder else None
    fusion_model.train()
    gpt2_decoder.train()

    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        video_feats, in_ids, att_msks = [x.to(device) for x in batch]

        # shape for C3D: (B, 16, 3, 112, 112)
        # shape for CLIP: (B, 5, 3, 224, 224) if raw, or (B, 5, 512) if pre-encoded. 
        # Adjust code as needed depending on your .pt structure.

        if c3d_encoder is not None:
            # We might need a permutation if data is (B,16,3,112,112) -> (B,3,16,112,112)
            video_feats = video_feats.permute(0,2,1,3,4)
            enc_out = c3d_encoder(video_feats)
        else:
            enc_out = clip_encoder(video_feats)

        print("enc_out.shape", enc_out.shape) # Debugging line for checking output shape
        # Now feed into fusion model
        context_embeds = fusion_model(enc_out)  # shape (B, T_ctx, 768) presumably

        # Token embeddings for text portion
        B, T = in_ids.shape
        token_embeds = gpt2_decoder.gpt2.transformer.wte(in_ids)

        # Combine
        inputs_embeds = torch.cat([context_embeds, token_embeds], dim=1)

        # Build attention mask
        context_mask = torch.ones((B, context_embeds.size(1)), device=device)
        attention_mask = torch.cat([context_mask, att_msks], dim=1)

        # Labels
        # Mark context portion with -100 so it doesn't affect loss
        labels_pad = torch.full((B, context_embeds.size(1)), -100, device=device, dtype=torch.long)
        labels = torch.cat([labels_pad, in_ids], dim=1)

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


@torch.no_grad()
def validate_one_epoch(loader, c3d_encoder, clip_encoder, fusion_model, gpt2_decoder, device):
    c3d_encoder.eval() if c3d_encoder else None
    clip_encoder.eval() if clip_encoder else None
    fusion_model.eval()
    gpt2_decoder.eval()

    total_loss = 0
    for batch in tqdm(loader, desc="Validation"):
        video_feats, in_ids, att_msks = [x.to(device) for x in batch]

        if c3d_encoder is not None:
            video_feats = video_feats.permute(0,2,1,3,4)
            enc_out = c3d_encoder(video_feats)
        else:
            enc_out = clip_encoder(video_feats)

        context_embeds = fusion_model(enc_out)
        B, T = in_ids.shape
        token_embeds = gpt2_decoder.gpt2.transformer.wte(in_ids)

        inputs_embeds = torch.cat([context_embeds, token_embeds], dim=1)
        context_mask = torch.ones((B, context_embeds.size(1)), device=device)
        attention_mask = torch.cat([context_mask, att_msks], dim=1)

        labels_pad = torch.full((B, context_embeds.size(1)), -100, device=device, dtype=torch.long)
        labels = torch.cat([labels_pad, in_ids], dim=1)

        outputs = gpt2_decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask
        )
        loss = outputs.loss

        total_loss += loss.item()

    # return total_loss / len(loader)
    return total_loss / max(len(loader), 1) # just to avoid div by zero bug


################################################
# Main
################################################

def main():
    torch.manual_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    

    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def build_annotations_from_augmented_data(pt_path):
        data = torch.load(pt_path, map_location='cpu')
        annotations = []
        for vid, entries in data.items():
            for entry in entries:
                if 'caption' in entry:
                    annotations.append((vid, entry['caption']))
        return annotations

    # train_annotations = build_annotations_from_augmented_data(Config.PROCESSED_C3D_FEATS_AUG)
    # val_annotations = build_annotations_from_augmented_data(Config.PROCESSED_C3D_FEATS)

    # # 3) Create lazy loaders
    # # Decide which .pt file to use => e.g. version1_aug.pt, version2_aug.pt, etc.
    # train_loader = get_lazy_dataloader(
    #     pt_path=Config.PROCESSED_C3D_FEATS_AUG,  # e.g., version1_aug.pt
    #     annotations=train_annotations,
    #     tokenizer=tokenizer,
    #     batch_size=Config.BATCH_SIZE,
    #     shuffle=True,
    #     limit=Config.DEBUG_LIMIT  # e.g. set to 100 for debug
    # )

    # val_loader = get_lazy_dataloader(
    #     pt_path=Config.PROCESSED_C3D_FEATS,  # base version for val
    #     annotations=val_annotations,
    #     tokenizer=tokenizer,
    #     batch_size=Config.BATCH_SIZE,
    #     shuffle=False
    # )
    train_annotations = build_annotations_from_augmented_data(Config.PROCESSED_CLIP_FEATS_AUG)
    val_annotations   = build_annotations_from_augmented_data(Config.PROCESSED_CLIP_FEATS)

    train_loader = get_lazy_dataloader(
        pt_path=Config.PROCESSED_CLIP_FEATS_AUG,
        annotations=train_annotations,
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        limit=Config.DEBUG_LIMIT
    )

    print(f"Loaded {len(val_annotations)} validation annotations")
    val_loader = get_lazy_dataloader(
        pt_path=Config.PROCESSED_CLIP_FEATS,
        annotations=val_annotations,
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )



    # 4) Build model
    USE_C3D = False  # or True if using C3D

    if USE_C3D:
        c3d_encoder = C3DEncoder().to(device)
        clip_encoder = None
        fusion_model = SimpleFusion(
            context_tokens=Config.CONTEXT_TOKENS,
            input_dim=768,  # C3D outputs 768-dim features
            frames=16       # C3D uses 16 frames
        ).to(device)
    else:
        c3d_encoder = None
        clip_encoder = CLIPEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        fusion_model = SimpleFusion(
            context_tokens=Config.CONTEXT_TOKENS,
            input_dim=768,  # CLIP features are 512-dim
            frames=5        # CLIP uses 5 frames
        ).to(device)


    gpt2_decoder = GPT2Decoder().to(device)

    # 5) Optimizer
    params_to_optimize = list(fusion_model.parameters()) + list(gpt2_decoder.parameters())
    if c3d_encoder:
        params_to_optimize += list(c3d_encoder.parameters())
    if clip_encoder and not Config.FREEZE_CLIP:
        params_to_optimize += list(clip_encoder.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=Config.LEARNING_RATE)

    # Optional: LR scheduler
    # from torch.optim.lr_scheduler import ReduceLROnPlateau
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 6) Training loop with validation
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            train_loader,
            c3d_encoder,
            clip_encoder,
            fusion_model,
            gpt2_decoder,
            optimizer,
            device
        )
        print(f'Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f}')

        val_loss = validate_one_epoch(
            val_loader,
            c3d_encoder,
            clip_encoder,
            fusion_model,
            gpt2_decoder,
            device
        )
        print(f'Epoch {epoch+1}/{Config.EPOCHS} | Val   Loss: {val_loss:.4f}')

        # if scheduler:
        #     scheduler.step(val_loss)

        # save checkpoint
        ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "c3d_encoder": c3d_encoder.state_dict() if c3d_encoder else None,
            "clip_encoder": clip_encoder.state_dict() if clip_encoder else None,
            "fusion_model": fusion_model.state_dict(),
            "gpt2_decoder": gpt2_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
