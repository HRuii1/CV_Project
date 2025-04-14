import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import Config
from models.c3d_model import C3DEncoder
from models.clip_model import CLIPEncoder
from models.fusion import SimpleFusion
from models.gpt2_decoder import GPT2Decoder

from lazy_dataset import get_lazy_dataloader

def build_annotations_from_augmented_data(pt_path):
    """
    Load data from something like version2_aug.pt 
    which has structure:
      {
        video_id: [
          { 'caption': some_str, 'data': Tensor(...), ...},
          ...
        ],
        ...
      }
    and return a list of (vid, caption).
    """
    data = torch.load(pt_path, map_location='cpu')
    annotations = []
    for vid, entries in data.items():
        for entry in entries:
            if 'caption' in entry:
                annotations.append((vid, entry['caption']))
    return annotations

def train_one_epoch(loader, c3d_encoder, clip_encoder, fusion_model, gpt2_decoder, optimizer, device):
    c3d_encoder.train() if c3d_encoder else None
    clip_encoder.train() if clip_encoder else None
    fusion_model.train()
    gpt2_decoder.train()

    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        video_feats, in_ids, att_msks = [x.to(device) for x in batch]

        # If using C3D:
        # shape: (B,16,3,112,112), might do .permute(0,2,1,3,4)
        # If using CLIP: shape: (B,5,512) or something similar

        if c3d_encoder is not None:
            video_feats = video_feats.permute(0,2,1,3,4)
            enc_out = c3d_encoder(video_feats)
        else:
            enc_out = clip_encoder(video_feats)

        # We now pass enc_out to fusion
        context_embeds = fusion_model(enc_out)

        B, T = in_ids.shape
        token_embeds = gpt2_decoder.gpt2.transformer.wte(in_ids)

        inputs_embeds = torch.cat([context_embeds, token_embeds], dim=1)

        context_mask = torch.ones((B, context_embeds.size(1)), device=device)
        attention_mask = torch.cat([context_mask, att_msks], dim=1)

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

def main():
    torch.manual_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    
    tokenizer = GPT2Tokenizer.from_pretrained(Config.GPT2_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Build training annotations from CLIP *augmented* .pt
    # which has actual captions for train only
    train_annotations = build_annotations_from_augmented_data(Config.PROCESSED_CLIP_FEATS_AUG)
    print(f"Loaded {len(train_annotations)} training samples from {Config.PROCESSED_CLIP_FEATS_AUG}")

    # Build train loader
    train_loader = get_lazy_dataloader(
        pt_path=Config.PROCESSED_CLIP_FEATS_AUG,
        annotations=train_annotations,
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        limit=Config.DEBUG_LIMIT
    )

    # We SKIP validation here: no val_loader, no references
    # if you want a dummy "val_loss" to track, you can set val_loss=None.

    USE_C3D = False  # CLIP-based training
    if USE_C3D:
        c3d_encoder = C3DEncoder().to(device)
        clip_encoder = None
        fusion_model = SimpleFusion(
            context_tokens=Config.CONTEXT_TOKENS,
            input_dim=768, 
            frames=16       
        ).to(device)
    else:
        c3d_encoder = None
        clip_encoder = CLIPEncoder(freeze_clip=Config.FREEZE_CLIP).to(device)
        fusion_model = SimpleFusion(
            context_tokens=Config.CONTEXT_TOKENS,
            input_dim=768, 
            frames=5       
        ).to(device)

    gpt2_decoder = GPT2Decoder().to(device)

    # Collect all trainable params
    params_to_optimize = list(fusion_model.parameters()) + list(gpt2_decoder.parameters())
    if c3d_encoder:
        params_to_optimize += list(c3d_encoder.parameters())
    if clip_encoder and not Config.FREEZE_CLIP:
        params_to_optimize += list(clip_encoder.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=Config.LEARNING_RATE)

    # Training loop (no validation)
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
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(Config.OUTPUT_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "c3d_encoder": c3d_encoder.state_dict() if c3d_encoder else None,
            "clip_encoder": clip_encoder.state_dict() if clip_encoder else None,
            "fusion_model": fusion_model.state_dict(),
            "gpt2_decoder": gpt2_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
