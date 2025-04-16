# video-captioning/models/clip_model.py

import torch
import torch.nn as nn
import clip
from config import Config

class CLIPEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", freeze_clip=True):
        super().__init__()
        self.model, _ = clip.load(model_name, device=Config.DEVICE)
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # MLP mapping from 512 -> 768 * N_TOKENS_PER_FRAME
        # but simpler: we'll do 512 -> 768 repeated #frames times inside a loop
        self.proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.Tanh()
        )
        
    def forward(self, frame_feats):
        """
        Handles two types of inputs:
        1. Raw frames (B, F, 3, 224, 224) — use CLIP's model.encode_image
        2. Precomputed features (B, F, 512)
        """
        if frame_feats.dim() == 5:
            B, F, C, H, W = frame_feats.shape
            feats = []
            for i in range(F):
                img_batch = frame_feats[:, i, :, :, :]  # (B, 3, 224, 224)
                with torch.no_grad():
                    feat = self.model.encode_image(img_batch)  # (B, 512)
                feats.append(feat.unsqueeze(1))  # (B, 1, 512)
            frame_feats = torch.cat(feats, dim=1)  # (B, F, 512)

        # Project to 768
        B, F, C = frame_feats.size()
        frame_feats = frame_feats.to(dtype=torch.float32)  
        out = []
        for i in range(F):
            projected = self.proj(frame_feats[:, i, :])  # (B, 512) -> (B, 768)
            out.append(projected.unsqueeze(1))
        return torch.cat(out, dim=1)  # (B, F, 768)
