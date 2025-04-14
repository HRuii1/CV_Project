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
        frame_feats: (B, num_frames, 512) if they've already been pre-extracted
                     or if you want to use the model to extract, pass raw images.
        Returns a single 2D tensor (B, num_frames, 768).
        """
        # If you have raw images, you'd do: self.model.encode_image(...)
        # but here we assume 'frame_feats' are already the 512-dim CLIP embeddings per frame.
        B, F, C = frame_feats.size()  # typically (batch_size, 5, 512)
        frame_feats = frame_feats.to(dtype=torch.float32)  
        out = []
        for i in range(F):
            # For each frame, project to 768
            projected = self.proj(frame_feats[:, i, :])  # (B, 512) -> (B, 768)
            out.append(projected.unsqueeze(1))
        out = torch.cat(out, dim=1)  # shape (B, F, 768)
        return out
