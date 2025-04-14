# video-captioning/models/fusion.py

import torch
import torch.nn as nn
from config import Config

class SimpleFusion(nn.Module):
    """
    Projects features from either CLIP (B, F, 512) or C3D (B, 768*F) into
    (B, context_tokens, 768) format suitable for GPT-2 prefix embeddings.
    """
    def __init__(self, context_tokens=Config.CONTEXT_TOKENS, input_dim=None, frames=None):
        super().__init__()
        self.context_tokens = context_tokens
        self.input_dim = input_dim      # 512 for CLIP, 768 for C3D
        self.frames = frames            # 5 for CLIP, 16 for C3D

        if input_dim is None or frames is None:
            raise ValueError("Must specify both `input_dim` and `frames`.")

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * frames, 768 * context_tokens),
            nn.Tanh()
        )

    def forward(self, video_emb):
        """
        - video_emb.shape == (B, F, D) for CLIP (e.g., 5, 512)
        - video_emb.shape == (B, D*F) for C3D flattened (e.g., 16*768)
        """
        if len(video_emb.shape) == 2:
            # Already flattened (C3D)
            out = self.mlp(video_emb)
        elif len(video_emb.shape) == 3:
            B, F, D = video_emb.shape
            out = video_emb.view(B, F * D)
            out = self.mlp(out)
        else:
            raise ValueError("Expected 2D or 3D input to SimpleFusion.")

        return out.view(video_emb.size(0), self.context_tokens, 768)
