import os
import torch
from torch.utils.data import Dataset, DataLoader


def read_list_file(list_file):
    """Reads a list file (e.g., train_list.txt) and returns a set of video IDs."""
    ids = set()
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            vid = line.strip()
            if vid:
                ids.add(vid)
    return ids


class MSVDPTDataset(Dataset):
    """
    A dataset that loads all features from a .pt dictionary and returns (video_features, caption_text).
    'video_features' is a tensor from the preprocessed .pt file (e.g., version1.pt)
    """
    def __init__(self, feature_dict, annotations, tokenizer, max_length=30):
        """
        feature_dict: a dict mapping video_id -> tensor
        annotations: list of tuples (video_id, caption_str)
        tokenizer: GPT2 tokenizer
        max_length: maximum length of tokens
        """
        super().__init__()
        self.feature_dict = feature_dict
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_id, caption_str = self.annotations[idx]

        # Get preloaded tensor from dictionary
        video_features = self.feature_dict[video_id]  # Tensor, no need to load file
        video_features = video_features.float()

        # Tokenize the caption
        tokens = self.tokenizer(
            caption_str,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return video_features, input_ids, attention_mask

def collate_fn(batch):
    video_feats, in_ids, att_msks = zip(*batch)
    video_feats = torch.stack(video_feats, dim=0)

    max_len = max(x.size(0) for x in in_ids)
    padded_in_ids = []
    padded_att_msks = []

    for i in range(len(in_ids)):
        seq_len = in_ids[i].size(0)
        pad_len = max_len - seq_len
        padded_in_ids.append(
            torch.cat([in_ids[i], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_att_msks.append(
            torch.cat([att_msks[i], torch.zeros(pad_len, dtype=torch.long)])
        )

    padded_in_ids = torch.stack(padded_in_ids, dim=0)
    padded_att_msks = torch.stack(padded_att_msks, dim=0)

    return video_feats, padded_in_ids, padded_att_msks

def get_pt_dataloader(
    feature_pt_path, split_name, annotations, tokenizer, batch_size=8, shuffle=True
):
    feature_dict = torch.load(feature_pt_path)[split_name]  # e.g. "train", "val", "test"
    dataset = MSVDPTDataset(feature_dict, annotations, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return loader
