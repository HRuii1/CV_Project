import torch
from torch.utils.data import Dataset, DataLoader

class LazyAugDataset(Dataset):
    """
    Expects a .pt file structure like:
      {
        video_id: [ 
           { 'caption': some_text, 'data': tensor(...), ... },
           { 'caption': some_text, 'data': tensor(...), ... },
           ...
        ],
        ...
      }
    and an annotations list of (video_id, caption).
    """

    def __init__(self, pt_path, annotations, tokenizer, max_length=32, limit=None):
        super().__init__()
        self.pt_path = pt_path
        self.annotations = annotations[:limit] if limit else annotations
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load only a *subset* of the data structure keys into memory
        print(f"Loading index from {pt_path} (lazy)...")
        full_data = torch.load(pt_path, map_location='cpu')  # type: dict
        self.video_dict = {}
        for (vid, _) in self.annotations:
            # If video not in file, skip
            if vid not in full_data:
                continue
            self.video_dict[vid] = full_data[vid]
        del full_data  # free memory

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        vid, caption = self.annotations[idx]
        # In case the video is missing
        if vid not in self.video_dict:
            # Return dummy zero features
            return torch.zeros(1), torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)

        # Each vid entry is a list of { 'caption': x, 'data': Tensor(...), ... }
        video_entries = self.video_dict[vid]

        # We find the first matching entry with the same caption
        # or default to the 0th entry if not found
        matched = None
        for item in video_entries:
            if item.get('caption') == caption:
                matched = item
                break
        if matched is None:
            matched = video_entries[0]

        # matched['data'] is either shape (16,3,112,112) or (5,512), etc.
        video_tensor = matched['data']  # shape depends on version1 or version2

        # Convert the caption into tokens
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        return video_tensor, input_ids, attention_mask

def get_lazy_dataloader(pt_path, annotations, tokenizer, batch_size=4, shuffle=True, limit=None):
    dataset = LazyAugDataset(pt_path, annotations, tokenizer, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
