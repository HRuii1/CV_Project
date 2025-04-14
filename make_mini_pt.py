import torch
import cv2
import os
from PIL import Image
from torchvision import transforms as T

# Suppose you put your two raw videos in a folder "videos/"
# with names "cat_video.avi" and "dog_video.avi"
# We'll call them "vidA" and "vidB" for demonstration.

VIDEO_FOLDER = "videos"
video_paths = {
    "vidA": os.path.join(VIDEO_FOLDER, "cat_video.avi"),
    "vidB": os.path.join(VIDEO_FOLDER, "dog_video.avi")
}

# Optionally define some helper function to extract frames
def extract_frames(video_path, num_frames=16, size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames or total <= 0:
        # Not enough frames or invalid
        cap.release()
        return None
    step = total / float(num_frames - 1)
    frames = []
    for i in range(num_frames):
        frame_idx = int(round(i * step))
        frame_idx = min(frame_idx, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    # resize + normalize
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])
    frames_tensor = torch.stack([transform(f) for f in frames], dim=0)  # (16, 3, H, W)
    return frames_tensor

mini_data = {}

# We'll store a list of dicts for each video:
# [
#   { "caption": ..., "data": ... },
#   { "caption": ..., "data": ... },
#   ...
# ]

# Define your custom captions here:
captions_for_vidA = [
    "A cat is jumping",
    "The cat leaps across the floor"
]
captions_for_vidB = [
    "A dog is barking",
    "The dog stands near a door"
]

for vid_id, path in video_paths.items():
    frames_16 = extract_frames(path, num_frames=16, size=(112,112))
    if frames_16 is None:
        print(f"Skipping {vid_id} - not enough frames or invalid video.")
        continue
    # each entry has a 'caption' and 'data'
    # Suppose you define 2 captions per video
    if vid_id == "vidA":
        caps = captions_for_vidA
    else:
        caps = captions_for_vidB

    entries = []
    for caption_text in caps:
        # Our C3D approach wants shape: (16,3,112,112)
        # We'll store frames_16 as is. 
        # If you have augmentations, do them here.
        entries.append({
            "caption": caption_text,
            "data": frames_16.clone()  # or any augmentation
        })
    mini_data[vid_id] = entries

# Finally, save to "mini.pt"
torch.save(mini_data, "mini.pt")
print("Saved 2-video dataset to mini.pt")
