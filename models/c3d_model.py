# video-captioning/models/c3d_model.py

import torch
import torch.nn as nn

class C3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Following a simplified version of C3D
        # Typically, C3D can have 8 convolution layers, 5 pool layers, 3 FC layers, etc.
        # We'll define a smaller version for illustration.
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        
        self.fc4 = nn.Linear(256*4*14*14, 2048)
        self.fc5 = nn.Linear(2048, 15360)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Dynamically infer flattened size
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
