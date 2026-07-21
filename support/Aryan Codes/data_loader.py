import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class OriginalXBDDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        """
        Loads the original High-Resolution RGB xBD images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Only load POST-disaster images as per your requirement
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('post_disaster.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load the original xBD image (1024x1024 RGB .png)
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load the Mask (1024x1024 .png)
        # Assuming masks are 2D arrays where pixel values are classes (0, 1, 2, 3, 4)
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. Label Simplification (From Paper Section 2.1.3)
        # 0=Background, 1=No Damage, 2=Minor, 3=Major, 4=Destroyed
        # Mapping to: 0=Background, 1=Intact, 2=Damaged
        simplified_mask = np.zeros_like(mask)
        simplified_mask[mask == 1] = 1 # Intact
        simplified_mask[mask >= 2] = 2 # Minor, Major, Destroyed -> Damaged
        
        # 4. Preprocessing: Standard 0-1 normalization for RGB PNGs
        image = image.astype(np.float32) / 255.0
        
        # PyTorch format: [Channels, Height, Width]
        image = np.transpose(image, (2, 0, 1))
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(simplified_mask, dtype=torch.long)
