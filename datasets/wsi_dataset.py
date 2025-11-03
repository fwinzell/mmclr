import json
import os
import pandas as pd
import pyvips
import SimpleITK
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch

import glob
from pathlib import Path
import random

from torch.utils.data import Dataset

class UNIDataset(Dataset):
    def __init__(self, 
                 features_dir,
                 crop_size,
                 pids=None):
        
        self.features_dir = features_dir
        self.crop_size = crop_size  
        self.pids = pids if pids is not None else self._get_all_pids()

    def _get_all_pids(self):
        files = Path(os.path.join(self.features_dir, "features")).glob("*.pt")
        pids = {f.stem.split("_")[0] for f in files}
        return list(pids)

    def __len__(self):
        return len(self.pids)
    
    def _get_random_crop(self, pid):
        feature_paths = glob.glob(os.path.join(self.features_dir, "features", f"{pid}_*.pt"))
        coord_paths = glob.glob(os.path.join(self.features_dir, "coordinates", f"{pid}_*.npy"))

        idx = np.random.randint(0, len(feature_paths))
        feature_path = feature_paths[idx]
        coord_path = coord_paths[idx]

        features = torch.load(feature_path)  # Shape: (N, D)
        coordinates = np.load(coord_path)  # Shape: (N, 2)

        coords = np.array(coordinates.tolist())[:,0:2].astype(int)
        x_max = max(coords[:, 0])
        y_max = max(coords[:, 1])

        valid = np.where((coords[:, 0] <= x_max - self.crop_size) &
                         (coords[:, 1] <= y_max - self.crop_size))[0]
        ridx = np.random.choice(valid)

        x_start = coords[ridx, 0]
        y_start = coords[ridx, 1]
        crop_indices = np.where((coords[:, 0] >= x_start) & (coords[:, 0] < x_start + self.crop_size) &
                                (coords[:, 1] >= y_start) & (coords[:, 1] < y_start + self.crop_size))[0]
        
        features = features[crop_indices]
        coordinates = coordinates[crop_indices]

        return features, coordinates


    def __getitem__(self, idx):
        pid = self.pids[idx]

        features, coords = self._get_random_crop(pid)

        return features, coords
    

if __name__ == "__main__":
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/pathology/features"

    dataset = UNIDataset(features_dir=main_dir,
                         crop_size=20000)
    
    features, coords = dataset[0]
    print(f"Features shape: {features.shape}")
    print(f"Coordinates shape: {coords.shape}")
