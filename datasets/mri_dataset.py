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


class MRIDataset(Dataset):
    def __init__(self,
                 radiology_dir,
                 pids=None):
        self.radiology_dir = radiology_dir
        self.pids = pids if pids is not None else self._load_patient_ids()

    def _load_patient_ids(self):
        # Load patient IDs from the radiology directory
        return [f for f in os.listdir(self.radiology_dir) if os.path.isdir(os.path.join(self.radiology_dir, f))]
    
    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        folder = os.path.join(self.radiology_dir, pid)

        mri_vols = os.listdir(folder)
        if len(mri_vols) < 4:
            raise ValueError(f"Not enough MRI volumes for patient {pid}")

        x_adc = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_adc.mha"))[0])
        x_adc = SimpleITK.GetArrayFromImage(x_adc).astype(np.float32)

        x_hbv = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_hbv.mha"))[0])
        x_hbv = SimpleITK.GetArrayFromImage(x_hbv).astype(np.float32)   

        x_t2w = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_t2w.mha"))[0])
        x_t2w = SimpleITK.GetArrayFromImage(x_t2w).astype(np.float32)

        x_mask = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_mask.mha"))[0])
        x_mask = SimpleITK.GetArrayFromImage(x_mask)

        idxs = np.where(x_mask == 1)
    
        ymid = int(np.median(np.sort(idxs[1])))
        xmid = int(np.median(np.sort(idxs[2])))

        x_t2w = x_t2w[:, ymid-64:ymid+64, xmid-60:xmid+60]

        return {
            'pid': pid,
            'adc': torch.tensor(x_adc, dtype=torch.float32).unsqueeze(0),
            'hbv': torch.tensor(x_hbv, dtype=torch.float32).unsqueeze(0),
            't2w': torch.tensor(x_t2w, dtype=torch.float32).unsqueeze(0)
        }



if __name__ == "__main__":
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
    dataset = MRIDataset(radiology_dir=main_dir)
    print(f"Number of patients in dataset: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"ADC shape: {sample['adc'].shape}")
    print(f"HBV shape: {sample['hbv'].shape}")
    print(f"T2W shape: {sample['t2w'].shape}")
    print(f"Patient ID: {sample['pid']}")
