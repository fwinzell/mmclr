import json
import os
import random
import re
import pandas as pd
import SimpleITK
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
import math

import glob

from torch.utils.data import Dataset

# Debug import hack
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import MM_Dataset
from datasets.transformation import TransformMRI, FroFA

class CLRDataset(MM_Dataset):
    def __init__(self, 
                 clinical_data_folder, 
                 pathology_dir, 
                 radiology_dir,
                 pids,
                 clinical_exclude_keys=["BCR", "BCR_PSA", "time_to_follow-up/BCR"],
                 wsi_crop_grid=48, 
                 feature_extractor="unet",
                 mri_input_size=None,
                 mri_include=["t2w"],
                 clinical_dropout=0.5,
                 mri_dropout=0.25,
                 use_frofa=True,
                 verbose=False
                 ):
        """
        Dataset for CLR training
        - Returns two inputs, a key with full data including all modalitites
          and a query where modalities are masked and clinical variables are dropped
        *Args
        clinical_data_folder: path to clinical data
        pathology_dir: path to folder with UNI representations
        radiology_dir: path to MRI data or nnUnet features
        pids: list of patient ids to include
        clinical_exclude_keys: list of keys to remove from input, e.g. BCR
        wsi_crop_grid: grid size of cropping WSI in terms of number of patches, e.g. select NxN number of 224x224 patches
        clinical_dropout: probability of a clinical variable being dropped in the query
        feature_extractor: which feature extractor to use for the MRI data (default: unet)
        mri_input_size: if not None, the mri volumes will be cropped to this size, does not have an effect if features are used directly

        Returns
        query (masked/dropout input to student) and key (full data to teacher)
        """
        self.verbose = verbose
        if self.verbose: print("Initializing multi-modal dataset...")

        super().__init__(clinical_data_folder=clinical_data_folder,
                         clinical_exclude_keys=clinical_exclude_keys,
                         pathology_dir=pathology_dir,
                         wsi_crop_grid=wsi_crop_grid,
                         radiology_dir=radiology_dir,
                         pids=pids,
                         mri_input_size=mri_input_size,
                         feature_extractor=feature_extractor)
        
        if self.verbose: print("Done")
        
        self.clinical_dropout = clinical_dropout

        self.mri_transform = TransformMRI(apply_query_mask=True, modality_keys=mri_include,
                                          num_ghosts=2, flip_probability=0.5)
        self.use_frofa = use_frofa
        if use_frofa:
            self.frofa = FroFA(channel_wise=True)


    def dropout(self,x):
        """ Dropout augmentation for clinical features"""
        mask = (torch.rand_like(x) > self.clinical_dropout).float()
        return x*mask
    
    def modality_mask(self, query):
        """ Random masking of one modality """
        n_mods = len(query)
        drop_idx = random.randint(0, n_mods-1)
        masked_query = list(query)
        if isinstance(query[drop_idx], dict):
            masked_query[drop_idx] = {k: torch.zeros_like(v) for k, v in query[drop_idx].items()}
        else:
            masked_query[drop_idx] = torch.zeros_like(query[drop_idx])
        return tuple(masked_query)
        
    def __getitem__(self, idx):
        pid = self.pids[idx]
        
        if self.verbose: print("Loading clinical data...")
        clinical_data = self.df[self.df['id'] == int(pid)].drop(columns=['id']).values
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32).squeeze(0)
        if self.verbose: print("Done")

        if self.verbose: print("Loading pathology data...")
        pathology_key, _ = self._get_random_fixed_patches(pid)

        pathology_query, _ = self._get_random_fixed_patches(pid)
        if self.use_frofa:
            pathology_query = self.frofa(pathology_query)
            pathology_query = pathology_query.squeeze()
        if self.verbose: print("Done")
        

        if self.verbose: print("Loading radiology data...")
        if self.radiology_dir is not None:
            if self.feature_extractor == "unet":
                radiology_data = self._get_unet_features(pid)
            elif self.feature_extractor == "resnet":
                radiology_data = self._get_resnet_embs(pid, mask=True)
            else:
                radiology_data = self._get_mri_vols(pid)
        else:
            radiology_data = None
        if self.verbose: print("Done")

        if self.verbose: print("Applying transformations...")
        clinical_query = self.dropout(clinical_data)
        radiology_query, radiology_key = self.mri_transform(radiology_data)
        
        input_key = (clinical_data, radiology_key, pathology_key)
        input_query = self.modality_mask(query=(clinical_query, radiology_query, pathology_query))
        if self.verbose: print("Done")

        return input_query, input_key
        
        
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    parser = argparse.ArgumentParser(description='Test Models Main Script')
    parser.add_argument('-c', '--cluster', action='store_true', help='Flag for being on the cluster')
    args = parser.parse_args()
    if args.cluster:
        # SOL paths
        print("Cluster")
        main_dir = "/data/pa_cpgarchive/projects/chimera/_aws/task1/"
        keyfile = "/data/pa_cpgarchive//projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"
    else:
        # local mac paths
        print("Local")
        main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/"
        keyfile = "/Volumes/PA_CPGARCHIVE/projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    clinical_folder = os.path.join(main_dir, "clinical_data")
    patient_info = pd.read_excel(keyfile, usecols=["miccai_id", "Split", "BCR", "BCR_PSA"])

    train_ids = patient_info.loc[patient_info["Split"] == "train", "miccai_id"].tolist()


    dataset = CLRDataset(
        clinical_data_folder=clinical_folder,
        pathology_dir=os.path.join(main_dir, "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(main_dir, "radiology/images"),
        pids=train_ids,
        feature_extractor="resnet",
        mri_input_size=(20, 128, 120),
        mri_include=['t2w', 'adc', 'hbv']
    )   

    loader = DataLoader(dataset, batch_size=2)
    for query, key in loader:

        print(f"MRI shapes: t2w: {key[1]['t2w'].shape}, adc: {key[1]['adc'].shape}")
