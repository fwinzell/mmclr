import json
import os
import random
import re
import pandas as pd
import SimpleITK
import numpy as np
import matplotlib.pyplot as plt
import math

import cv2
import torch
from copy import deepcopy

import glob

from torch.utils.data import Dataset
#from monai.transforms import CenterSpatialCrop, SpatialPad
import torch.nn.functional as F
import torchio

from datasets.transformation import TransformMRI, FroFA

class MM_Dataset(Dataset):
    def __init__(self, 
                 clinical_data_folder, 
                 pathology_dir, 
                 radiology_dir,
                 pids,
                 clinical_exclude_keys=["BCR", "BCR_PSA", "time_to_follow-up/BCR"],
                 wsi_crop_size=20000,
                 wsi_crop_grid=48,
                 mri_input_size=None,
                 feature_extractor="none"):
        
        self.clinical_data_folder = clinical_data_folder
        self.pathology_dir = pathology_dir
        self.radiology_dir = radiology_dir
        self.pids = pids
        self.feature_extractor = feature_extractor
        self.mri_size = mri_input_size

        self.data_files = [f for f in os.listdir(clinical_data_folder) if f.endswith('.json')]

        self.df = self._get_data_frame(clinical_exclude_keys)
        #self.print_head()

        self._ensure_numeric()
        #self._validate_pids()

        self.num_attributes = self.df.shape[1]

        self.wsi_crop_size = wsi_crop_size 
        self.crop_grid = wsi_crop_grid
        
    
    def _get_data_frame(self, exclude_keys):
        # For clinical data, get all available patients
        # Args:
        #   exclude_keys: list of keys to exclude from the dataframe
        # Returns:
        #   df: pandas dataframe with clinical data
        from sklearn.preprocessing import LabelEncoder

        df = pd.DataFrame()

        for f in self.data_files:
            with open(os.path.join(self.clinical_data_folder, f), 'r') as file:
                data = json.load(file)
                id = f.split(".json")[0]
                data['id'] = int(id)
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        # Set missing values ('x') to 2
        for col in ['positive_lymph_nodes', 'capsular_penetration', 'positive_surgical_margins', 'invasion_seminal_vesicles']:
            if col in df.columns:
                df[col] = df[col].replace({'x': 2}).astype(int)
            else:
                print(f"Warning: Column {col} not found in clinical data")

        # categorical keys to encode
        for col in ["pT_stage", "earlier_therapy"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        if "BCR" in df.columns:
            df["BCR"] = np.array(df["BCR"]).astype(float).astype(int)
        if "BCR_PSA" in df.columns:
            df["BCR_PSA"] = pd.to_numeric(df["BCR_PSA"], errors='coerce').fillna(-1.0)
        df["tertiary_gleason"] = pd.to_numeric(df["tertiary_gleason"], errors='coerce').fillna(-1)

        for key in exclude_keys or []:
            if key in df.columns:
                df = df.drop(columns=[key])

        df_encoded = pd.get_dummies(df)

        return df_encoded
    
    def _ensure_numeric(self):
        for col in self.df.columns:
            if not np.issubdtype(self.df[col].dtype, np.number):
                if self.df[col].dtype == bool:
                    #print(f"Converting column {col} from bool to int")
                    self.df[col] = self.df[col].astype(int)
                else:
                    #print(f"Converting column {col} to numeric")
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def _validate_pids(self):
        # Remove pids with missing pathology data
        for pid in self.pids:
            feature_paths = glob.glob(os.path.join(self.pathology_dir, "features", f"{pid}_*.pt"))
            if len(feature_paths) == 0:
                print(f"No pathology features found for pid {pid} in {self.pathology_dir}, removing...")
                self.pids.remove(pid)

    
    def _get_two_crops(self, pid):
        # For the pathology data of patient [pid], return two views of a random WSI crop
        # 1. Select random global crop
        # 2. Sample two local crops within the global
        # 3. Handle missing features: 
        #   - Sample K patches with replacement
        #   - Missing feature token
        # 
        # Need to make sure that there are atleast a minimal number of patches within the global crop?
        # 
        feature_paths = glob.glob(os.path.join(self.pathology_dir, "features", f"{pid}_*.pt"))
        if len(feature_paths) == 0:
            print(f"No pathology features found for pid {pid} in {self.pathology_dir}")
            return torch.zeros((1, 1024)), None
        
        feature_path = random.choice(feature_paths)

        idx = feature_path.split("_")[-1].split(".pt")[0]

        coord_path = os.path.join(self.pathology_dir, "coordinates", f"{pid}_{idx}.npy")

        features = torch.load(feature_path, map_location="cpu").half()  # Shape: (N, D)
        #coordinates = np.load(coord_path)  
        coords = np.array(np.load(coord_path).tolist())[:,0:2].astype(int) # Shape: (N, 2)

        assert features.shape[0] == coords.shape[0], f"Number of features ({features.shape}) and coordinates ({coords.shape}) must match"

        # 1. Select random global crop
        x_max = max(coords[:, 0])
        y_max = max(coords[:, 1])

        valid = np.where((coords[:, 0] <= x_max - self.wsi_global_crop_size) &
                         (coords[:, 1] <= y_max - self.wsi_global_crop_size))[0]
        ridx = np.random.choice(valid)

        x1, y1 = coords[ridx, :]
        x2, y2 = (x1, y1) + self.wsi_global_crop_size - self.wsi_local_crop_size
        
        # Possible coordinates for top left corner of local crops
        local_start_idxs  = np.where((
            (coords[:,0] >= x1) & (coords[:,0] < x2) &
            (coords[:,1] >= y1) & (coords[:,1] < y2)
        ))[0]

        starts = np.random.choice(local_start_idxs, size=2, replace=False)

        featureList = []
        for start_idx in starts:
            x1, y1 = coords[start_idx, :]
            x2, y2 = (x1, y1) + self.wsi_local_crop_size
            
            # Possible coordinates for top left corner of local crops
            local_crop_idxs  = np.where((
                (coords[:,0] >= x1) & (coords[:,0] < x2) &
                (coords[:,1] >= y1) & (coords[:,1] < y2)
            ))[0]
            featureList.append(features[local_crop_idxs])
        return featureList
 
            
    def _get_random_crop(self, pid):
        # For pathology data, get a random crop of features for the given patient ID
        # Args:
        #   pid: patient ID
        # Returns:
        #   features: torch tensor of shape (N, D) where N is number of patches in the crop and D is feature dimension
        feature_paths = glob.glob(os.path.join(self.pathology_dir, "features", f"{pid}_*.pt"))
        if len(feature_paths) == 0:
            print(f"No pathology features found for pid {pid} in {self.pathology_dir}")
            return torch.zeros((1, 1024)), None
        
        feature_path = random.choice(feature_paths)
        
        idx = feature_path.split("_")[-1].split(".pt")[0]

        coord_path = os.path.join(self.pathology_dir, "coordinates", f"{pid}_{idx}.npy")

        features = torch.load(feature_path)  # Shape: (N, D)
        coordinates = np.load(coord_path)  # Shape: (N, 2)

        """if features.shape[0] != coordinates.shape[0]:
            print(f"Warning: Number of features ({features.shape}) and coordinates ({coordinates.shape}) do not match for pid {pid}")
            print(f"Feature path: {feature_path}")
            print(f"Coordinate path: {coord_path}")
            print(idx)
            print(f"Features: {feature_paths}")
            print(f"Coordinates: {coord_paths}")"""

        assert features.shape[0] == coordinates.shape[0], f"Number of features ({features.shape}) and coordinates ({coordinates.shape}) must match"

        coords = np.array(coordinates.tolist())[:,0:2].astype(int)
        x_max = max(coords[:, 0])
        y_max = max(coords[:, 1])

        valid = np.where((coords[:, 0] <= x_max - self.wsi_crop_size) &
                         (coords[:, 1] <= y_max - self.wsi_crop_size))[0]
        ridx = np.random.choice(valid)

        x_start = coords[ridx, 0]
        y_start = coords[ridx, 1]
        crop_indices = np.where((coords[:, 0] >= x_start) & (coords[:, 0] < x_start + self.wsi_crop_size) &
                                (coords[:, 1] >= y_start) & (coords[:, 1] < y_start + self.wsi_crop_size))[0]

        try:
            features = features[crop_indices]
        except IndexError:
            print(f"Error cropping features for pid {pid} at indices {crop_indices}")
            print(f"Crop area: x[{x_start}:{x_start + self.wsi_crop_size}], y[{y_start}:{y_start + self.wsi_crop_size}]")
            features = features[0:1000]

        coordinates = coordinates[crop_indices]

        return features, coordinates
    
    def _get_random_fixed_patches(self, pid):
        feature_paths = glob.glob(os.path.join(self.pathology_dir, "features", f"{pid}_*.pt"))

        if len(feature_paths) == 0:
            print(f"No pathology features found for pid {pid} in {self.pathology_dir}")
            return torch.zeros((self.crop_grid * self.crop_grid, 1024)), None

        feature_path = random.choice(feature_paths)
        
        idx = feature_path.split("_")[-1].split(".pt")[0]

        coord_path = os.path.join(self.pathology_dir, "coordinates", f"{pid}_{idx}.npy")

        features = torch.load(feature_path, map_location="cpu").half()  # Shape: (N, D)
        #coordinates = np.load(coord_path)  
        coordinates = np.array(np.load(coord_path).tolist())[:,0:2].astype(int) # Shape: (N, 2)

        num_patches_needed = self.crop_grid * self.crop_grid

        assert features.shape[0] == coordinates.shape[0], f"Number of features ({features.shape}) and coordinates ({coordinates.shape}) must match"
        
        idx = np.random.randint(0, features.shape[0]-num_patches_needed)
        
        features = features[idx:idx+num_patches_needed].float()
        coordinates = coordinates[idx:idx+num_patches_needed, :]

        return features, coordinates
    
    def _crop_and_pad(self, mimg):
        while mimg.ndim < 4:
            mimg = mimg.unsqueeze(0)

        crop_pad = torchio.CropOrPad(self.mri_size)
        img = crop_pad(mimg)

        return img

    
    def _get_mri_vols(self, pid, mask=True):
        # Loads all three MRI volumes for a given patient ID
        # Args:
        #   pid: patient ID
        #   mask: Boolean, mask t2w to make it the same size as the others
        # Returns:
        #   dict with keys 'adc', 'hbv', 't2w' and corresponding MRI volumes as torch tensors
        folder = os.path.join(self.radiology_dir, str(pid))

        # If folder does not exist, return no mri data? For now raise error
        if not os.path.exists(folder):
            raise ValueError(f"Radiology path: {folder} does not exist")    
            #return None

        mri_vols = os.listdir(folder)
        if len(mri_vols) < 4:
            raise ValueError(f"Not enough MRI volumes for patient {pid}")

        x_adc = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_adc.mha"))[0])
        x_adc = SimpleITK.GetArrayFromImage(x_adc).astype(np.float32)

        x_hbv = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_hbv.mha"))[0])
        x_hbv = SimpleITK.GetArrayFromImage(x_hbv).astype(np.float32)   

        x_t2w = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_t2w.mha"))[0])
        x_t2w = SimpleITK.GetArrayFromImage(x_t2w).astype(np.float32)

        if mask:
            x_mask = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_mask.mha"))[0])
            x_mask = SimpleITK.GetArrayFromImage(x_mask)

            idxs = np.where(x_mask == 1)
        
            ymid = int(np.median(np.sort(idxs[1])))
            xmid = int(np.median(np.sort(idxs[2])))

            x_t2w = x_t2w[:, ymid-64:ymid+64, xmid-60:xmid+60]

        x_adc = torch.tensor(x_adc, dtype=torch.float32)
        x_hbv = torch.tensor(x_hbv, dtype=torch.float32)
        x_t2w = torch.tensor(x_t2w, dtype=torch.float32)

        if self.mri_size is not None:
            x_adc = self._crop_and_pad(x_adc)
            x_hbv = self._crop_and_pad(x_hbv)
            x_t2w = self._crop_and_pad(x_t2w)

        return {
            'adc': x_adc,
            'hbv': x_hbv,
            't2w': x_t2w
        }
    
    def _get_resnet_embs(self, pid, mask):
        # Run this method to load pre-computed resnet features
        # Not yet implemented
        return self._get_mri_vols(pid, mask=mask)

    
    def _get_unet_features(self, pid, pad=True):
         # Loads features from nnUnet of MRI data
        # Args:
        #   pid: patient ID
        # Returns:
        #   features as torch tensors
        file = os.path.join(self.radiology_dir, f"{pid}_0001_features.pt")

        # If folder does not exist, return no mri data
        if not os.path.exists(file):
            #raise FileNotFoundError(f"File {file} not found")
            feats = torch.zeros((1024))  # Return zero features if file not found
        else:
            feats = torch.load(file)

        if pad:
            # Pad to size 1024 with zeros
            feats =  torch.nn.functional.pad(feats, (0, 1024 - feats.size(0)))
        return feats

    def print_head(self):
        colnames = self.df.columns.tolist()
        print("Clinical DataFrame Columns:")
        for col in colnames:
            print(f" - {col}")
            print(f"dtype: {self.df[col].dtype}")
        print(self.df.head())    

    def __len__(self):
        return len(self.pids)
    

    def __getitem__(self, idx):
        pid = self.pids[idx]
        
        clinical_data = self.df[self.df['id'] == int(pid)].drop(columns=['id']).values
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32).squeeze(0)
        pathology_data, _ = self._get_random_fixed_patches(pid)

        if self.radiology_dir is not None:
            if self.feature_extractor == "unet":
                radiology_data = self._get_unet_features(pid)
            elif self.feature_extractor == "resnet":
                radiology_data = self._get_resnet_embs(pid, mask=True)
            else:
                radiology_data = self._get_mri_vols(pid)
        else:
            radiology_data = None

        
        return {
            'pid': pid,
            'clinical_data': clinical_data,
            'pathology_data': pathology_data,
            'radiology_data': radiology_data
        }


class MMCLS_Dataset(MM_Dataset):
    def __init__(self, 
                 clinical_data_folder,
                 pathology_dir, 
                 radiology_dir,
                 wsi_crop_grid,
                 pids,
                 keyfile_path,
                 feature_extractor="none",
                 clinical_exclude_keys=["BCR", "BCR_PSA", "time_to_follow-up/BCR"],
                 mri_input_size=None,
                 ):
        super().__init__(clinical_data_folder=clinical_data_folder,
                         clinical_exclude_keys=clinical_exclude_keys,
                         pathology_dir=pathology_dir,
                         wsi_crop_grid=wsi_crop_grid,
                         radiology_dir=radiology_dir,
                         pids=pids,
                         mri_input_size=mri_input_size,
                         feature_extractor=feature_extractor)

        if pids is None:
            pids = [f.split(".json")[0] for f in os.listdir(clinical_data_folder) if f.endswith(".json")]
        self.pids = pids
        self.labels = self._get_labels(keyfile_path)

    def _get_labels(self, keyfile_path):
        # Load labels from a CSV file
        labels_df = pd.read_excel(keyfile_path, usecols=["miccai_id", "BCR", "BCR_PSA"])
        return labels_df.set_index('miccai_id').to_dict(orient='index')
    
    def _get_all_crops(self, pid):
        feature_paths = glob.glob(os.path.join(self.pathology_dir, "features", f"{pid}_*.pt"))

        if len(feature_paths) == 0:
            print(f"No pathology features found for pid {pid} in {self.pathology_dir}")
            return [torch.zeros((self.crop_grid * self.crop_grid, 1024))]

        #feature_path = random.choice(feature_paths)
        featureList = []
        for fp in feature_paths:
            idx = fp.split("_")[-1].split(".pt")[0]

            coord_path = os.path.join(self.pathology_dir, "coordinates", f"{pid}_{idx}.npy")

            features = torch.load(fp)  # Shape: (N, D)
            coordinates = np.load(coord_path)  # Shape: (N, 2)

            num_features = self.crop_grid * self.crop_grid

            assert features.shape[0] == coordinates.shape[0], f"Number of features ({features.shape}) and coordinates ({coordinates.shape}) must match"

            n_chunks = math.ceil(len(features)/num_features)
            chunks = [round(i * (len(features) - num_features) / (n_chunks - 1)) for i in range(n_chunks)]
            for i in chunks:
                featureList.append(features[i:i+num_features, :])

        return featureList


    def __getitem__(self, idx):
        pid = self.pids[idx]
        
        clinical_data = self.df[self.df['id'] == int(pid)].drop(columns=['id']).values
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32).squeeze(0)

        label = self.labels[int(pid)]["BCR"]
        label = torch.tensor(label, dtype=torch.float32).squeeze(0)

        #pathology_data, _ = self._get_random_fixed_patches(pid)

        pathology_data = self._get_all_crops(pid)

        if self.radiology_dir is not None:
            if self.feature_extractor == "unet":
                radiology_data = self._get_unet_features(pid)
            elif self.feature_extractor == "resnet":
                radiology_data = self._get_resnet_embs(pid, mask=True)
            else:
                radiology_data = self._get_mri_vols(pid)
        else:
            radiology_data = None

        data = (clinical_data, radiology_data, pathology_data)
        return data, label
    
class MMSurv_Dataset(MMCLS_Dataset):
    def __init__(self,
                 n_bins, 
                 clinical_data_folder,
                 pathology_dir, 
                 radiology_dir,
                 wsi_crop_grid,
                 pids,
                 keyfile_path,
                 feature_extractor="none",
                 mask_t2w_volume=True,  # This needs to be true if you want to use the mask to crop t2w
                 clinical_exclude_keys=["BCR", "BCR_PSA", "time_to_follow-up/BCR"],
                 mri_input_size=(25, 128, 120) # This forces the MRI volumes to this size
                 ):
        super().__init__(clinical_data_folder=clinical_data_folder,
                         clinical_exclude_keys=clinical_exclude_keys,
                         pathology_dir=pathology_dir,
                         wsi_crop_grid=wsi_crop_grid,
                         radiology_dir=radiology_dir,
                         keyfile_path=keyfile_path,
                         pids=pids,
                         feature_extractor=feature_extractor,
                         mri_input_size=mri_input_size)
        
        label_data = pd.read_excel(keyfile_path)
        label_data = label_data.set_index('miccai_id')
        label_data = label_data.dropna(subset=["time_to_follow-up/BCR"])
        uncensored_df = label_data[label_data["BCR"] == 0]
        
        eps = 1e-6
        disc_labels, q_bins = pd.qcut(uncensored_df["time_to_follow-up/BCR"], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = label_data["time_to_follow-up/BCR"].max() + eps
        q_bins[0] = label_data["time_to_follow-up/BCR"].min() - eps
        
        disc_labels, q_bins = pd.cut(label_data["time_to_follow-up/BCR"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)

        self.bins = q_bins
        self.labels = disc_labels
        self.label_data = label_data

        self.mask = mask_t2w_volume

    def toggle_mask(self, toggle: bool):
        self.mask = toggle

    def __getitem__(self, idx):
        pid = self.pids[idx]
        
        clinical_data = self.df[self.df['id'] == int(pid)].drop(columns=['id']).values
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32).squeeze(0)

        label = self.labels[int(pid)]
        label = torch.tensor(label, dtype=torch.int64).squeeze(0)

        event_time = self.label_data.loc[int(pid)]['time_to_follow-up/BCR']
        indicator = self.label_data.loc[int(pid)]['BCR']

        #pathology_data, _ = self._get_random_fixed_patches(pid)

        pathology_data = self._get_all_crops(pid)

        if self.radiology_dir is not None:
            if self.feature_extractor == "unet":
                radiology_data = self._get_unet_features(pid)
            elif self.feature_extractor == "resnet":
                radiology_data = self._get_resnet_embs(pid, mask=self.mask)
            else:
                radiology_data = self._get_mri_vols(pid)
        else:
            radiology_data = None

        data = (clinical_data, radiology_data, pathology_data)
        label_dict = {
            'pid': pid,
            'label': label,
            'event_time': event_time,
            'indicator': indicator
        }
        return data, label_dict


class GenericDataset(Dataset):
    def __init__(self):
        super().__init__()

    def toggle_augment(self, toggle):
        self.augment = toggle

    def dropout(self,x):
        """ Dropout augmentation for clinical features"""
        mask = (torch.rand_like(x) > self.clinical_dropout).float()
        return x*mask

    def get_num_clinical_vars(self):
        return len(self.clinical_vars)-1

    def __len__(self):
        return len(self.pids)
    
    def _set_survival_labels(self):
        """
        This method sets the survival analysis labels. It converts the survival times to discrete time-bins which will serve as the labels.
        It uses n quantiles, where n is controlled by the parameter self.n_bins

        Only need to run this once
        """
        label_df = self.key_df[self.key_df['anon_pid'].isin(self.pids)][["anon_pid", "BCR", "time_to_follow-up/BCR"]]
        label_df = label_df.set_index('anon_pid')
        label_df = label_df.dropna(subset=["time_to_follow-up/BCR"])
        uncensored_df = label_df[label_df["BCR"] == 0]
        
        eps = 1e-6
        disc_labels, q_bins = pd.qcut(uncensored_df["time_to_follow-up/BCR"], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = label_df["time_to_follow-up/BCR"].max() + eps
        q_bins[0] = label_df["time_to_follow-up/BCR"].min() - eps
        
        disc_labels, q_bins = pd.cut(label_df["time_to_follow-up/BCR"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)

        self.bins = q_bins
        self.labels = disc_labels
        self.label_df = label_df
    
    def _preprocess_clinical_data(self):
        from sklearn.preprocessing import LabelEncoder

        df = self.key_df[self.clinical_vars]

        # Set missing values ('x') to 2
        for col in ['positive_lymph_nodes', 'capsular_penetration', 'positive_surgical_margins', 'invasion_seminal_vesicles']:
            if col in df.columns:
                df[col] = df[col].replace({'x': 2}).fillna(2).astype(int)
            else:
                print(f"Warning: Column {col} not found in clinical data")

        # categorical keys to encode
        for col in ["pT_stage"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        if "BCR" in df.columns:
            df.loc[:, "BCR"] = np.array(df["BCR"]).astype(float).astype(int)
        if "BCR_PSA" in df.columns:
            df.loc[:, "BCR_PSA"] = pd.to_numeric(df["BCR_PSA"], errors='coerce').fillna(-1.0)
        if "tertiary_gleason" in df.columns:
            df.loc[:, "tertiary_gleason"] = pd.to_numeric(df["tertiary_gleason"], errors='coerce').fillna(-1)

        df_encoded = pd.get_dummies(df)

        self.clinical_data = df_encoded

    
    def select_split(self, split):
        split_ids = self.key_df[self.key_df["Split"] == split]["anon_pid"]
        self.pids = [int(p) for p in set(self.pids).intersection(split_ids)]
    
    def _crop_and_pad(self, mimg):
        while mimg.ndim < 4:
            mimg = mimg.unsqueeze(0)

        crop_pad = torchio.CropOrPad(self.mri_size)
        img = crop_pad(mimg)

        return img
    
    def _get_mri_vols_npy(self, pid, mask=True):
        # Loads all three MRI volumes for a given patient ID
        # Args:
        #   pid: patient ID
        #   mask: Boolean, mask t2w to make it the same size as the others
        # Returns:
        #   dict with keys 'adc', 'hbv', 't2w' and corresponding MRI volumes as torch tensors
        mri_paths = glob.glob(os.path.join(self.radiology_dir, f"{pid}_*.npy"))

        if len(mri_paths) < 4:
            raise ValueError(f"Not enough MRI volumes for patient {pid}")
        if len(mri_paths) > 4:
            print(f"Warning: More than 4 MRI files for patient {pid}, might cause errors")

        x_adc = np.load(next(f for f in mri_paths if f.endswith("adc.npy"))).astype(np.float32)
        x_hbv = np.load(next(f for f in mri_paths if f.endswith("hbv.npy"))).astype(np.float32)
        x_t2w = np.load(next(f for f in mri_paths if f.endswith("t2w.npy"))).astype(np.float32)

        if mask:
            x_mask = np.load(next(f for f in mri_paths if f.endswith("mask.npy"))).astype(np.float32)

            idxs = np.where(x_mask == 1)
        
            ymid = int(np.median(np.sort(idxs[1])))
            xmid = int(np.median(np.sort(idxs[2])))

            x_t2w = x_t2w[:, ymid-64:ymid+64, xmid-60:xmid+60]

        x_adc = torch.tensor(x_adc, dtype=torch.float32)
        x_hbv = torch.tensor(x_hbv, dtype=torch.float32)
        x_t2w = torch.tensor(x_t2w, dtype=torch.float32)

        if self.mri_size is not None:
            x_adc = self._crop_and_pad(x_adc)
            x_hbv = self._crop_and_pad(x_hbv)
            x_t2w = self._crop_and_pad(x_t2w)

        return {
            'adc': x_adc,
            'hbv': x_hbv,
            't2w': x_t2w
        }
    
    def _get_mri_vols_mha(self, pid, mask=True):
        # Loads all three MRI volumes for a given patient ID
        # Args:
        #   pid: patient ID
        #   mask: Boolean, mask t2w to make it the same size as the others
        # Returns:
        #   dict with keys 'adc', 'hbv', 't2w' and corresponding MRI volumes as torch tensors
        miccai_id = self.key_df.loc[self.key_df['anon_pid'] == pid, 'miccai_id'].iloc[0]
        folder = os.path.join(self.radiology_dir, str(miccai_id))

        # If folder does not exist, return no mri data? For now raise error
        if not os.path.exists(folder):
            raise ValueError(f"Radiology path: {folder} does not exist")    
            #return None

        mri_vols = os.listdir(folder)
        if len(mri_vols) < 4:
            raise ValueError(f"Not enough MRI volumes for patient {pid}")

        x_adc = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_adc.mha"))[0])
        x_adc = SimpleITK.GetArrayFromImage(x_adc).astype(np.float32)

        x_hbv = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_hbv.mha"))[0])
        x_hbv = SimpleITK.GetArrayFromImage(x_hbv).astype(np.float32)   

        x_t2w = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_t2w.mha"))[0])
        x_t2w = SimpleITK.GetArrayFromImage(x_t2w).astype(np.float32)

        if mask:
            x_mask = SimpleITK.ReadImage(glob.glob(os.path.join(folder, "*_mask.mha"))[0])
            x_mask = SimpleITK.GetArrayFromImage(x_mask)

            idxs = np.where(x_mask == 1)
        
            ymid = int(np.median(np.sort(idxs[1])))
            xmid = int(np.median(np.sort(idxs[2])))

            x_t2w = x_t2w[:, ymid-64:ymid+64, xmid-60:xmid+60]

        x_adc = torch.tensor(x_adc, dtype=torch.float32)
        x_hbv = torch.tensor(x_hbv, dtype=torch.float32)
        x_t2w = torch.tensor(x_t2w, dtype=torch.float32)

        if self.mri_size is not None:
            x_adc = self._crop_and_pad(x_adc)
            x_hbv = self._crop_and_pad(x_hbv)
            x_t2w = self._crop_and_pad(x_t2w)

        return {
            'adc': x_adc,
            'hbv': x_hbv,
            't2w': x_t2w
        }
    
    def _get_mri_vols(self, pid, mask=True, extension='npy'):
        if extension == 'npy':
            return self._get_mri_vols_npy(pid=pid, mask=mask)
        elif extension == 'mha':
            return self._get_mri_vols_mha(pid=pid, mask=mask)
        else:
            raise NotImplementedError(f"Extension {extension} not available")
    
    
    def __getitem__(self, index):
        pass

    

class CWZDataset(GenericDataset):
    def __init__(self, 
                 keyfile, 
                 main_dir,
                 n_bins,
                 mri_input_size=None,
                 wsi_fm_model="prism",
                 clinical_vars=None,
                 augment=True):
        
        super().__init__()
        
        self.key_df = pd.read_excel(keyfile)
        self.key_df = self.key_df[self.key_df["Exclude"] != 1]
        #self.key_df = self.key_df[self.key_df["Split"] == split]
        self.main_dir = main_dir
        self.pathology_dir = os.path.join(main_dir, "wsi_embeddings", wsi_fm_model)
        self.radiology_dir = os.path.join(main_dir, "images")
        self.mri_size = mri_input_size
        self.n_bins = n_bins

        main_df = pd.read_excel(os.path.join(self.main_dir, "cohort_overview.xlsx"))
        #self.pids = list(set(self.key_df['anon_pid']).intersection(main_df['patient_id']))
        self.pids = [int(p) for p in set(self.key_df['anon_pid']).intersection(main_df['patient_id'])]


        if clinical_vars is None:
            self.clinical_vars = ['anon_pid','ISUP', 'pre_operative_PSA', 'pT_stage', 'positive_lymph_nodes', 
                                  'capsular_penetration', 'positive_surgical_margins', 'invasion_seminal_vesicles', 
                                  'yT?', 'lymphovascular_invasion']
        else:
            self.clinical_vars = clinical_vars
        
        self._preprocess_clinical_data() 
        self._set_survival_labels()

        self.mri_transform = TransformMRI(apply_query_mask=True, modality_keys=['adc', 'hbv', 't2w'],
                                          num_ghosts=2, flip_probability=0.5, two_views=False)
        self.scale_mri = torchio.RescaleIntensity((0, 1), include=['adc', 'hbv', 't2w'])
        self.frofa = FroFA(channel_wise=True)
        self.augment = augment

    
    def __getitem__(self, index):
        pid = self.pids[index]

        cvec = self.clinical_data[self.clinical_data['anon_pid'] == int(pid)].drop(columns=['anon_pid']).values
        cvec = torch.tensor(cvec, dtype=torch.float32).squeeze(0)

        wsi_emb = np.load(os.path.join(self.pathology_dir, f"{pid}.npy"))
        wsi_emb = torch.tensor(wsi_emb)

        mri_dict = self._get_mri_vols(pid, mask=True)

        label = self.labels[int(pid)]
        label = torch.tensor(label, dtype=torch.int64).squeeze(0)

        event_time = float(self.label_df.loc[int(pid)]['time_to_follow-up/BCR'])
        indicator = int(self.label_df.loc[int(pid)]['BCR'])

        if self.augment:
            mri_dict = self.mri_transform(mri_dict)
            #wsi_emb = self.frofa(wsi_emb)
            #wsi_emb = wsi_emb.squeeze()
        else:
            mri_dict = self.scale_mri(mri_dict)

        data = (cvec, mri_dict, wsi_emb)
        label_dict = {
            'pid': pid,
            'label': label,
            'event_time': event_time,
            'indicator': indicator
        }
        return data, label_dict


class PairedDataset(GenericDataset):
    def __init__(self, 
                 keyfile, 
                 pathology_dir,
                 radiology_dir,
                 n_bins,
                 use_miccai=False,
                 mri_input_size=None,
                 clinical_vars=None,
                 split='train',
                 augment=True):
        
        self.key_df = pd.read_excel(keyfile)
        self.key_df = self.key_df.loc[self.key_df["Exclude"] != 1]
        self.pids = self.key_df.loc[self.key_df['Split'] == split, 'anon_pid'].astype(int).tolist()
        self.pathology_dir = pathology_dir
        self.radiology_dir = radiology_dir

        self.wsi_pids = [int(os.path.splitext(imf)[0]) for imf in os.listdir(self.pathology_dir)]

        self.use_miccai = use_miccai
        if self.use_miccai:
            mri_miccai_pids = [int(f) for f in os.listdir(self.radiology_dir) if f.isdigit()]
            self.mri_pids = self.key_df.loc[self.key_df["miccai_id"].isin(mri_miccai_pids), "anon_pid"]
            self.mri_ext = "mha"
        else:
            mri_pids = [s.split("_", 1)[0] for s in os.listdir(self.radiology_dir)]
            self.mri_pids = list(set(mri_pids))
            self.mri_ext = "npy"
        self.mri_pids = [int(p) for p in self.mri_pids]

        # Filter any patients that does not have any mri or wsi embeddings
        self.pids = [p for p in self.pids if p in self.wsi_pids or p in self.mri_pids]

        self.mri_size = mri_input_size
        self.n_bins = n_bins

        if clinical_vars is None:
            self.clinical_vars = ['anon_pid','ISUP', 'pre_operative_PSA', 'pT_stage', 'positive_lymph_nodes', 
                                  'capsular_penetration', 'positive_surgical_margins', 'invasion_seminal_vesicles', 
                                  'yT?', 'lymphovascular_invasion']
        else:
            self.clinical_vars = clinical_vars
        
        self._preprocess_clinical_data() 
        self._set_survival_labels()

        self.mri_transform = TransformMRI(apply_query_mask=False, modality_keys=['adc', 'hbv', 't2w'],
                                          num_ghosts=2, flip_probability=0.5, two_views=True)
        self.scale_mri = torchio.RescaleIntensity((0, 1), include=['adc', 'hbv', 't2w'])
        self.augment = augment

    def random_modality_drop(self, data, n_drops):
        # Only drop data modalities if they are not already missing
        while data.count(None) < n_drops:
            rnd_idx = np.random.randint(0, len(data))
            data[rnd_idx] = None

        return data
            

    def __getitem__(self, index):
        pid = self.pids[index]

        cvec = self.clinical_data[self.clinical_data['anon_pid'] == int(pid)].drop(columns=['anon_pid']).values
        cvec = torch.tensor(cvec, dtype=torch.float32).squeeze(0)

        if pid in self.wsi_pids:
            wsi_emb = np.load(os.path.join(self.pathology_dir, f"{pid}.npy"))
            wsi_emb = torch.tensor(wsi_emb)
        else:
            wsi_emb = None

        if pid in self.mri_pids:
            mri_dict = self._get_mri_vols(pid, mask=True, extension=self.mri_ext)
            if self.augment:
                mri_dict, mri_dict_ = self.mri_transform(mri_dict)
            else:
                mri_dict = self.scale_mri(mri_dict)
                mri_dict_ = deepcopy(mri_dict)
        else:
            mri_dict, mri_dict_ = None, None

        label = self.labels[int(pid)]
        label = torch.tensor(label, dtype=torch.int64).squeeze(0)

        event_time = float(self.label_df.loc[int(pid)]['time_to_follow-up/BCR'])
        indicator = int(self.label_df.loc[int(pid)]['BCR'])

        data = (cvec, mri_dict, wsi_emb)
        data = self.random_modality_drop(data=data, n_drops=1)
        data_ = (cvec, mri_dict_, wsi_emb)
        data_ = self.random_modality_drop(data=data_, n_drops=2)

        label_dict = {
            'pid': pid,
            'label': label,
            'event_time': event_time,
            'indicator': indicator
        }
        return data, data_, label_dict



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    parser = argparse.ArgumentParser(description='Test Models Main Script')
    parser.add_argument('-c', '--cluster', action='store_true', help='Flag for being on the cluster')
    args = parser.parse_args()
    if args.cluster:
        # SOL paths
        print("Cluster")
        main_dir = "/data/pa_cpgarchive/projects/chimera/_gc/task1/val/"
        keyfile = "/data/pa_cpgarchive//projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"
    else:
        # local mac paths
        print("Local")
        main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/"
        keyfile = "/Volumes/PA_CPGARCHIVE/projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    
    clinical_folder = os.path.join(main_dir, "clinical_data")
    pids = [f.split(".json")[0] for f in os.listdir(clinical_folder) if f.endswith(".json")]


    dataset = MMSurv_Dataset(
        n_bins=10,
        clinical_data_folder=clinical_folder,
        pathology_dir=os.path.join(main_dir, "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(main_dir, "radiology"),
        pids=pids,
        feature_extractor="resnet",
        mask_t2w_volume=True,
        mri_input_size=(20, 128, 120),
        keyfile_path=keyfile
    )   

    loader = DataLoader(dataset, batch_size=2)
    for data, label_dict in loader:
        print(f"Clinical Data: {data[0]}")
        print(f"Label: {label_dict['label']}")
        print(f"Time: {label_dict['event_time']} months")
        print(f"BCR?: {label_dict['indicator']}")

        print(f"MRI shapes: t2w: {data[1]['t2w'].shape}, adc: {data[1]['adc'].shape}")
    

        


        





