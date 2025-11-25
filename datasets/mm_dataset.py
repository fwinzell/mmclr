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

import glob

from torch.utils.data import Dataset
#from monai.transforms import CenterSpatialCrop, SpatialPad
import torch.nn.functional as F
import torchio


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
        coordinates = np.load(coord_path) #.astype(np.int16)  # Shape: (N, 2)

        num_patches_needed = self.crop_grid * self.crop_grid

        assert features.shape[0] == coordinates.shape[0], f"Number of features ({features.shape}) and coordinates ({coordinates.shape}) must match"
        
        idx = np.random.randint(0, features.shape[0]-num_patches_needed)
        
        features = features[idx:idx+num_patches_needed].float()
        coordinates = coordinates[idx:idx+num_patches_needed]

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
    

        


        





