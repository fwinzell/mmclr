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

from torch.utils.data import Dataset



class MMCLR_Dataset(Dataset):
    def __init__(self, 
                 clinical_data_folder,
                 clinical_exclude_keys, 
                 pathology_dir,
                 wsi_crop_size, 
                 radiology_dir,
                 pids,
                 use_unet_features=False):
        
        self.clinical_data_folder = clinical_data_folder
        self.pathology_dir = pathology_dir
        self.radiology_dir = radiology_dir
        self.pids = pids
        self.use_unet = use_unet_features

        self.data_files = [f for f in os.listdir(clinical_data_folder) if f.endswith('.json')]

        self.df = self._get_data_frame(clinical_exclude_keys)
        self.print_head()

        self._ensure_numeric()

        self.num_attributes = self.df.shape[1]

        self.wsi_crop_size = wsi_crop_size 
        
    
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
            df[col] = df[col].replace({'x': 2}).astype(int)

        # categorical keys to encode
        for col in ["pT_stage", "earlier_therapy"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        df["BCR"] = np.array(df["BCR"]).astype(float).astype(int)
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
                print(f"Converting column {col} to numeric")
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
    
    def _get_random_crop(self, pid):
        # For pathology data, get a random crop of features for the given patient ID
        # Args:
        #   pid: patient ID
        # Returns:
        #   features: torch tensor of shape (N, D) where N is number of patches in the crop and D is feature dimension
        feature_paths = glob.glob(os.path.join(self.pathology_dir, "features", f"{pid}_*.pt"))
        coord_paths = glob.glob(os.path.join(self.pathology_dir, "coordinates", f"{pid}_*.npy"))

        idx = np.random.randint(0, len(feature_paths))
        feature_path = feature_paths[idx]
        coord_path = coord_paths[idx]

        features = torch.load(feature_path)  # Shape: (N, D)
        coordinates = np.load(coord_path)  # Shape: (N, 2)

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
        
        features = features[crop_indices]
        coordinates = coordinates[crop_indices]

        return features, coordinates
    
    def _get_mri_vols(self, pid):
        # Loads all three MRI volumes for a given patient ID
        # Args:
        #   pid: patient ID
        # Returns:
        #   dict with keys 'adc', 'hbv', 't2w' and corresponding MRI volumes as torch tensors
        folder = os.path.join(self.radiology_dir, pid)

        # If folder does not exist, return no mri data
        if not os.path.exists(folder):
            return None

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
    
    def _get_unet_features(self, pid, pad=True):
         # Loads features from nnUnet of MRI data
        # Args:
        #   pid: patient ID
        # Returns:
        #   features as torch tensors
        file = os.path.join(self.radiology_dir, f"{pid}_0001_features.pt")

        # If folder does not exist, return no mri data
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found")
        
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
        pathology_data, _ = self._get_random_crop(pid)

        if self.radiology_dir is not None:
            if self.use_unet:
                radiology_data = self._get_unet_features(pid)
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


class MMCLS_Dataset(MMCLR_Dataset):
    def __init__(self, 
                 clinical_data_folder,
                 pathology_dir,
                 wsi_crop_size, 
                 radiology_dir,
                 pids,
                 use_unet_features=True,
                 clinical_exclude_keys=["BCR_PSA"],
                 clinical_label_key="BCR"):
        super().__init__(clinical_data_folder,
                         clinical_exclude_keys,
                         pathology_dir,
                         wsi_crop_size,
                         radiology_dir,
                         pids,
                         use_unet_features=use_unet_features)
        self.clinical_label_key = clinical_label_key
        if pids is None:
            pids = [f.split(".json")[0] for f in os.listdir(clinical_data_folder) if f.endswith(".json")]
        self.pids = pids

    def __getitem__(self, idx):
        pid = self.pids[idx]
        
        clinical_data = self.df[self.df['id'] == int(pid)].drop(columns=['id', self.clinical_label_key]).values
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32).squeeze(0)

        label = self.df[self.df['id'] == int(pid)][self.clinical_label_key].values
        label = torch.tensor(label, dtype=torch.float32).squeeze(0)

        pathology_data, _ = self._get_random_crop(pid)

        if self.radiology_dir is not None:
            if self.use_unet:
                radiology_data = self._get_unet_features(pid)
            else:
                radiology_data = self._get_mri_vols(pid)
        else:
            radiology_data = None

        data = (clinical_data, pathology_data, radiology_data)
        return data, label

if __name__ == "__main__":
    # local mac path
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/"
    # SOL path
    #main_dir = "/data/pa_cpgarchive/projects/chimera/_aws/task1/"
    
    clinical_folder = os.path.join(main_dir, "clinical_data")
    pids = [f.split(".json")[0] for f in os.listdir(clinical_folder) if f.endswith(".json")]

    """ CLR test
    dataset = MMCLR_Dataset(
        clinical_data_folder=clinical_folder,
        clinical_exclude_keys=["BCR", "BCR_PSA"],
        pathology_dir=os.path.join(main_dir, "pathology/features"),
        wsi_crop_size=5000,
        radiology_dir=os.path.join(main_dir, "radiology/features"),
        pids=pids,
        use_unet_features=True
    )

    sample = dataset[0]
    clinical = sample['clinical_data']
    pathology = sample['pathology_data']
    radiology = sample['radiology_data']
    print(f"PID: {sample['pid']}")
    print(f"Clinical Data: {clinical}")
    print(f"Pathology Data Shape: {pathology.shape}")
    print(f"Unet features shape: {radiology.shape}")
    #print(f"Radiology Data Keys: {list(sample['radiology_data'].keys())}")
    #print(f"Radiology ADC Shape: {sample['radiology_data']['adc'].shape}")
    """

    dataset = MMCLS_Dataset(
        clinical_data_folder=clinical_folder,
        pathology_dir=os.path.join(main_dir, "pathology/features"),
        wsi_crop_size=5000,
        radiology_dir=os.path.join(main_dir, "radiology/features"),
        pids=pids,
        use_unet_features=True,
        clinical_exclude_keys=["BCR_PSA"],
        clinical_label_key="BCR"
    )   

    data, label = dataset[0]

    print(f"Clinical Data: {data[0]}")
    print(f"Label: {label}")
    

        


        





