import json
import os
import pandas as pd
import pyvips
import SimpleITK
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch

from models import CMLP
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import resnet
from datasets import CLRDataset
from main_moco import load_config

class ClinicalDataset(Dataset):
    def __init__(self, 
                 clinical_data_folder,
                 exclude_keys=None):
        self.clinical_data_folder = clinical_data_folder
        self.data_files = [f for f in os.listdir(clinical_data_folder) if f.endswith('.json')]
        self.df = self._get_data_frame(exclude_keys)

        self.num_attributes = self.df.shape[1]
        
    def _get_data_frame(self, exclude_keys):
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

    def print_head(self):
        print(self.df.head())


    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        features = self.df.iloc[idx]

        return torch.tensor(features, dtype=torch.float32)


def load_clinical_data(folder=None):
    if folder is None:
        folder = "/Users/filipwinzell/CHIMERA/clinical_data/"
    df = pd.DataFrame()

    for f in os.listdir(folder):
        if f.endswith(".json"):
            with open(os.path.join(folder, f), 'r') as file:
                data = json.load(file)
                id = f.split(".json")[0]
                data['id'] = id
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    return df


def load_pathology():
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/pathology/features"
    fdir = os.path.join(main_dir, "features")
    cdir = os.path.join(main_dir, "coordinates")
    features = [f for f in os.listdir(fdir) if f.endswith(".pt")]
    
    coords = [os.path.splitext(f)[0]+".npy" for f in features]

    fvec = torch.load(os.path.join(fdir, features[0]))
    cvec = np.load(os.path.join(cdir, coords[0]))

    print(f"Feature vector shape: {fvec.shape}")
    print(f"Coordinate vector shape: {cvec.shape}")


def load_and_display_wsi(pid, max_size=1000):
    img_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/pathology/images"
    img_path = os.path.join(img_dir, pid, f"{pid}_1.tif")

    cdir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/pathology/features/coordinates"
    coords = np.load(os.path.join(cdir, f"{pid}_1.npy"))
    x_coords = np.array(coords.tolist())[:,0]
    y_coords = np.array(coords.tolist())[:,1]
    tile_size = coords[0][2]
    
    # Use PyVips for pathology images (.tif, .tiff, .mrxs, .svs, .ndpi)
    print(f"Loading pathology image using PyVips: {img_path}")
    image = pyvips.Image.new_from_file(img_path)


    scale_factor = min(max_size / image.width, max_size / image.height)
    if scale_factor < 1.0:
        print(f"Downsampling image by factor {scale_factor:.3f} (from {image.width}x{image.height} to {int(image.width*scale_factor)}x{int(image.height*scale_factor)})")
        image = image.resize(scale_factor)
        x_coords = (x_coords * scale_factor).astype(int)
        y_coords = (y_coords * scale_factor).astype(int)
        tile_size = int(tile_size * scale_factor)
    else:
        print(f"Image size {image.width}x{image.height} is within max_size={max_size}, no downsampling needed")

    memory_image = image.write_to_memory()
    array = np.frombuffer(memory_image, dtype=np.uint8)
        
    # Reshape based on image dimensions and bands
    height = image.height
    width = image.width
    bands = image.bands
        
    if bands == 1:
        # Grayscale image
        array = array.reshape((height, width))
    else:
        # Multi-channel image (RGB, RGBA, etc.)
        array = array.reshape((height, width, bands))

    cv2.rectangle(array, (x_coords[0], y_coords[0]), (x_coords[0]+tile_size, y_coords[0]+tile_size), (255, 0, 0), 2)

    cv2.imshow("WSI Image with Tile", array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        


def load_radiology(i):
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
    folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
    mri_paths = [f for f in os.listdir(os.path.join(main_dir, folders[i])) if f.endswith(".mha")]
    
    for vol_path in mri_paths:
        _path = os.path.join(main_dir, folders[i], vol_path)

        print(f"Loading radiology image using SimpleITK: {vol_path}")
        image = SimpleITK.ReadImage(_path)
        array = SimpleITK.GetArrayFromImage(image)  

        print(f"Image shape (slices, height, width): {array.shape}")

    slices = array.shape[0]
    for s in range(slices):
        img = array[s, :, :]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(f"Slice {s+1}/{slices}", img)
        cv2.waitKey(500)  # Wait for a key press to show the next


def load_mri_mask(i):
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
    folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
    mri_paths = [f for f in os.listdir(os.path.join(main_dir, folders[i])) if f.endswith(".mha")]

    vol_path = [f for f in mri_paths if "t2w" in f.lower()][0]
    mask_path = vol_path.replace("t2w.mha", "mask.mha")
    
    vol_path = os.path.join(main_dir, folders[i], vol_path)
    mask_path = os.path.join(main_dir, folders[i], mask_path)

    print(f"Loading radiology image using SimpleITK: {vol_path}")
    image = SimpleITK.ReadImage(vol_path)
    array = SimpleITK.GetArrayFromImage(image)  

    mask = SimpleITK.ReadImage(mask_path)
    mask_array = SimpleITK.GetArrayFromImage(mask)

    slices = array.shape[0]
    for s in range(slices):
        img = array[s, :, :]
        msk = mask_array[s, :, :]
        img = img*msk  # Apply mask
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #msk = cv2.normalize(msk, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(f"Slice", img)
        #cv2.imshow(f"Mask", msk)
        cv2.waitKey(500)  # Wait for a key press to show the next


def load_tw2_with_mask(i):
    main_dir = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/radiology/images"
    folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
    mri_paths = [f for f in os.listdir(os.path.join(main_dir, folders[i])) if f.endswith(".mha")]

    vol_path = [f for f in mri_paths if "t2w" in f.lower()][0]
    mask_path = vol_path.replace("t2w.mha", "mask.mha")
    
    vol_path = os.path.join(main_dir, folders[i], vol_path)
    mask_path = os.path.join(main_dir, folders[i], mask_path)

    print(f"Loading radiology image using SimpleITK: {vol_path}")
    image = SimpleITK.ReadImage(vol_path)
    array = SimpleITK.GetArrayFromImage(image)  

    mask = SimpleITK.ReadImage(mask_path)
    mask_array = SimpleITK.GetArrayFromImage(mask)

    idxs = np.where(mask_array == 1)
    #zmin, zmax = np.min(idxs[0]), np.max(idxs[0])
    ymid = int(np.median(np.sort(idxs[1])))
    xmid = int(np.median(np.sort(idxs[2])))

    cropped_array = array[:, ymid-64:ymid+64, xmid-60:xmid+60]
    return cropped_array.astype(np.float32) / np.max(cropped_array)
    
    #print(f"Cropped shape: {cropped_array.shape}")

    #slices = cropped_array.shape[0]
    #for s in range(slices):
    #    img = cropped_array[s, :, :]
    #    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #    cv2.imshow(f"Slice", img)
   #cv2.waitKey(500)  # Wait for a key press to show the next

def load_keyfile(path):
    keydf = pd.read_excel(path)
    return keydf


def gen_mri_embeddings(i):
    t2w_arr = load_tw2_with_mask(i)
    model = resnet.resnet18(pretrained=True, spatial_dims=3, n_input_channels=1, feed_forward=False,
                            shortcut_type='A')

    x = torch.tensor(t2w_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    print(f"Input tensor shape: {x.shape}")
    with torch.no_grad():
        embeddings = model(x)  # Shape: (slices, output_dim)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def test_clr_dataset(config_path):
    config = load_config(config_path=config_path)
    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"
    patient_info = pd.read_excel(keyfile, usecols=["miccai_id", "Split", "BCR", "BCR_PSA"])
    train_ids = patient_info.loc[patient_info["Split"] == "train", "miccai_id"].tolist()

    print(f"Number of patients: {len(train_ids)}")

    dataset = CLRDataset(
        clinical_data_folder=config.data["clinical_path"],
        pathology_dir=os.path.join(config.data["main_dir"], "pathology/features"),
        radiology_dir=os.path.join(config.data["main_dir"], "radiology/images"),
        wsi_crop_grid=1,
        clinical_dropout=config.training["clinical_dropout"],
        mri_dropout=config.training["mri_dropout"],
        feature_extractor="resnet",
        mri_input_size=(25, 128, 120),
        pids=train_ids,
        use_frofa=True,
        mri_include=['t2w', 'adc', 'hbv'],
        verbose=True
    )
    loader = DataLoader(dataset=dataset, batch_size=1)

    for query, key in loader:
        radiology_q = query[1]

        slices = 25
        for s in range(slices):
            img = np.array(radiology_q['t2w'].squeeze()[s,:,:])
            print(img.shape)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow(f"Slice {s+1}/{slices}", img)
            cv2.waitKey(500)  # Wait for a key press to show the next
            cv2.destroyAllWindows()
        
        
            



if __name__ == "__main__":
    #load_and_display_wsi('1021', max_size=2000)


    #for j in range(5): load_radiology(j)
    #load_pathology()

    #load_tw2_with_mask(0)
    #emb = gen_mri_embeddings(0)

    config_path = "/Users/filipwinzell/Python/Radboud/MMCLR/configs/moco_config_local.yaml"
    test_clr_dataset(config_path=config_path)

    """
    keydf = load_keyfile("/Volumes/PA_CPGARCHIVE/projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx")

    local_path = "/Volumes/PA_CPGARCHIVE/projects/chimera/_aws/task1/clinical_data/"
    sol_path = "/data/pa_cpgarchive/projects/chimera/_gc/task1/val/clinical_data/"
    clidat = ClinicalDataset(clinical_data_folder=local_path,
                             exclude_keys=[])
    clidat.print_head()

    dataloader = DataLoader(clidat, batch_size=1, shuffle=True)

    model = CMLP(input_dim=clidat.num_attributes, hidden_dim=512, output_dim=1024)

    for batch in dataloader:
        print(batch.shape)
        outputs = model(batch)
        print(outputs.shape)
        break
    """
    
