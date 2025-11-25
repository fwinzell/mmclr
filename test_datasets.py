from datasets import CLRDataset, MM_Dataset, MMSurv_Dataset

from torch.utils.data import DataLoader
import argparse

import os
import pandas as pd


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
    mri_input_size=(20, 128, 120)
)   

loader = DataLoader(dataset, batch_size=2)
for query, key in loader:
    print(f"MRI shapes: t2w: {query[1]['t2w'].shape}, adc: {query[1]['adc'].shape}")
    print(f"MRI shapes: t2w: {key[1]['t2w'].shape}, adc: {key[1]['adc'].shape}")
