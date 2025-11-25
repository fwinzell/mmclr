from mm_dataset import MMCLS_Dataset, MMCLR_Dataset
import os
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    mounted_dir = "/data/pa_cpgarchive/projects/chimera/"
    keyfile = f"{mounted_dir}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"
    main_dir = f"{mounted_dir}/_gc/task1/"
    clincal_folder = f"{main_dir}/clinical_data/"

    dataset = MMCLS_Dataset(
        clinical_data_folder=clincal_folder,
        pathology_dir=os.path.join(main_dir, "pathology/features"),
        wsi_crop_size=20000,
        radiology_dir=os.path.join(main_dir, "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, (data, label) in enumerate(dataloader):
        clinical_data, radiology_data, pathology_data = data
        print(f"Batch {i}:")
        print(f"  Clinical data shape: {clinical_data.shape}")
        if radiology_data is not None:
            print(f"  Radiology data shape: {radiology_data.shape}")
        else:
            print("  Radiology data: None")
        print(f"  Pathology data shape: {pathology_data.shape}")
        print(f"  Label: {label.item()}")




