from datasets import CLRDataset
from models import MMViTModel, MoCoWrapper
from utils import MoCoTrainer

import torch
import os
import argparse
import yaml
import pandas as pd
import datetime

from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description='MMCLR Main Training Script')
    parser.add_argument('--config', type=str, help='Path to the config file', default='MMCLR/configs/moco_config_local.yaml')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return SimpleNamespace(**config)

def main():
    args = parse_args()
    config = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if config.model["base_model_type"].lower() == "vit":
        base_model = MMViTModel
    else:
        raise NotImplementedError("Base model type not implemented")
    
    moco_model = MoCoWrapper(base_encoder=base_model,
                             encoder_args=config.model,
                             K=config.moco["K"],
                             dim=config.moco["dim"],
                             m=config.moco["m"],
                             T=config.moco["T"],
                             device=device).to(device)

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"
    patient_info = pd.read_excel(keyfile, usecols=["miccai_id", "Split", "BCR", "BCR_PSA"])

    train_ids = patient_info.loc[patient_info["Split"] == "train", "miccai_id"].tolist()


    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])


    dataset = CLRDataset(
        clinical_data_folder=config.data["clinical_path"],
        pathology_dir=os.path.join(config.data["main_dir"], "pathology/features"),
        radiology_dir=os.path.join(config.data["main_dir"], "radiology/images"),
        wsi_crop_grid=config.training["wsi_grid_size"],
        clinical_dropout=config.training["clinical_dropout"],
        mri_dropout=config.training["mri_dropout"],
        feature_extractor="resnet",
        mri_input_size=(25, 128, 120),
        pids=train_ids,
        use_frofa=True,
        mri_include=['t2w', 'adc', 'hbv']
    )

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    trainer = MoCoTrainer(
        dataset=dataset,
        model=moco_model,
        max_epochs=config.training["epochs"],
        batch_size=config.training["batch_size"],
        virtual_batch_size=config.training["virtual_batch_size"],
        learning_rate=config.training["learning_rate"],
        optimizer_name=config.training["optimizer"],
        optimizer_hparams=config.training["opt_hparams"],
        device=device,
        model_name=f"{config.experiment['name']}_{date}",
        save_dir=config.experiment["save_dir"]
    )

    trainer.train()

if __name__ == "__main__":
    main()


