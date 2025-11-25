from datasets import MMCLR_Dataset, MMCLS_Dataset
from utils import SimpleTrainer
from models import MMCLRModel, MMCLSModel, MMViTModel

import torch
import os
import argparse
import yaml

from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description='MMCLR Main Training Script')
    parser.add_argument('--config', type=str, help='Path to the config file', default='MMCLR/configs/simple_config_local.yaml')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return SimpleNamespace(**config)

def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])

    # Initialize datasets
    tr_dataset = MMCLS_Dataset(
        clinical_data_folder=config.data["training"]["clinical_path"],
        pathology_dir=os.path.join(config.data["training"]["main_dir"], "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.data["training"]["main_dir"], "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
    )

    val_dataset = MMCLS_Dataset(
        clinical_data_folder=config.data["validation"]["clinical_path"],
        pathology_dir=os.path.join(config.data["validation"]["main_dir"], "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.data["validation"]["main_dir"], "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
    )

    # Initialize model
    if config.model["type"] == "Perceiver":
        base_model = MMCLRModel(
            clinical_input_dim=config.model["clinical_input_dim"],
            clinical_hidden_dim=config.model["clinical_hidden_dim"],
            clinical_output_dim=config.model["clinical_output_dim"],
            device=device
        ).to(device)
    elif config.model["type"] == "ViT":
        base_model = MMViTModel(
            clinical_input_dim=config.model["clinical_input_dim"],
            clinical_hidden_dim=config.model["clinical_hidden_dim"],
            feature_dim=config.model["feature_dim"],
            num_features=config.model["vit_num_features"],
            num_modalities=config.model["num_modalities"],
            device=device
        ).to(device)
    else:
        raise NotImplementedError(f"Model type {config.model['type']} not implemented.")
    
    model = MMCLSModel(
        base_model=base_model,
        cls_input_dim=config.model["clinical_output_dim"],
        threshold=0.5
    ).to(device)

    # Initialize trainer
    trainer = SimpleTrainer(
        model=model,
        tr_dataset=tr_dataset,
        val_dataset=val_dataset,
        max_epochs=config.training["epochs"],
        batch_size=config.training["batch_size"],
        learning_rate=config.training["learning_rate"],
        optimizer_name=config.training["optimizer"],
        device=device,
        save_dir=config.experiment["save_dir"],
        debug_mode=False,
        model_name=config.experiment["name"]
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()