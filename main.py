from datasets import MMCLR_Dataset, MMCLS_Dataset
from utils import SimpleTrainer
from models import MMCLRModel, MMCLSModel

import torch
import os
import argparse
import yaml

from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description='MMCLR Main Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
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

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment.seed)

    # Initialize datasets
    tr_dataset = MMCLS_Dataset(
        clinical_data_folder=config.data.training.clinical_data_folder,
        pathology_dir=os.path.join(config.data.training.main_dir, "pathology/features"),
        wsi_crop_size=config.data.training.wsi_crop_size,
        radiology_dir=os.path.join(config.data.training.main_dir, "radiology/features"),
        pids=None,
        use_unet_features=True,
        clinical_exclude_keys=["BCR_PSA"],
        clinical_label_key="BCR"
    )

    val_dataset = MMCLS_Dataset(
        clinical_data_folder=config.data.validation.clinical_path,
        pathology_dir=os.path.join(config.data.validation.main_dir, "pathology/features"),
        wsi_crop_size=config.data.validation.wsi_crop_size,
        radiology_dir=os.path.join(config.data.validation.main_dir, "radiology/features"),
        pids=None,
        use_unet_features=True,
        clinical_exclude_keys=["BCR_PSA"],
        clinical_label_key="BCR"
    )

    # Initialize model
    if config.model.type == "MMCLS":
        model = MMCLSModel(
            clinical_input_dim=config.model.clinical_input_dim,
            clinical_hidden_dim=config.model.clinical_hidden_dim,
            clinical_output_dim=config.model.perceiver_input_dim,
            perceiver_input_dim=config.model.perceiver_input_dim,
            perceiver_output_dim=config.model.perceiver_output_dim,
            device=device
        )
    else:
        raise NotImplementedError(f"Model type {config.model.type} not implemented.")

    # Initialize trainer
    trainer = SimpleTrainer(
        model=model,
        tr_dataset=tr_dataset,
        val_dataset=val_dataset,
        max_epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        optimizer_name=config.training.optimizer.name,
        device=device,
        save_dir=None,
        debug_mode=False,
        model_name=config.experiment.name
    )

    # Start training
    trainer.train()