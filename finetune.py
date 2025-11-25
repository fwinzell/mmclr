from datasets import MMCLS_Dataset, MMSurv_Dataset
from utils import SurvTrainer, NLLSurvLoss, CrossEntropySurvLoss
from models import MMViTModel, MoCoWrapper, ClassifierHead, ADMIL_Model

import torch
import argparse
import yaml

import datetime

from types import SimpleNamespace

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description='MMCLR Main Training Script')
    parser.add_argument('--moco_config', type=str, help='Path to the moco config file', default='configs/moco_config.yaml')
    parser.add_argument('--config', type=str, help='Path to config file', default='configs/surv_config.yaml')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)

    return config

def load_pretrained(config, model_path, device):
    if config.model["base_model_type"].lower() == "vit":
        base_model = MMViTModel
    else:
        raise NotImplementedError("Base model type not implemented")
    
    moco_model = MoCoWrapper(base_encoder=base_model,
                             encoder_args=config.model,
                             device=device).to(device)
    
    moco_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=True)
    return moco_model.encoder_q

def load_encoder(config, model_path, device):
    if config.model["base_model_type"].lower() == "vit":
        base_model = MMViTModel(**config.model, device=device)

    base_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=True)
    return base_model

def get_cls_datasets(config, keyfile):
    # Initialize datasets
    tr_dataset = MMCLS_Dataset(
        clinical_data_folder=config.data["clinical_path"],
        pathology_dir=os.path.join(config.data["main_dir"], "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.data["main_dir"], "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
    )

    val_dataset = MMCLS_Dataset(
        clinical_data_folder=config.validation["clinical_path"],
        pathology_dir=os.path.join(config.validation["main_dir"], "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.validation["main_dir"], "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
    )

    return tr_dataset, val_dataset

def get_surv_datasets(config, keyfile):
    tr_dataset = MMSurv_Dataset(
        n_bins=config.data["num_time_bins"],
        clinical_data_folder=config.data["clinical_path"],
        pathology_dir=os.path.join(config.data["main_dir"], "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.data["main_dir"], "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
    )  

    val_dataset = MMSurv_Dataset(
        n_bins=config.data["num_time_bins"],
        clinical_data_folder=config.validation["clinical_path"],
        pathology_dir=os.path.join(config.validation["main_dir"], "pathology/features"),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.validation["main_dir"], "radiology/features"),
        pids=None,
        use_unet_features=True,
        keyfile_path=keyfile
        )

    return tr_dataset, val_dataset


def main():
    args = parse_args()
    moco_config = load_config(args.moco_config)
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])

    if config.experiment["objective"] == "surv":
        tr_dataset, val_dataset = get_surv_datasets(config, keyfile)
    else:
        tr_dataset, val_dataset = get_cls_datasets(config, keyfile)

    save_dir = moco_config.experiment["save_dir"]
    model_path = os.path.join(save_dir, "moco_2025-11-14", "version_10", "moco_2025-11-14_best.pth")
    base_model = load_encoder(config=moco_config, model_path=model_path, device=device)
    
    if config.model["type"] == "classifier":
        model = ClassifierHead(
            input_dim=moco_config.model["feature_dim"],
            hidden_dim=1024, 
            num_output=1,
            dropout=0.0
        ).to(device)
    elif config.model["type"] == "admil":
        model = ADMIL_Model(
            n_classes = config.model["num_time_bins"],
            hidden_layers=0,
            feature_size=moco_config.model["feature_dim"],
            dropout=False
        ).to(device)
    else:
        raise NotImplementedError(f"Undefined model type: {config.model['type']}")
    
    if config.training["loss"] == "nll":
        loss = NLLSurvLoss(alpha=config.training["alpha"])
    elif config.training["loss"] == "ce":
        loss = CrossEntropySurvLoss(alpha=config.training["alpha"])
    else:
        raise NotImplementedError(f"Undefined loss: {config.training['loss']}") 
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Initialize trainer
    trainer = SurvTrainer(
        backbone=base_model,
        model=model,
        tr_dataset=tr_dataset,
        val_dataset=val_dataset,
        max_epochs=config.training["epochs"],
        batch_size=config.training["batch_size"],
        learning_rate=config.training["learning_rate"],
        loss=loss,
        optimizer_name=config.training["optimizer"],
        device=device,
        save_dir=config.experiment["save_dir"],
        debug_mode=False,
        model_name=f"{config.experiment['name']}_{date}"
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()