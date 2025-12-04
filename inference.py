from datasets import MMCLS_Dataset, CWZDataset
from models import MMCLRModel, MMViTModel, MMCLSModel, MMSurvivalModel

import torch
import os
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('--config', type=str, help='Path to the config file')
    args = parser.parse_args()
    return args

def load_config(model_path, name="surv_super"):
    config_path = os.path.join(os.path.dirname(model_path), f"{name}_config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return SimpleNamespace(**config)

def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [to_device(d, device) for d in data]
        elif isinstance(data, dict):
            return {k: to_device(v, device) for k, v in data.items()}
        else:
            return data.to(device)

def simple_classifier(model_path):
    config = load_config(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])

    clinical_path = "_gc/task1/test/clinical_data/batch1"
    pathology_path = "_gc/task1/test/pathology/features"
    radiology_path = "_gc/task1/test/radiology/features"

    # Initialize datasets
    dataset = MMCLS_Dataset(
        clinical_data_folder=os.path.join(config.experiment['mounted_dir'], clinical_path),
        pathology_dir=os.path.join(config.experiment['mounted_dir'], pathology_path),
        wsi_crop_grid=48,
        radiology_dir=os.path.join(config.experiment['mounted_dir'], radiology_path),
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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model weights
    #checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=True)

    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in dataloader:
            label=batch[1]
            inputs = to_device(batch[0], device)
            logit, prob, pred = model(inputs)
            predictions.update(dict(zip(label, pred.cpu().numpy())))

            print(f"Processed a patient, True Label: {label.item()}, Predicted: {prob.cpu().numpy()}")

    return predictions


def surivial_analysis(model_path):
    from sksurv.metrics import concordance_index_censored

    #args = parse_args()
    config = load_config(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])

    clinical_vars = config.data["clinical_vars"]

    dataset = CWZDataset(keyfile=keyfile, main_dir=config.experiment["main_dir"], n_bins=config.model["num_time_bins"], 
                         mri_input_size=(20,128,120), wsi_fm_model="prism", augment=False, clinical_vars=clinical_vars)
    dataset.select_split("test")

    model = MMSurvivalModel(clinical_input_dim=dataset.get_num_clinical_vars(), 
                            feature_dim=config.model['feature_dim'],
                            num_modalities=3,
                            use_modality_embeddings=True,
                            model_type="admil",
                            n_bins=config.model["num_time_bins"]
                            ).to(device)
    
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=True)
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    risk_scores = np.zeros(len(dataloader))
    indicators = np.zeros(len(dataloader))
    event_times = np.zeros(len(dataloader))

    with torch.no_grad():
        loop = tqdm(dataloader)
        for i, (inst, label) in enumerate(loop):
            inst = to_device(inst, device) 
            label = to_device(label, device)

            logits, hazards, y_hat = model(inst)

            S = torch.cumprod(1 - hazards, dim=-1)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            risk_scores[i] = risk[0]
            indicators[i] = label['indicator']
            event_times[i] = label['event_time']
    
    c_index = concordance_index_censored(indicators.astype(bool), event_times, risk_scores, tied_tol=1e-8)[0]

    print("C-index:", c_index)


if __name__ == "__main__":
    model_name = "surv_super_2025-12-03"
    path = f"/data/temporary/filip/MMCLR/experiments/{model_name}/version_0/{model_name}_best.pth"
    surivial_analysis(path)

    """
    path = "/data/temporary/filip/MMCLR/experiments/simple_classifier/version_2/simple_classifier_last.pth"
    preds = simple_classifier(path)
    for true, pred in preds.items():
        print(f"True Label: {true.item()}, Predicted Class: {pred}")
    """
    