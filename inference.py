from datasets import MMCLS_Dataset
from MMCLR.main_simple_cls import parse_args, load_config
from models import MMCLRModel, MMViTModel, MMCLSModel

import torch
import os
from types import SimpleNamespace

def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [to_device(d, device) for d in data]
        elif isinstance(data, dict):
            return {k: to_device(v, device) for k, v in data.items()}
        else:
            return data.to(device)

def simple_classifier(model_path):
    args = parse_args()
    config = load_config(args.config)

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

if __name__ == "__main__":
    path = "/data/temporary/filip/MMCLR/experiments/simple_classifier/version_2/simple_classifier_last.pth"
    preds = simple_classifier(path)
    for true, pred in preds.items():
        print(f"True Label: {true.item()}, Predicted Class: {pred}")