from models import MMSurvivalModel
from utils import GenericTrainer, NLLSurvLoss, CrossEntropySurvLoss
from datasets import CWZDataset

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy  as np
from sksurv.metrics import concordance_index_censored

import argparse
import yaml
from types import SimpleNamespace
import datetime

class SupervisedTrainer(GenericTrainer):
    def __init__(
            self,
            model,
            tr_dataset,
            val_dataset,
            max_epochs,
            batch_size,
            learning_rate,
            loss,
            optimizer_name,
            device,
            optimizer_hparams,
            save_dir,
            log,
            save, 
            model_name
    ):
        super(SupervisedTrainer, self).__init__(
            model=model,
            tr_dataset=tr_dataset,
            val_dataset=val_dataset,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            device=device,
            optimizer_hparams=optimizer_hparams,
            save_dir=save_dir,
            log=log,
            save=save,
            model_name=model_name
        )

        if loss is None:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.loss_fn = loss

        self.best_c = 0

    def _step(self, inst, label):
        logits, hazards, _ = self.model(inst)
        loss = self.loss_fn(hazards, label)
        if torch.isnan(loss):
            print("Loss is nan")
        if torch.isinf(loss):
            print("Loss is inf")

        return loss, hazards
    
    def train(self):

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            risk_scores = np.zeros(len(self.tr_loader))
            indicators = np.zeros(len(self.tr_loader))
            event_times = np.zeros(len(self.tr_loader))
            losses = []
            for i, (inst, label) in enumerate(tr_loop):
                inst = self.to_device(inst, self.device) 
                label = self.to_device(label, self.device)

                self.optimizer.zero_grad()
                loss, hazards = self._step(inst, label)

                S = torch.cumprod(1 - hazards, dim=-1)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()

                risk_scores[i] = risk[0]
                indicators[i] = label['indicator']
                event_times[i] = label['event_time']

                if self.log:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=self.n_iter)

                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                losses.append(loss.item())

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item())
               
            epoch_loss = np.mean(losses)
            c_index = concordance_index_censored(indicators.astype(bool), event_times, risk_scores, tied_tol=1e-8)[0]

    
            print(f"\nEpoch {epoch}: Mean Loss: {epoch_loss} C-index: {c_index}")
            if self.log:
                self.writer.add_scalar('epoch_loss', epoch_loss, global_step=epoch)
                self.writer.add_scalar('c_index', c_index, global_step=epoch)
            self.validate(epoch)
            self.scheduler.step()

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + '_last.pth'))

    def validate(self, ep):
        self.model.eval()

        risk_scores = np.zeros(len(self.val_loader))
        indicators = np.zeros(len(self.val_loader))
        event_times = np.zeros(len(self.val_loader))
        losses = torch.zeros(len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for i, (inst, label) in enumerate(val_loop):
                inst = self.to_device(inst, self.device) 
                label = self.to_device(label, self.device)

                loss, hazards = self._step(inst, label)

                S = torch.cumprod(1 - hazards, dim=-1)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()

                risk_scores[i] = risk[0]
                indicators[i] = label['indicator']
                event_times[i] = label['event_time']

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(loss=loss.item())

        val_c_index = concordance_index_censored(indicators.astype(bool), event_times, risk_scores, tied_tol=1e-8)[0]
        val_loss = torch.mean(losses)
        if self.log:
            self.writer.add_scalar("val_c_index", val_c_index.item(), global_step=ep)
            self.writer.add_scalar("val_loss", val_loss.item(), global_step=ep)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        if val_c_index > self.best_c:
            self.best_c = val_c_index
            print('____New best model____')
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + "_best.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description='MMCLR Main Training Script')
    parser.add_argument('--config', type=str, help='Path to config file', default='configs/surv_config.yaml')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)

    return config


def data_test():
    import argparse
    parser = argparse.ArgumentParser(description='Test Models Main Script')
    parser.add_argument('-c', '--cluster', action='store_true', help='Flag for being on the cluster')
    args = parser.parse_args()
    if args.cluster:
        # SOL paths
        print("Cluster")
        main_dir = "/data/temporary/filip/cohorts/cwz_retrospective/"
        keyfile = "/data/pa_cpgarchive//projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"
    else:
        # local mac paths
        print("Local")
        main_dir = "/Volumes/temporary/filip/cohorts/cwz_retrospective"
        keyfile = "/Volumes/PA_CPGARCHIVE/projects/chimera/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    dataset = CWZDataset(keyfile=keyfile, main_dir=main_dir, n_bins=5, 
                         mri_input_size=(20,128,120), wsi_fm_model="prism")
    
    dataset.select_split("train")
    
    #loader = DataLoader(dataset, batch_size=4)
    data, label_dict = dataset[0]
    print(f"Clinical Data: {data[0]}")
    print(f"Label: {label_dict['label']}")
    print(f"Time: {label_dict['event_time']} months")
    print(f"BCR?: {label_dict['indicator']}")

    print(f"MRI shapes: t2w: {data[1]['t2w'].shape}, adc: {data[1]['adc'].shape}, hbv: {data[1]['hbv'].shape}")

def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])

    clinical_vars = config.data["clinical_vars"]
    #clinical_vars = ["anon_pid", "age_at_prostatectomy", "ISUP", "pre_operative_PSA", "pT_stage", 
    #                 "positive_lymph_nodes", "capsular_penetration",	"positive_surgical_margins", 
    #                 "invasion_seminal_vesicles", "lymphovascular_invasion"]

    tr_dataset = CWZDataset(keyfile=keyfile, main_dir=config.experiment["main_dir"], n_bins=config.model["num_time_bins"], 
                         mri_input_size=(20,128,120), wsi_fm_model="prism", augment=True, clinical_vars=clinical_vars)
    tr_dataset.select_split("train")

    val_dataset = CWZDataset(keyfile=keyfile, main_dir=config.experiment["main_dir"], n_bins=config.model["num_time_bins"], 
                         mri_input_size=(20,128,120), wsi_fm_model="prism", augment=False, clinical_vars=clinical_vars)
    val_dataset.select_split("val")

    model = MMSurvivalModel(clinical_input_dim=tr_dataset.get_num_clinical_vars(),
                            feature_dim=config.model['feature_dim'],
                            num_modalities=3,
                            use_modality_embeddings=True,
                            model_type="admil",
                            n_bins=config.model["num_time_bins"]
                            ).to(device)
    
    if config.training["loss"] == "nll":
        loss = NLLSurvLoss(alpha=config.training["alpha"])
    elif config.training["loss"] == "ce":
        loss = CrossEntropySurvLoss(alpha=config.training["alpha"])
    else:
        raise NotImplementedError(f"Undefined loss: {config.training['loss']}") 
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Initialize trainer
    trainer = SupervisedTrainer(
        model=model,
        tr_dataset=tr_dataset,
        val_dataset=val_dataset,
        max_epochs=config.training["epochs"],
        batch_size=config.training["batch_size"],
        learning_rate=config.training["learning_rate"],
        loss=loss,
        optimizer_name=config.training["optimizer"],
        optimizer_hparams=config.training["opt_hparams"],
        device=device,
        save_dir=config.experiment["save_dir"],
        log=True,
        save=True,
        model_name=f"{config.experiment['name']}_{date}"
    )

    # Start training
    trainer.train()

    # Save config
    with open(trainer.save_dir + f"/{config.experiment['name']}_config.yaml", 'w') as file:
        yaml.safe_dump(vars(config), file)

if __name__ == "__main__":
    main()



