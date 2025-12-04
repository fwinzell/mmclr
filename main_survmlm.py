from models import MMSurvivalModel
from utils import GenericTrainer, BlendedLoss
from datasets import PairedDataset
from datasets.collate import simple_collate

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

class MLMTrainer(GenericTrainer):
    def __init__(
            self,
            model,
            tr_dataset,
            val_dataset,
            max_epochs,
            batch_size,
            learning_rate,
            optimizer_name,
            device,
            optimizer_hparams,
            loss_hparams,
            save_dir,
            collate_fn=None,
            log=True,
            save=True, 
            model_name=''
    ):
        super(MLMTrainer, self).__init__(
            model=model,
            tr_dataset=tr_dataset,
            val_dataset=val_dataset,
            max_epochs=max_epochs,
            batch_size=batch_size,
            collate_fn=collate_fn,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            device=device,
            optimizer_hparams=optimizer_hparams,
            save_dir=save_dir,
            log=log,
            save=save,
            model_name=model_name
        )

        self.loss_fn = BlendedLoss(**loss_hparams)
        self.best_c = 0

    def _step(self, data, data_, label):
        _, hazards, _ = self.model(data)
        _, hazards_, _ = self.model(data_)
        hazards = torch.cat((hazards, hazards_), dim=0)
        loss = self.loss_fn(hazards, label)

        return loss, hazards
    
    def train(self):

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            risk_scores = np.zeros(len(self.tr_loader))
            indicators = np.zeros(len(self.tr_loader))
            event_times = np.zeros(len(self.tr_loader))
            losses = []
            for i, (data, data_, label) in enumerate(tr_loop):
                data = self.to_device(data, self.device)
                data_ = self.to_device(data_, self.device) 
                label = self.to_device(label, self.device)

                loss, hazards = self._step(data, data_, label)

                loss['task_loss'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                S = torch.cumprod(1 - hazards, dim=-1)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()

                risk_scores[i] = risk[0]
                indicators[i] = label['indicator']
                event_times[i] = label['event_time']

                if self.log:
                    for key in loss:
                        self.writer.add_scalar(key, loss[key].item(), global_step=self.n_iter)
                
                self.n_iter += 1
                losses.append(loss['task_loss'].item())

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss['task_loss'].item())
               
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
                losses[i] = loss['task_loss'].item()

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(loss=loss['task_loss'].item())

        val_c_index = concordance_index_censored(indicators.astype(bool), event_times, risk_scores, tied_tol=1e-8)[0]
        val_loss = torch.mean(losses)
        if self.log:
            self.writer.add_scalar("val_c_index", val_c_index.item(), global_step=ep)
            self.writer.add_scalar("val_loss", val_loss.item(), global_step=ep)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        if val_c_index > self.best_c:
            self.best_c = val_c_index
            print('____New best model____')
            print(f"C-index: {self.best_c}")
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + "_best.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description='MLM Survival training')
    parser.add_argument('--config', type=str, help='Path to config file', default='configs/surv_config.yaml')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)

    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keyfile = f"{config.experiment['mounted_dir']}/prostate/patient_data/MICCAI_keyfiles/MICCAI_keyfile0306251636.xlsx"

    # Set random seed for reproducibility
    torch.manual_seed(config.experiment["seed"])


    tr_dataset = PairedDataset(keyfile=keyfile, 
                               pathology_dir=config.data["pathology_dir"], 
                               radiology_dir=config.data["radiology_dir"],
                               n_bins=config.model["num_time_bins"],
                               use_miccai=True, 
                               mri_input_size=(20,128,120),
                               augment=True, 
                               split='train',
                               clinical_vars=config.data["clinical_vars"])

    val_dataset = PairedDataset(keyfile=keyfile, 
                               pathology_dir=config.data["pathology_dir"], 
                               radiology_dir=config.data["radiology_dir"],
                               n_bins=config.model["num_time_bins"],
                               use_miccai=True, 
                               mri_input_size=(20,128,120),
                               augment=True, 
                               split='val',
                               clinical_vars=config.data["clinical_vars"])


    model = MMSurvivalModel(clinical_input_dim=tr_dataset.get_num_clinical_vars(),
                            feature_dim=config.model['feature_dim'],
                            num_modalities=3,
                            use_modality_embeddings=True,
                            model_type="admil",
                            n_bins=config.model["num_time_bins"]
                            ).to(device)
    
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    loss_hparams = {
        'alpha': 0.15,
        'compute_ranking_loss': True,
        'surv_loss_weight': 0.5, 
        'ranking_loss_weight': 0.5
    }

    # Initialize trainer
    trainer = MLMTrainer(
        model=model,
        tr_dataset=tr_dataset,
        val_dataset=val_dataset,
        max_epochs=config.training["epochs"],
        batch_size=config.training["batch_size"],
        learning_rate=config.training["learning_rate"],
        loss_hparams=loss_hparams,
        optimizer_name=config.training["optimizer"],
        optimizer_hparams=config.training["opt_hparams"],
        device=device,
        save_dir=config.experiment["save_dir"],
        collate_fn=simple_collate,
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