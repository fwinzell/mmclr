import torch
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy  as np
from sksurv.metrics import concordance_index_censored

from torch.utils.tensorboard import SummaryWriter

class GenericTrainer(object):
    def __init__(self,
                 model,
                 tr_dataset,
                 val_dataset,
                 max_epochs,
                 batch_size,
                 collate_fn,
                 learning_rate,
                 optimizer_name,
                 device,
                 optimizer_hparams=None,
                 save_dir=None,
                 debug_mode=False,
                 log=True,
                 save=True,
                 model_name="", 
                 **kwargs):
        
        super().__init__(**kwargs)

        self.device = device
        self.model = model
        self.tr_dataset = tr_dataset
        self.val_dataset = val_dataset
        self.max_epochs = max_epochs
        self.batch_sz = batch_size
        self.lr = learning_rate
        self.optimizer_name = optimizer_name
        self.collate_fn = collate_fn

        if optimizer_hparams is None:
            optimizer_hparams = {'lr': learning_rate,
                                'betas': (0.9, 0.999),
                                'eps': 1e-08,
                                'weight_decay': 0.01}

        self.optimizer_hparams = optimizer_hparams
    
        self.save = save
        self.log = log
        self.model_name = model_name

        if save_dir is not None:
            self.save_dir = self.create_save_dir(os.path.join(save_dir, model_name))
        else:
            self.save = False
            self.log = False

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        self.workers = 0
        
        self.tr_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer, self.scheduler = self.configure_optimizers()

        if self.log:
            self.writer = SummaryWriter(self.create_save_dir(os.path.join(save_dir, model_name, "logs")))
        else:
            self.writer = None
        self.n_iter = 0
        self.best_acc = 0

    def train_dataloader(self):
        return DataLoader(self.tr_dataset, batch_size=self.batch_sz, shuffle=True, 
                          drop_last=False, num_workers=self.workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_sz, shuffle=False, 
                          drop_last=False, num_workers=self.workers, collate_fn=self.collate_fn)

    def get_model(self):
        return self.model

    def create_save_dir(self, path):
        dir_exists = True
        i = 0
        while dir_exists:
            save_dir = os.path.join(path, f"version_{str(i)}")
            dir_exists = os.path.exists(save_dir)
            i += 1
        os.makedirs(save_dir)
        return save_dir


    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.model.parameters(), **self.optimizer_hparams)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), **self.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        # Linear warm up of learning rate
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
        # We will reduce the learning rate by 0.1 after 50 and 75 epochs
        decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[10])
        return optimizer, scheduler

    def _step(self, inst, label):
        # Check all devices
        logits, y_prob, y_hat = self.model(inst)
        loss = self.loss_fn(logits.squeeze(), label.squeeze())
        if torch.isnan(loss):
            print("Loss is nan")
        if torch.isinf(loss):
            print("Loss is inf")

        return loss, y_hat, y_prob
    
    @staticmethod
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [GenericTrainer.to_device(d, device) for d in data]
        elif isinstance(data, dict):
            return {k: GenericTrainer.to_device(v, device) for k, v in data.items()}
        else:
            if hasattr(data, "to") and callable(data.to):
                try:
                    return data.to(device)
                except TypeError:
                    # Some objects have .to() but don't accept a device argument
                    return data
            else:
                # If not a tensor, return without device 
                return data
        
        
    def train(self):

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            acc = 0
            losses = []
            for i, (inst, label) in enumerate(tr_loop):
                inst = self.to_device(inst, self.device) 
                label = self.to_device(label, self.device)

                self.optimizer.zero_grad()
                loss, y_hat, y_prob = self._step(inst, label)

                if isinstance(label, dict):
                    y_true = label['label']
                else:
                    y_true = label

                pred = torch.eq(y_hat, y_true).float()

                if self.log:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=self.n_iter)

                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                losses.append(loss.item())

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item(), pred=pred.cpu().numpy())
               
                acc += int(pred.sum())

            acc /= (len(self.tr_loader) * self.batch_size)
            print(f"\nEpoch {epoch}: Mean Accuracy: {acc}")
            if self.log:
                self.writer.add_scalar('train_acc', acc, global_step=epoch)
            self.validate(epoch)
            self.scheduler.step()

            torch.cuda.empty_cache()

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + 'last.pth'))

    def validate(self, ep):
        self.model.eval()

        accs = torch.zeros(len(self.val_loader))
        losses = torch.zeros(len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for i, (inst, label) in enumerate(val_loop):
                inst = self.to_device(inst, self.device) 
                label = self.to_device(label, self.device)

                loss, y_hat, y_prob = self._step(inst, label)

                if isinstance(label, dict):
                    y_true = label['label']
                else:
                    y_true = label

                accs[i] = torch.eq(y_hat, y_true).int() 
                losses[i] = loss.item()

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(loss=loss.item(), acc=accs[i])

        val_acc = torch.mean(accs)
        val_loss = torch.mean(losses)
        if self.log:
            self.writer.add_scalar("val_acc", val_acc.item(), global_step=ep)
            self.writer.add_scalar("val_loss", val_loss.item(), global_step=ep)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print('____New best model____')
            if self.save:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + "_best.pth"))


class SurvTrainer(GenericTrainer):
    def __init__(self,
                 backbone,
                 model,        
                 tr_dataset,
                 val_dataset,
                 max_epochs,
                 batch_size,
                 learning_rate,
                 optimizer_name,
                 device,
                 mode="admil",
                 loss=None,
                 optimizer_hparams=None,
                 save_dir=None,
                 debug_mode=False,
                 log=True,
                 save=True,
                 model_name=""):
        """
        Class for training a classifier on top of a pre-trained model for survival predictions

        model: classifier to be trained
        backbone: pre-trained model, should output a feature vector
        mode: different training modes, options: admil, other -> classification
        """
        
        super(SurvTrainer, self).__init__(
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
            debug_mode=debug_mode,
            log=log,
            save=save,
            model_name=model_name
        )

        self.backbone = backbone
        self.admil = mode == "admil"

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        if loss is None:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.loss_fn = loss

        self.best_c = 0


    def _step(self, inst, label):
        # Check all devices
        with torch.no_grad():
            if self.admil:
                features = []
                for x in inst[2]:
                    x_hat = self.backbone((inst[0], inst[1], x))
                    features.append(x_hat)
                features = torch.stack(features)
            else:
                features = self.backbone(inst)
        
        logits, hazards, _ = self.model(features)
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

                risk_scores[i] = risk
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

                risk_scores[i] = risk
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
    
    