import torch
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class SimpleTrainer(object):
    def __init__(self,
                 model,
                 tr_dataset,
                 val_dataset,
                 max_epochs,
                 batch_size,
                 learning_rate,
                 optimizer_name,
                 device,
                 optimizer_hparams=None,
                 save_dir=None,
                 debug_mode=False,
                 log=True,
                 save=True,
                 model_name=""):
        
        super().__init__()

        self.device = device
        self.model = model
        self.tr_dataset = tr_dataset
        self.val_dataset = val_dataset
        self.max_epochs = max_epochs
        self.batch_sz = batch_size
        self.lr = learning_rate
        self.optimizer_name = optimizer_name

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
            self.save_dir = self.create_save_dir(os.path.join(save_dir, "models"))
        else:
            self.save = False
            self.log = False

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        if debug_mode:
            self.workers = 0
        else:
            self.workers = 10

        self.tr_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer, self.scheduler = self.configure_optimizers()

        if self.log:
            self.writer = SummaryWriter(self.create_save_dir(os.path.join(save_dir, "logs")))
        else:
            self.writer = None
        self.n_iter = 0
        self.best_acc = 0

    def train_dataloader(self):
        return DataLoader(self.tr_dataset, batch_size=self.batch_sz,
                          shuffle=True, drop_last=False, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_sz,
                          shuffle=False, drop_last=False, num_workers=self.workers)

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
        logits, y_prob, y_hat = self.model(inst)
        loss = self.loss_fn(logits.squeeze(), label.squeeze())
        if torch.isnan(loss):
            print("Loss is nan")
        if torch.isinf(loss):
            print("Loss is inf")

        return loss, y_hat, y_prob


    def train(self):

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            acc = 0
            losses = []
            for i, (pid, inst, label) in enumerate(tr_loop):
                inst, label = inst.squeeze().to(self.device), label.to(self.device).float()

                self.optimizer.zero_grad()
                loss, y_hat, _ = self._step(inst, label)
                pred = torch.eq(y_hat, label).float()
                pred = pred[~torch.isnan(label)]

                if self.log:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=self.n_iter)

                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                losses.append(loss.item())

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item(), pred=pred.cpu().numpy())
                if self.long_mode:
                    acc += int(torch.mean(pred))
                else:
                    acc += int(pred)

            acc /= len(self.tr_loader)
            print(f"\nEpoch {epoch}: Mean Accuracy: {acc}")
            if self.log:
                self.writer.add_scalar('train_acc', acc, global_step=epoch)
            self.validate(epoch)
            self.scheduler.step()

        if self.save:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + 'last.pth'))

    def validate(self, ep):
        self.model.eval()

        accs = torch.zeros(len(self.val_loader))
        losses = torch.zeros(len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for i, (pid, inst, label) in enumerate(val_loop):
                inst, label = inst.squeeze().to(self.device), label.to(self.device).float()

                loss, y_hat, y_prob = self._step(inst, label)

                if self.long_mode:
                    accs[i] = int(torch.mean(torch.eq(y_hat, label).float()))
                else:
                    accs[i] = torch.eq(y_hat, label).int() #label.item() * y_prob.item() + (1-label.item()) * (1-y_prob.item())  # Cross-entropy ish
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
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model_name + "best.pth"))


  