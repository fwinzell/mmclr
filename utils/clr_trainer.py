import torch
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class MoCoTrainer(object):
    def __init__(self,
                 model,
                 dataset,
                 max_epochs,
                 batch_size,
                 virtual_batch_size,
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
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.batch_sz = batch_size
        self.lr = learning_rate
        self.optimizer_name = optimizer_name
        self.accum_steps = virtual_batch_size // batch_size

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

        # Using cross-entropy loss after computing softmax over all keys in the MoCoWrapper class is
        # equivalent to using the InfoNCE loss
        self.criterion = nn.CrossEntropyLoss().to(device)

        if debug_mode:
            self.workers = 0
        else:
            self.workers = 4

        self.loader = self.dataloader()
    
        self.optimizer, self.scheduler = self.configure_optimizers()

        if self.log:
            self.writer = SummaryWriter(self.create_save_dir(os.path.join(save_dir, model_name, "logs")))
        else:
            self.writer = None
        self.n_iter = 0
        self.best_acc = 0

    def dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_sz,
                          shuffle=True, drop_last=True, num_workers=self.workers)


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
    
    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    
    @staticmethod
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [MoCoTrainer.to_device(d, device) for d in data]
        elif isinstance(data, dict):
            return {k: MoCoTrainer.to_device(v, device) for k, v in data.items()}
        else:
            return data.to(device)


    def train(self):
        self.model.train()

        top_acc = 0
        for epoch in range(self.max_epochs):
            tr_loop = tqdm(self.loader)
            acc = 0
            losses = []
            for i, (input_q, input_k) in enumerate(tr_loop):
                input_q = self.to_device(input_q, self.device)
                input_k = self.to_device(input_k, self.device) 
                
                # Gradient accumulation
                if i % self.accum_steps == 0:
                    self.optimizer.zero_grad()

                logits, labels, query = self.model(input_q, input_k)
                
                loss = self.criterion(logits, labels) / self.accum_steps
                loss.backward()

                if (i + 1) % self.accum_steps == 0:
                    self.optimizer.step()
                    self.n_iter += 1

                losses.append(loss.item() * self.accum_steps)

                acc1, acc5 = self.accuracy(logits, labels, topk=(1, 5))

                mean_std = query.std(dim=0).mean()

                if self.log:
                    self.writer.add_scalar('train_loss', loss.item() * self.accum_steps, global_step=self.n_iter)
                    self.writer.add_scalar('embedding_std', mean_std.item(), global_step=self.n_iter)
                    self.writer.add_scalar('train_acc1', acc1.item(), global_step=self.n_iter)
                    self.writer.add_scalar('train_acc5', acc5.item(), global_step=self.n_iter)

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item() * self.accum_steps, 
                                    acc1=acc1.cpu().numpy(), 
                                    std=mean_std.cpu().detach().numpy())
               
                acc += int(acc1)

            acc /= len(self.loader)
            print(f"\nEpoch {epoch}: Mean Accuracy: {acc}")
            if self.log:
                self.writer.add_scalar('mean_epoch_acc', acc, global_step=epoch)
            self.scheduler.step()

            if acc > top_acc:
                top_acc = acc
                if self.save:
                    # Save only query encoder
                    torch.save(self.model.encoder_q.state_dict(), os.path.join(self.save_dir, self.model_name + '_best.pth'))


        if self.save:
            torch.save(self.model.encoder_q.state_dict(), os.path.join(self.save_dir, self.model_name + '_last.pth'))

    def validate(self, ep):
        self.model.eval()
        # Implement an intermediate testing layer here to monitor progress
        # For example, train a one layer MLP on the output to predict BCR 
        # Do this sort of every 10-25 epochs
        pass