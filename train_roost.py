import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.dataset_roost_archie import *
from utils.lr_sched import *
from model.roost_single import *
import shutil

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.dataset = RoostDatasetWrapper(**config['dataset'])
        self.criterion = nn.L1Loss()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    @staticmethod
    def _save_config_file(ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            shutil.copy('./config.yaml', os.path.join(ckpt_dir, 'config.yaml'))

    def train(self):
        train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        comp_feas, _ = train_dataset[0]
        elem_emb_len = train_dataset.elem_emb_len
        self.config["model"]["elem_emb_len"] = elem_emb_len

        model = Roost(**self.config["model"])
        
        model = self._load_weights(model)
        model = model.to(self.device)

        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay'])

        optimizer = Adam(
            model.parameters(), self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

        # Scheduler
        # scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=0.5)

        ckpt_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        self._save_config_file(ckpt_dir)

        best_valid_loss = np.inf
        n_iter = 0
        valid_n_iter = 0
        for epoch_counter in range(self.config['epochs']):
            model.train()
            for bn, batch in enumerate(tqdm(train_loader)):
                adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx = batch[0][0].to(self.device), batch[0][1].to(self.device), batch[0][2].to(self.device), batch[0][3].to(self.device), batch[0][4].to(self.device)
                # print(type(batch[1][0]))
                # print(type(batch[1][1]))
                # print(type(batch[1][2]))
                # print(type(batch[1][3]))
                target = batch[-1].to(self.device)

                output = model(elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)
                # print(type(output))
                loss = self.criterion(output.squeeze(), target.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                    print(epoch_counter, bn, 'loss', loss.item())
                n_iter += 1

            print("Start Validation")
            valid_loss = self._validate(model, valid_loader, train_dataset.mu, train_dataset.std)
            self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
            print('Validation', epoch_counter, 'MAE Loss', valid_loss)

            # scheduler.step(valid_loss)

            states = {
                'model_state_dict': model.state_dict(),
                "best_valid_loss": best_valid_loss,
                'epoch': epoch_counter,
                'mu': train_dataset.mu,
                'std': train_dataset.std,
            }

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(states, os.path.join(ckpt_dir, 'model_best.pth'))

           
            torch.save(states, os.path.join(ckpt_dir, 'model.pth'))

            valid_n_iter += 1

        # print("Average Validation MAE:", np.mean(loss_fold))
        # print("Standard Deviation:", np.std(loss_fold))
            
        # Test
        # test_loss = self._validate(model, test_loader, train_dataset.mu, train_dataset.std)
        # print("Test Loss:", test_loss)

    def _validate(self, model, valid_loader, mu, std):
        valid_loss = 0
        with torch.no_grad():
            model.eval()
            for bn, batch in enumerate(tqdm(valid_loader)):
                elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx = batch[0][0].to(self.device), batch[0][1].to(self.device), batch[0][2].to(self.device), batch[0][3].to(self.device), batch[0][4].to(self.device)
                target = batch[-1].to(self.device)   

                output = model(elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)
                # print(type(output))
                output = output * std + mu
                target = target * std + mu
                loss = self.criterion(output.squeeze(), target.squeeze())
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            print("Valid Loss: {:.4f}".format(valid_loss))
        return valid_loss
    
    def _load_weights(self, model):
        try:
            checkpoints_folder = os.path.join('runs', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model
    
def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()