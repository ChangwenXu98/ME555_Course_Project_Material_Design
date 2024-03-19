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
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        # self.log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        # self.writer = SummaryWriter(log_dir=self.log_dir)
        # self.dataset = RoostDatasetWrapper(**config['dataset'])
        self.dataset = RoostInferenceDatasetWrapper(**config['dataset'])
        self.criterion = nn.L1Loss()

        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)

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
        valid_dataset, valid_loader = self.dataset.get_data_loaders()
        comp_feas, _ = valid_dataset[0]
        elem_emb_len = valid_dataset.elem_emb_len
        self.config["model"]["elem_emb_len"] = elem_emb_len

        model = Roost(**self.config["model"])
        
        model, mu, std = self._load_weights(model)
        model = model.to(self.device)

        print("Start Inference")
        # output, target = self._validate(model, valid_loader, mu, std)
        output = self._validate(model, valid_loader, mu, std)
        valid_dataset.df.iloc[:, -1] = output
        valid_dataset.df.to_csv(self.config["save_path"], index=None)
        # mae = np.sum(np.abs(output - target)) / len(output)
        # r2 = r2_score(output, target)
        # print("MAE:", mae)
        # print("r2:", r2)
        # x = np.arange(0, 0.15, 0.01)
        # y = np.arange(0, 0.15, 0.01)

        # plt.scatter(output, target, c='blue')
        # plt.plot(x, y, 'black', linestyle='--')
        # plt.xlabel("Prediction")
        # plt.ylabel("Ground Truth")
        # plt.legend()
        # plt.savefig(self.config["save_path"])

    def _validate(self, model, valid_loader, mu, std):
        # valid_loss = 0
        with torch.no_grad():
            model.eval()
            for bn, batch in enumerate(tqdm(valid_loader)):
                elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx = batch[0][0].to(self.device), batch[0][1].to(self.device), batch[0][2].to(self.device), batch[0][3].to(self.device), batch[0][4].to(self.device)
                target = batch[-1].to(self.device)   

                output = model(elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)
                # print(type(output))
                output = output * std + mu
                # loss = self.criterion(output.squeeze(), target.squeeze())
                # valid_loss += loss.item()

            # valid_loss /= len(valid_loader)
            # print("Valid Loss: {:.4f}".format(valid_loss))
        # return output.cpu().numpy().flatten(), target.cpu().numpy().flatten()
        return output.cpu().numpy().flatten()
    
    def _load_weights(self, model):
        try:
            checkpoints_folder = os.path.join('runs', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_best.pth'), map_location=torch.device('cpu'))
            model.load_state_dict(state_dict["model_state_dict"])
            print("Loaded pre-trained model with success.")
            mu = state_dict["mu"]
            std = state_dict["std"]
            print("mu:", mu)
            print("std:", std)
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model, mu, std
    
def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()