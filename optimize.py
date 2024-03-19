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
from roost_optimizer import RoostOptimizer

def _load_weights(model, config):
      try:
        checkpoints_folder = os.path.join('runs', config['load_model'], 'checkpoints')
        state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
        # model.load_state_dict(state_dict)
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
    config = yaml.load(open("config_optimize.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    with open(os.path.join('data/element', "matscholar200" + '.json')) as file:
            elem_features = json.load(file)
    elem_emb_len = len(next(iter(elem_features.values())))
    config["model"]["elem_emb_len"] = elem_emb_len

    model = Roost(**config["model"])    
    model, mu, std = _load_weights(model, config)

    model.eval()

    optimizer = RoostOptimizer(model, elem_features, config["num_steps"])

    start_comp = config["start_composition"]
    formula = []
    for elem in list(start_comp.keys()):
        formula.append(elem)
        formula.append(str(start_comp[elem]))
    formula = ''.join(formula)
    print("formula:", formula)

    optimized_weights, optimized_properties = optimizer.optimize(formula, mu, std, **config["optimizer"])
    # print("shape of weights traj:", optimized_weights.shape)
    # print(optimized_weights)

    optimized_traj = np.concatenate([optimized_weights, optimized_properties.reshape(-1,1)], axis=1)
    # print("shape of optimized traj:", optimized_traj.shape)

    np.savetxt(config["save_path"], optimized_traj)

if __name__ == "__main__":
    main()