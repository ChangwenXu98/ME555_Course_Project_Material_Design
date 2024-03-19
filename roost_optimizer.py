import os
import json
import warnings
from typing import Dict, List, TYPE_CHECKING, Any, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from matbench.bench import MatbenchBenchmark
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import functools
from pymatgen.core import Composition
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

class RoostOptimizer:
    def __init__(self, model, elem_features, num_steps=100, grad_tol=1e-6):
        self.model = model
        self.num_steps = num_steps
        self.grad_tol = grad_tol
        self.elem_features = elem_features
    
    def optimize(self, formula, mu, std, initial_lr=0.01, lr_decay=0.1, t=0.01, patience=100):
        # Create a composition object from the formula
        print("initial lr:", initial_lr)
        print("lr decay:", lr_decay)
        print("t:", t)
        print("patience:", patience)

        comp_dict = Composition(formula).get_el_amt_dict()
        elements = list(comp_dict)

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)
        weights = torch.cat([torch.tensor(weights, requires_grad=True, dtype=torch.float32)], dim=0)
        
        # Create the input features for the model
        elem_fea = self._get_elem_features(elements)
        self_idx, nbr_idx = self._get_atom_indices(elements)
        cry_elem_idx = torch.cat([torch.tensor([0] * len(elements))])

        weights = weights.detach().requires_grad_(True)

        # print("shape of elem_fea:", elem_fea.shape)
        # print("shape of weights:", weights.shape)

        # print("shape of elem_fea:", elem_fea.shape)
        # print("data type:", elem_fea.dtype)
        
        # Perform optimization
        # lr = initial_lr
        weights_traj = []
        weights_traj.append(deepcopy(weights.T.detach().numpy()))
        # print("appended weights:", weights.T.detach().numpy())
        # print(weights_traj)
        property_traj = []
        for step in tqdm(range(self.num_steps)):
            # Forward pass through the model
            lr = initial_lr
            output = self.model(weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)

            # print(weights_traj)

            property_traj.append(output.detach().numpy().flatten() * std + mu)
            
            # Backpropagate the gradients
            output.backward()

            # print(weights_traj)

            # print("Finish backward pass")
            
            # print(output)
            # print(weights)
            # print(weights.grad)

            # Get the gradients
            gradients = weights.grad

            # Compute the norm of the gradients
            grad_norm = torch.norm(gradients)

            # TODO normalize the gradient (or not)
            # gradients = gradients / grad_norm

            # print("Finish norm")
            # print("step:", step)
            # print("grad norm:", grad_norm)
            # print("gradients:", gradients)
            
            # Check the stopping criterion
            if grad_norm < self.grad_tol:
                print(f"Optimization converged at step {step}")
                break

            # print("Start line search")
            # Perform inexact line search to determine the learning rate
            lr = self._inexact_line_search(weights, output, gradients, lr, lr_decay, t, patience, elem_fea, self_idx, nbr_idx, cry_elem_idx)
            # print(lr)
            # print(type(lr))
            # print("Finish line search")
            # Update the weights using the determined learning rate
            # print(weights_traj)
            weights.data += lr * gradients
            weights.data = torch.clamp(weights.data, min=1e-5, max=1.0)

            # print(weights_traj)

            # print("lr:", lr)
            # print("new weights:", weights.data)

            # Zero the gradients for the next iteration
            weights.grad.data.zero_()

            # Normalize the weights to ensure they sum up to 1
            # weights.data = F.softmax(weights.data, dim=0)
            weights.data = weights.data / torch.norm(weights.data, p=1)
            # print("weights.data:", weights.data)

            weights = weights.detach().requires_grad_(True)

            # print(weights_traj)

            weights_traj.append(deepcopy(weights.T.detach().numpy()))
            # print("appended weights:", weights.T.detach().numpy())
            # print("appended weight:", weights)
            # print(weights.T.detach().numpy().shape)

        if len(weights_traj) == len(property_traj):
            return np.concatenate(weights_traj, axis=0), np.concatenate(property_traj, axis=0)
        elif len(weights_traj) == len(property_traj) + 1:
            # print(weights_traj)
            return np.concatenate(weights_traj[:-1], axis=0), np.concatenate(property_traj, axis=0)
        else:
            raise ValueError(
                "Wrong Iteration"
            )
    
    def _get_elem_features(self, elements):
        # Extract element features for the given composition
        try:
            elem_fea = np.vstack([self.elem_features[element] for element in elements])
            # return torch.Tensor(elem_fea, dtype=torch.double).reshape(len(elements),-1)
            return torch.cat([torch.Tensor(elem_fea)], dim=0)
        except AssertionError as exc:
            raise AssertionError(
                "Contains element types not in embedding"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "Composition cannot be parsed into elements"
            ) from exc
    
    def _get_atom_indices(self, elements):
        # Generate self_idx and nbr_idx for the given composition
        n_elems = len(elements)
        self_idx = []
        nbr_idx = []
        for idx in range(n_elems):
            self_idx += [idx] * n_elems
            nbr_idx += list(range(n_elems))
        return torch.cat([torch.LongTensor(self_idx)], dim=0), torch.cat([torch.LongTensor(nbr_idx)], dim=0)
    
    def _inexact_line_search(self, weights, output, gradients, lr, lr_decay, t, patience, elem_fea, self_idx, nbr_idx, cry_elem_idx):
        # Perform inexact line search to determine the learning rate
        count = 0
        # print("Start line search")
        while count < patience:
            
            # Create a copy of the weights
            new_weights = weights.data + lr * gradients
            
            # Normalize the new weights
            # new_weights = F.softmax(new_weights, dim=0)
            new_weights = new_weights / torch.norm(new_weights, p=1)
            
            # Evaluate the model with the new weights
            phi = self.model(new_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)
            
            # Check if the output has increased
            if phi > output + t * lr * torch.matmul(gradients.reshape(1,-1), gradients.reshape(-1,1)):
                # print("Early stop line search")
                return lr
            
            # if phi > output:
            #     return lr
            
            # Decrease the learning rate
            lr *= lr_decay

            count += 1
            # print("count:", count)
            # print("lr:", lr)
        print("terminate because line search takes too long")
        return lr 