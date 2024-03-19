"""
Adapted from:
Roost: https://github.com/CompRhys/aviary/tree/main
"""

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


class RoostDataset(Dataset):
    def __init__(self, dataset, mu=0.0, std=1.0, elem_embedding="matscholar200", train=True):
        """
        fold: the number for current fold
        max_num_nbr: the maximum number of neighbors while constructing the crystal graph
        radius: the cutoff radius for searching neighbors
        scaler: fit and transform target
        elem_embedding: element embedding file
        train: train or validation mode
        """
        # self.scaler = scaler
        self.is_train = train
        self.df = dataset
        with open(os.path.join('data/element',elem_embedding+'.json')) as file:
             self.elem_features = json.load(file)
        self.elem_emb_len = len(next(iter(self.elem_features.values())))

        self.mu = mu
        self.std = std

        # if self.is_train:
        #     self.df.iloc[:,1] = self.scaler.fit_transform(self.df.iloc[:,1].values.reshape(-1,1))
        # else:
        #     self.df.iloc[:,1] = self.scaler.transform(self.df.iloc[:,1].values.reshape(-1,1))

        # TODO
        # self.df.iloc[:,1] = (self.df.iloc[:,1] - self.mu) / self.std

        # self.df = self.df.iloc[:5, :]
        print("Number of data:", self.df.shape[0])

    # def load_matbench(self, dataset_name):
    #     mb = MatbenchBenchmark(autoload=False,subset=[dataset_name])
    #     for task in mb.tasks:
    #         task.load()
    #         if self.is_train:   
    #             df = task.get_train_and_val_data(self.fold, as_type="df")
    #             df.iloc[:,1] = self.scaler.fit_transform(df.iloc[:,1].values.reshape(-1,1))
    #         else:
    #             df = task.get_test_data(self.fold, as_type="df", include_target=True)
    #             df.iloc[:,1] = self.scaler.transform(df.iloc[:,1].values.reshape(-1,1))
    #         # slices = list(map(self.map.get, list(df.index)))
    #         # df["slices"] = slices
    #     return df
    
    def __len__(self):
        self.len = len(self.df)
        return self.len
    
    # Cache data for faster training
    @functools.lru_cache(maxsize=None)  # noqa: B019
    def __getitem__(self, index):
        formula = self.df.iloc[index, 0]
        target = self.df.iloc[index, 1]

        # Composition
        comp_dict = Composition(formula).get_el_amt_dict()
        elements = list(comp_dict)

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            elem_fea = np.vstack([self.elem_features[element] for element in elements])
        except AssertionError as exc:
            raise AssertionError(
                f"{self.df.index[index]} ({formula}) contains element types not in embedding"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"{self.df.index[index]} ({formula}) composition cannot be parsed into elements"
            ) from exc
        
        n_elems = len(elements)
        self_idx = []
        nbr_idx = []
        for idx in range(n_elems):
            self_idx += [idx] * n_elems
            nbr_idx += list(range(n_elems))

        # convert all data to tensors
        elem_weights = torch.Tensor(weights)
        elem_fea = torch.Tensor(elem_fea)
        self_idx = torch.LongTensor(self_idx)
        nbr_idx = torch.LongTensor(nbr_idx)

        # Target
        target = torch.Tensor([float(target)])

        return (
            (elem_weights, elem_fea, self_idx, nbr_idx),
            target,
        )
    
def collate_batch(
    samples: tuple[
        tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor],
        torch.Tensor,
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting crystal properties.

    Args:
        samples (list): list of tuples for each data point where each tuple contains:
            (elem_fea, nbr_fea, nbr_idx, target)
            - elem_fea (Tensor):  _description_
            - nbr_fea (Tensor):
            - self_idx (LongTensor):
            - nbr_idx (LongTensor):
            - target (Tensor | LongTensor): target values containing floats for regression or
                integers as class labels for classification
            - cif_id: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor]: batched Roost model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            # TODO this last tuple is unpacked how to do type hint?
            *tuple[str | int]: Identifiers like material_id, composition
        ]
    """
    # define the lists
    batch_elem_weights = []
    batch_elem_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_elem_idx = []
    batch_targets = []
    # batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs_comp, target) in enumerate(samples):
        elem_weights, elem_fea, self_idx, nbr_idx = inputs_comp

        n_sites = elem_fea.shape[0]  # number of atoms for this crystal

        # batch the features for Roost together
        batch_elem_weights.append(elem_weights)
        batch_elem_fea.append(elem_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_elem_idx.append(torch.tensor([i] * n_sites))

        # batch the targets and ids
        batch_targets.append(target)

        # increment the id counter
        cry_base_idx += n_sites

    return (
        (
            torch.cat(batch_elem_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_elem_idx),
        ),
        torch.stack(batch_targets, dim=0),
        # *zip(*batch_cry_ids),
    )

class RoostDatasetWrapper(object):
    def __init__(
        self, dataset_name, batch_size, n_splits, seed, valid_size=0.1, elem_embedding="matscholar200", num_workers=4
    ):
        super(object, self).__init__()
        self.df = pd.read_csv(dataset_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.elem_embedding = elem_embedding
        self.valid_size = valid_size
        self.seed = seed

        # splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # self.split_index = splits.split(np.arange(self.df.shape[0]))  

    def get_data_loaders(self):
        num_data = self.df.shape[0]
        indices = list(range(num_data))
        random_state = np.random.RandomState(seed=self.seed)
        random_state.shuffle(indices)
        split = int(np.floor(self.valid_size * num_data))
        test_split = int(np.floor(split / 2))
        test_idx, val_idx, train_idx = indices[:test_split], indices[test_split:split], indices[split:]
        train_data = self.df.loc[train_idx, :].reset_index(drop=True)
        valid_data = self.df.loc[val_idx, :].reset_index(drop=True)
        test_data = self.df.loc[test_idx, :].reset_index(drop=True)
        mu = np.mean(train_data.values[:, -1])
        std = np.std(train_data.values[:, -1])
        print("mu:", mu)
        print("std:", std)
        # scaler = StandardScaler()
        train_dataset = RoostDataset(train_data, mu, std, self.elem_embedding)
        valid_dataset = RoostDataset(valid_data, mu, std, self.elem_embedding, train=False)
        test_dataset = RoostDataset(test_data, mu, std, self.elem_embedding, train=False)
        # TODO
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_batch)
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_batch)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_batch)
        # train_loader = DataLoader(train_dataset, train_data.shape[0], shuffle=False, num_workers=self.num_workers, collate_fn=collate_batch)
        # valid_loader = DataLoader(valid_dataset, valid_data.shape[0], shuffle=False, num_workers=self.num_workers, collate_fn=collate_batch)
        return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader
    

class RoostInferenceDatasetWrapper(object):
    def __init__(
        self, dataset_name, batch_size, n_splits, seed, valid_size=0.1, elem_embedding="matscholar200", num_workers=4
    ):
        super(object, self).__init__()
        # self.df = pd.read_csv(dataset_name, names=['composition'])
        self.df = pd.read_csv(dataset_name)
        self.df.loc[:, "target"] = 0

        # self.df = pd.read_csv(dataset_name)

        self.batch_size = self.df.shape[0]
        self.num_workers = num_workers
        self.elem_embedding = elem_embedding
        self.valid_size = valid_size
        self.seed = seed

        # splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # self.split_index = splits.split(np.arange(self.df.shape[0]))  

    def get_data_loaders(self):
        valid_data = self.df
        valid_dataset = RoostDataset(valid_data, self.elem_embedding, train=False)
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_batch)
        return valid_dataset, valid_loader