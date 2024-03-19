import random
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

class GA_Optimizer:
    def __init__(self, model, mu, std, elem_features, 
                 pop_size, num_generations, crossover_rate, element_mutation_rate, ratio_mutation_rate):
        self.model = model
        self.mu = mu
        self.std = std
        self.elem_features = elem_features
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.element_mutation_rate = element_mutation_rate
        self.ratio_mutation_rate = ratio_mutation_rate

        # self.elements_ls = ["Al", "Cr", "Fe", "Ni", "Co", "Cu", "Zn", "Mg", "Sm", "Si"]
        self.elements_ls = ["Al", "Cr", "Ni", "Cu", "Zn", "Mg", "Sm", "Si"]

    def surrogate_model(self, formula):
        comp_dict = Composition(formula).get_el_amt_dict()
        elements = list(comp_dict)

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)
        weights = torch.cat([torch.Tensor(weights)], dim=0)
        
        # Create the input features for the model
        elem_fea = self._get_elem_features(elements)
        self_idx, nbr_idx = self._get_atom_indices(elements)
        cry_elem_idx = torch.cat([torch.tensor([0] * len(elements))])

        output = self.model(weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)
        return output.detach().numpy().flatten()[0] * self.std + self.mu
        # return random.random()

    def _get_elem_features(self, elements):
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

    def fitness(self, individual):
        element_indices = individual[:3]
        ratios = individual[3:]
        alloy_composition = "".join(f"{self.elements_ls[index]}{ratio:.2f}" for index, ratio in zip(element_indices, ratios))
        # print("alloy composition:", alloy_composition)
        material_property = self.surrogate_model(alloy_composition)
        return material_property

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            element_indices = random.sample(range(len(self.elements_ls)), 3)
            ratios = np.random.dirichlet(np.ones(3))
            individual = element_indices + list(ratios)
            population.append(individual)
        return population

    def crossover(self, parent1, parent2):
        element_indices1 = parent1[:3]
        element_indices2 = parent2[:3]
        offspring_elements = [random.choice([i, j]) for i, j in zip(element_indices1, element_indices2)]

        ratios1 = parent1[3:]
        ratios2 = parent2[3:]
        alpha = random.random()
        offspring_ratios = [alpha * i + (1 - alpha) * j for i, j in zip(ratios1, ratios2)]
        offspring_ratios = [ratio / sum(offspring_ratios) for ratio in offspring_ratios]

        offspring = offspring_elements + offspring_ratios
        return offspring

    def mutation(self, individual):
        for i in range(3):
            if random.random() < self.element_mutation_rate:
                individual[i] = random.randint(0, len(self.elements_ls) - 1)

        for i in range(3, 6):
            if random.random() < self.ratio_mutation_rate:
                individual[i] = random.uniform(0, 1)

        ratios = individual[3:]
        ratios = [ratio / sum(ratios) for ratio in ratios]
        individual[3:] = ratios

        return individual

    def replacement(self, population, offspring):
        combined_population = population + offspring
        sorted_population = sorted(combined_population, key=lambda x: self.fitness(x), reverse=True)
        new_population = sorted_population[:len(population)]
        return new_population

    def genetic_algorithm(self):
        population = self.initialize_population()

        # print("initial population:", population)

        for generation in range(self.num_generations):
            offspring = []

            for _ in range(self.pop_size // 2):
                parent1, parent2 = random.sample(population, 2)

                if random.random() < self.crossover_rate:
                    offspring1 = self.crossover(parent1, parent2)
                    offspring2 = self.crossover(parent2, parent1)
                else:
                    offspring1, offspring2 = parent1, parent2

                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)

                offspring.extend([offspring1, offspring2])
                # print("offspring:", offspring)

            population = self.replacement(population, offspring)
            # print("final population:", population)

        best_individual = max(population, key=lambda x: self.fitness(x))
        element_indices = best_individual[:3]
        ratios = best_individual[3:]
        best_alloy_composition = "".join(f"{self.elements_ls[index]}{ratio:.2f}" for index, ratio in zip(element_indices, ratios))

        return best_alloy_composition

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
    config = yaml.load(open("config_ga.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    with open(os.path.join('data/element', "matscholar200" + '.json')) as file:
            elem_features = json.load(file)
    elem_emb_len = len(next(iter(elem_features.values())))
    config["model"]["elem_emb_len"] = elem_emb_len

    model = Roost(**config["model"])    
    model, mu, std = _load_weights(model, config)

    model.eval()

    optimizer = GA_Optimizer(model, mu, std, elem_features, **config["optimizer"])

    best_alloy = optimizer.genetic_algorithm()
    best_property = optimizer.surrogate_model(best_alloy)
    print("Best alloy composition:", best_alloy)
    print("Best alloy property:", best_property)

if __name__ == "__main__":
    main()