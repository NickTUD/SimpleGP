# Libraries
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from simplegp.Utils.SimpleNet import RegressionDataset

from copy import deepcopy

from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.LossFunctionEvoNodes import *
from simplegp.Fitness.FitnessFunction import LossFunctionEvoFitness
from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Utils.SimpleNet import SimpleNeuralNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
params = {"input_size": 11, "hidden_size": 100, "num_classes": 1, "num_epochs": 100, "batch_size": 200,
          "learning_rate": 0.001}

init_dataset = RegressionDataset(csv_file='~\\Documents\\Thesis\SimpleGP\\data\\winequality-red.csv')
splits = [round(len(init_dataset)*0.8), round(len(init_dataset)*0.2)]
train, test = random_split(init_dataset, splits)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                            batch_size=len(train),
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=len(test),
                                          shuffle=False)

mse = nn.MSELoss()
model = SimpleNeuralNet(params["input_size"], params["hidden_size"] , nn.Sigmoid())
fitness_func = LossFunctionEvoFitness(model, train_loader, test_loader, params, mse)

functions = [AddNode(), SubNode(), MulNode(), DivNode(), LogNode(), SumNode(), AbsNode()]
terminals = [FeatureNode("target"), FeatureNode("output")]

for i in range(3):
    sgp = SimpleGP(fitness_func, functions, terminals, pop_size=10, max_generations=30, verbose=True)
    sgp.Run()
