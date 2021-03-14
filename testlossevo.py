# Libraries
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import os


from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.LossFunctionEvoNodes import *
from simplegp.Fitness.LossFunctionEvoFitness import LossFunctionEvoFitness
from simplegp.Evolution.Evolution import SimpleGP
from simplegp.Utils.SimpleNet import SimpleNeuralNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
params = {"input_size": 1, "hidden_size": 100, "num_classes": 1, "num_epochs": 20000, "batch_size": 200,
          "learning_rate": 0.01}

popsizes = [8,16,32]
runs = 3

targetfile = "parabola.csv"
train = pd.read_csv(os.path.expanduser('data/train/' + targetfile), sep=None, engine='python').values
val = pd.read_csv(os.path.expanduser('data/val/' + targetfile), sep=None, engine='python').values
test = pd.read_csv(os.path.expanduser('data/test/' + targetfile), sep=None, engine='python').values

train_loader = torch.utils.data.DataLoader(dataset=train,
                                            batch_size=len(train),
                                            shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val,
                                          batch_size=len(val),
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=len(test),
                                          shuffle=False)

mse = nn.MSELoss()
functions = [AddNode(), SubNode(), MulNode(), DivNode(), LogNode(), AbsNode(), SqrtNode()]
terminals = [FeatureNode("targets"), FeatureNode("outputs")]


for pop in popsizes:
    for i in range(runs):
        fitness_func = LossFunctionEvoFitness(train_loader, val_loader, params, mse, adjusted=True)
        sgp = SimpleGP(fitness_func, functions, terminals, pop_size=pop, max_generations=20, initialization_max_tree_height=2, verbose=True)
        sgp.Run()
