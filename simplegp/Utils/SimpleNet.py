import torch.nn as nn
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class SimpleNeuralNet(nn.Module):

	def __init__(self, input_size, hidden_size, activ):
		super(SimpleNeuralNet, self).__init__()
		self.inputl = nn.Linear(input_size, hidden_size)
		self.hiddenl = nn.Linear(hidden_size, hidden_size)
		self.outputl = nn.Linear(hidden_size, 1)
		self.activ = activ

	def forward(self, x):
		out = self.inputl(x)
		out = self.activ(out)
		for x in range(2):
			out = self.hiddenl(out)
			out = self.activ(out)
		out = self.outputl(out)
		return out


class RegressionDataset(Dataset):

	def __init__(self, csv_file):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
		"""
		self.regression_frame = pd.read_csv(csv_file, delimiter=";")

	def __len__(self):
		return len(self.regression_frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		variables = np.array(self.regression_frame.iloc[idx, :], dtype='float32')

		sample = {'indep': variables[:-1], 'dep': variables[-1]}

		return sample