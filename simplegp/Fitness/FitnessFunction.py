import numpy as np
from copy import deepcopy
from simplegp.Utils.SimpleNet import SimpleNeuralNet
import torch

class SymbolicRegressionFitness:

	def __init__( self, X_train, y_train, use_linear_scaling=True ):
		self.X_train = X_train
		self.y_train = y_train
		self.use_linear_scaling = use_linear_scaling
		self.elite = None
		self.elite_scaling_a = 0.0
		self.elite_scaling_b = 1.0
		self.evaluations = 0
		self.archive = {}

	def Evaluate( self, individual ):

		self.evaluations = self.evaluations + 1

		output = individual.GetOutput( self.X_train )

		a = 0.0
		b = 1.0

		if self.use_linear_scaling:
			b = np.cov(self.y_train, output)[0,1] / (np.var(output) + 1e-10)
			a = np.mean(self.y_train) - b*np.mean(output)

		scaled_output = a + b*output

		fit_error = np.mean( np.square( self.y_train - scaled_output ) )
		if np.isnan(fit_error):
			fit_error = np.inf

		individual.fitness = fit_error

		if not self.elite or individual.fitness < self.elite.fitness:
			del self.elite
			self.elite = deepcopy(individual)
			self.elite_scaling_a = a
			self.elite_scaling_b = b


class LossFunctionEvoFitness:

	def __init__(self, trainloader, testloader, params, eval_func):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.trainloader = trainloader
		self.testloader =  testloader
		self.params = params
		self.eval_func = eval_func
		self.evaluations = 0
		self.elite = None
		self.model = None
		self.fitness_archive = {}

	def Evaluate(self, individual):

		key = individual.GetHumanExpression()
		if key in self.fitness_archive:
			individual.fitness = self.fitness_archive[key]

		else:

			self.model = SimpleNeuralNet(self.params["input_size"], self.params["hidden_size"] , torch.nn.Sigmoid()).to(self.device)

			self.evaluations = self.evaluations + 1

			# Loss and optimizer
			optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

			loss_func_str = individual.GetPytorchExpression()

			# Train the model
			total_step = len(self.trainloader)
			for epoch in range(self.params["num_epochs"]):
				for i, data in enumerate(self.trainloader):
					# Move tensors to the configured device
					indep = data["indep"].to(self.device)
					target = data["dep"].to(self.device)

					# Forward pass
					output = self.model(indep)
					loss = torch.mean(eval(loss_func_str))


					# Backward and optimize
					optimizer.zero_grad(set_to_none=True)
					try:
						loss.backward()
					except RuntimeError as e:
						print('Individual {} had the following error: {}'.format(key, e))
						individual.fitness = np.inf
						self._updateElite(individual)
						return
					optimizer.step()

			print('Individual {} has the following training loss: {}'.format(key, loss.item()))

			if np.isnan(loss.item()):
				individual.fitness = np.inf
			else:
				with torch.no_grad():
					for test_data in self.testloader:
						test_indep = test_data["indep"].to(self.device)
						test_target = test_data["dep"].to(self.device)

						# Forward pass
						test_outputs = self.model(test_indep)
						mse_loss = self.eval_func(test_outputs, test_target).item()
						if np.isnan(mse_loss):
							mse_loss = np.inf

				individual.fitness = mse_loss

		print('Individual {} has the following fitness: {}'.format(key, individual.fitness))

		self._updateElite(individual)

	def _updateElite(self, individual):
		if not self.elite or individual.fitness < self.elite.fitness:
			del self.elite
			self.elite = deepcopy(individual)
