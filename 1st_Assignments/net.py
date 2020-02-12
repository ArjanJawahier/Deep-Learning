import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		torch.manual_seed(0) # Reproduction purposes (Always generate the same initial weights)

		# Make layers 
		self.conv1 = nn.Conv2d(1, 6, 3) # 1 input channel, 6 output channels, 3x3 convolution
		self.conv2 = nn.Conv2d(6, 16, 3)

		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10) 	# 10 Output classes

	def forward(self, x, **kwargs):
		"""Second argument to a function call should be the activation function.
		We can test all standard activation functions here.
		"""
		if "activation_func" not in kwargs.keys():
			print("ERROR: Specify activation function as a keyworded argument.")
			exit(-1)
		else:
			activation_func = kwargs["activation_func"]

		# x = activation_func(self.conv1(x))
		x = activation_func(self.conv1(x))
		x = F.max_pool2d(x, (2, 2))
		x = activation_func(self.conv2(x))
		x = F.max_pool2d(x, 2) # 2 is the same as (2, 2)
		x = x.view(-1, self.num_flat_features(x)) # Flatten the tensor so we can use the FC-layers
		x = activation_func(self.fc1(x))
		x = activation_func(self.fc2(x))
		x = F.softmax(self.fc3(x), dim=1)
		return x

	def num_flat_features(self, x):
		"""Calculates the number of features in data
		e.g. a shape of (80, 5, 5) has num features = 80*5*5."""
		num = 1
		for i in x.size()[1:]:
			num *= i
		return num

	def evaluate(self, testset, activation_func):
		"""Evaluate the net on a DataLoader testset object"""
		with torch.no_grad():
			correct, total = 0, 0
			for data in testset:
				X, y = data
				output = self(X, activation_func=activation_func)
				for idx, out in enumerate(output):
					if torch.argmax(out) == y[idx]:
						correct += 1
					total += 1

		accuracy = correct/total
		return accuracy

	def get_all_preds(self, loader):
		with torch.no_grad():
		    all_preds = torch.tensor([])
		    for batch in loader:
		        images, labels = batch

		        preds = model(images)
		        all_preds = torch.cat(
		            (all_preds, preds)
		            ,dim=0
		        )
    return all_preds