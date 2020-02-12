import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from net import *

#### TODOLIST (we don't have to do all of these):
## 1) Using different optimizers such as SGD, SGD with momentum, Adam, RMSProp, Nadam, etc.
## 2) Using different activation functions such as ReLU, ELU, Leaky ReLU, PReLU, SoftPlus, Sigmoid, etc.
# Comparison of Deep architectures such as AlexNet, VGG, Inception V-3, ResNet, etc.
# Using dropout, batch normalization, weight decay, etc.
# Using pre-trained networks, data augmentation

# Download dataset if needed and convert it to a DataLoader mini-batch generator object (train / test)
train = datasets.CIFAR100("Datasets", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.CIFAR100("Datasets", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True) 	# Convert to Mini-batch
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)	# Convert to Mini-batch

# IF CUDA is available, we will use it (GPU is much faster than CPU)
device = torch.device("cpu")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	device = torch.device("cuda")

# Net is defined in net.py
net = Net()

## 1) Use different optimizers here
optimizer = optim.Adam(net.parameters(), lr=0.001)

## 2) Use different activation functions here. 
# https://pytorch.org/docs/stable/nn.functional.html
activation_funcs = [F.relu, F.tanh, F.hardtanh, F.leaky_relu, F.sigmoid]
for activation_func in activation_funcs:
	EPOCHS = 10
	for epoch in range(EPOCHS):
		print(f"Epoch: {epoch + 1}/{EPOCHS}")
		for data in trainset:
			X, y = data 					# data is a batch
			net.zero_grad()					# Reset the gradient to zero
			output = net(X, activation_func=activation_func)			# Feed inputs to the net, get output, second parameter: activation func
			loss = F.nll_loss(output, y)	# Negative log-likelihood loss
			loss.backward()					# Backprop
			optimizer.step()
		print(f"Loss: {loss}")
	test_acc = net.evaluate(testset, activation_func=activation_func)
	print(f"Test accuracy: {test_acc}")
