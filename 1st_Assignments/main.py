import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from net import *

from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

import pickle

#### TODOLIST (we don't have to do all of these):
## 1) Using different optimizers such as SGD, SGD with momentum, Adam, RMSProp, etc.
## 2) Using different activation functions such as ReLU, ELU, Leaky ReLU, PReLU, SoftPlus, Sigmoid, etc.
# Comparison of Deep architectures such as AlexNet, VGG, Inception V-3, ResNet, etc.
# Using dropout, batch normalization, weight decay, etc.
# Using pre-trained networks, data augmentation

torch.manual_seed(0) # Reproduction purposes

# Download dataset if needed and convert it to a DataLoader mini-batch generator object (train / test)
train = datasets.CIFAR100("Datasets", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.CIFAR100("Datasets", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True) 	# Convert to Mini-batch
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)	# Convert to Mini-batch

with open("./Datasets/cifar-100-python/meta", 'rb') as fo:
	classes_dict = pickle.load(fo, encoding='bytes')
classes = classes_dict[b'fine_label_names']
for i,item in enumerate(classes):
	item = item.decode("utf-8")
	classes[i] = item

# If CUDA is available, we will use it (GPU is much faster than CPU)
device = torch.device("cpu")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	device = torch.device("cuda")

## 2) Use different activation functions here. 
# https://pytorch.org/docs/stable/nn.functional.html
net = Net()
with open("output.txt", "a") as output:
	output.write("\n")
	output.write(f"The following tests were done with net: {net}.\n")

lr = 0.01
EPOCHS = 10
optimizers = [optim.Adam, optim.RMSprop, optim.SGD, optim.SGD]
activation_funcs = [F.relu, torch.tanh, F.hardtanh, F.leaky_relu, torch.sigmoid]
for activation_func in activation_funcs:
	## 1) Use different optimizers here
	for i, opt in enumerate(optimizers):
		net = Net()
		
		if i < len(optimizers) - 1:
			optimizer = opt(net.parameters(), lr=lr)
			print(optimizer)
		else:
			optimizer = opt(net.parameters(), lr=lr, momentum=0.9)
		
		for epoch in range(EPOCHS):
			print(net.conv1.weight[0])
			for data in trainset:
				X, y = data 					# data is a batch
				optimizer.zero_grad()					# Reset the gradient to zero
				output = net(X, activation_func=activation_func)			# Feed inputs to the net, get output, second parameter: activation func
				loss = F.nll_loss(output, y)	# Negative log-likelihood loss
				loss.backward()					# Backprop
				optimizer.step()
			print(f"Epoch: {epoch + 1}/{EPOCHS} .... Loss: {loss:.4f}")
			print(net.conv1.weight[0])

		test_acc = net.evaluate(testset, activation_func=activation_func)

		# Output to std.out and to the output.txt file
		output_string = f"Settings: {activation_func.__name__.rjust(11)}, {optimizer.__class__.__name__.rjust(11)}, Test accuracy: {test_acc:.4f}."
		print(output_string)
		with open("output.txt", "a") as output:
			output.write(output_string+"\n")

		# create confusion matrix
		preds = net.get_all_preds(testset, activation_func=activation_func)
		cm = confusion_matrix(test.targets, preds.argmax(dim=1).numpy())
		plt.figure(figsize=(20, 20))
		plot_confusion_matrix(cm, classes, normalize=True)
		plt.savefig(f"Figures/cm_{activation_func.__name__}_{optimizer.__class__.__name__}.png")
		plt.close()
