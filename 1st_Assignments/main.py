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
import numpy as np
import pickle


def perform_epochs(EPOCHS, trainset, testset, optimizer, net, activation_func):
	net.train() # Tells the net to go into training mode
	for epoch in range(EPOCHS):
		for data in trainset:
			X, y = data 					# data is a batch
			optimizer.zero_grad()			# Reset the gradient to zero
			output = net(X, activation_func=activation_func)			# Feed inputs to the net, get output, second parameter: activation func
			loss = F.nll_loss(output, y)	# Negative log-likelihood loss
			loss.backward()					# Backprop
			optimizer.step()
		train_acc = net.evaluate(trainset, activation_func = activation_func)
		print(f"Epoch: {epoch + 1}/{EPOCHS} .... Loss: {loss:.4f}, Train accuracy: {train_acc}")

	test_acc = net.evaluate(testset, activation_func=activation_func)
	return test_acc

def create_confusion_matrix(testset, activation_func, test, classes, optimizer, momentum, prefix=""):
	# create confusion matrix
	preds = net.get_all_preds(testset, activation_func=activation_func)
	cm = confusion_matrix(test.targets, preds.argmax(dim=1).numpy())
	plt.figure(figsize=(8,8))
	plot_confusion_matrix(cm, classes, normalize=True)
	plt.savefig(f"Figures/{prefix}_cm_{activation_func.__name__}_{optimizer.__class__.__name__}_momentum_{momentum}.png")
	plt.close()

torch.manual_seed(0) # Reproduction purposes
# Download dataset if needed and convert it to a DataLoader mini-batch generator object (train / test)
train = datasets.CIFAR10("Datasets", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.CIFAR10("Datasets", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True) 	# Convert to Mini-batch generator
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)	# Convert to Mini-batch generator

# If CUDA is available, we will use it (GPU is much faster than CPU)
device = torch.device("cpu")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	device = torch.device("cuda")



# Get the class labels
with open("./Datasets/cifar-10-batches-py/batches.meta", 'rb') as fo:
	classes_dict = pickle.load(fo, encoding='bytes')

classes = list(classes_dict.values())[1]
for i,item in enumerate(classes):
	item = item.decode("utf-8")
	classes[i] = item

net = Net()
with open("output.txt", "a") as output:
	output.write("\n")
	output.write(f"The following tests were done with net: {net}.\n")

# Tuning parameters
lr = 0.003
EPOCHS = 15
optimizers = [optim.Adam, optim.RMSprop, optim.SGD, optim.SGD]
activation_funcs = [F.relu, torch.tanh, F.hardtanh, F.leaky_relu, torch.sigmoid]

# ###########################################################################################################
# ## PHASE 1
# highest_test_acc = 0.0
# for activation_func in activation_funcs:
# 	## 1) Use different optimizers here
# 	for optimizer_index, opt in enumerate(optimizers):
# 		net = Net()
# 		if optimizer_index != 3:
# 			momentum = 0
# 			optimizer = optimizers[optimizer_index](net.parameters(), lr=lr)
# 		else:
# 			momentum = 0.9
# 			optimizer = optimizers[optimizer_index](net.parameters(), lr=lr, momentum=momentum)
		
# 		test_acc = perform_epochs(EPOCHS, trainset, testset, optimizer, net, activation_func)
# 		if test_acc > highest_test_acc:
# 			highest_test_acc = test_acc
# 			best_combo = (activation_func, optimizer_index)

# 		# Output to std.out and to the output.txt file
# 		output_string = f"Settings: {activation_func.__name__.rjust(11)}, {optimizer.__class__.__name__.rjust(11)}, Test accuracy: {test_acc:.4f}."
# 		print(output_string)
# 		with open("output.txt", "a") as output:
# 			output.write(output_string+"\n")

# 		create_confusion_matrix(testset, activation_func, test, classes, optimizer, momentum, "1st_phase")

# print(f"The best combo of activation function and optimizer is: {best_combo}, "
#  	  f"because they got the highest test accuracy of: {highest_test_acc}")
# # Do stuff with the best combo
# # Start using the same net but with regularization methods
# # Such as dropout, batch normalization and weight decay
# activation_func, optimizer_index = best_combo

activation_func, optimizer_index = torch.tanh, 3
###########################################################################################################
## PHASE 2
highest_test_acc_phase_2 = 0
for weight_decay in np.arange(0.0001, 0.011, 0.0003):
	net = Net()
	if optimizer_index != 3:
		momentum = 0
		optimizer = optimizers[optimizer_index](net.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		momentum = 0.9
		optimizer = optimizers[optimizer_index](net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

	test_acc = perform_epochs(EPOCHS, trainset, testset, optimizer, net, activation_func)
	if test_acc > highest_test_acc_phase_2:
		highest_test_acc_phase_2 = test_acc
		best_weight_decay = weight_decay
	print(f"Test acc: {test_acc}")

	create_confusion_matrix(testset, activation_func, test, classes, optimizer, momentum, f"2nd_phase_wd_{weight_decay}")

print(f"The best weight decay was {best_weight_decay}, which yielded a test acc of: {highest_test_acc_phase_2}.")
if highest_test_acc_phase_2 > highest_test_acc:
	print(f"This is an improvement over the previous highest test acc: {highest_test_acc}.")
else:
	print(f"This is NOT an improvement over the previous highest test acc: {highest_test_acc}.")
###########################################################################################################
## PHASE 3
highest_test_acc_phase_3 = 0
for i in [0, 1]:
	net = Net(dropout=True)
	if optimizer_index != 3:
		momentum = 0
		optimizer = optimizers[optimizer_index](net.parameters(), lr=lr, weight_decay=i*best_weight_decay)
	else:
		momentum = 0.9
		optimizer = optimizers[optimizer_index](net.parameters(), lr=lr, weight_decay=i*best_weight_decay, momentum=momentum)

	test_acc = perform_epochs(EPOCHS, trainset, testset, optimizer, net, activation_func)
	if i == 0:
		test_acc_without_weight_decay = test_acc
		create_confusion_matrix(testset, activation_func, test, classes, optimizer, momentum, "3rd_phase")
	else:
		test_acc_with_weight_decay = test_acc
		create_confusion_matrix(testset, activation_func, test, classes, optimizer, momentum, "3rd_phase_wd")
	print(f"Test acc: {test_acc}")

try:
	print(f"Highest test acc after phase 1: {highest_test_acc}")
except NameError:
	# This just means that phase 1 was commented out
	print("Phase 1 was skipped")

print(f"Highest test acc after phase 2: {highest_test_acc_phase_2}")
print(f"Test acc after phase 3 without weight decay: {test_acc_without_weight_decay}")
print(f"Test acc after phase 3 with weight decay: {test_acc_with_weight_decay}")
