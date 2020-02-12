import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from net import *

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

# If CUDA is available, we will use it (GPU is much faster than CPU)
device = torch.device("cpu")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	device = torch.device("cuda")

## 2) Use different activation functions here. 
# https://pytorch.org/docs/stable/nn.functional.html
net = Net() # Net always starts with the same weights, so we can see what the influence of each activation func is

with open("output.txt", "a") as output:
	output.write("\n")
	output.write(f"The following tests were done with net: {net}.\n")

activation_funcs = [F.relu, torch.tanh, F.hardtanh, F.leaky_relu, torch.sigmoid]
optimizers = 
for activation_func in activation_funcs:
	## 1) Use different optimizers here
	optimizers = [optim.Adam(net.parameters(), lr=0.003), optim.RMSprop(net.parameters(), lr=0.003), optim.SGD(net.parameters(), lr=0.003),
				  optim.SGD(net.parameters(), lr=0.003, momentum=0.9)]
	for optimizer in optimizers:
		net = Net() # Define the net again
		EPOCHS = 1
		for epoch in range(EPOCHS):
			for data in trainset:
				X, y = data 					# data is a batch
				net.zero_grad()					# Reset the gradient to zero
				output = net(X, activation_func=activation_func)			# Feed inputs to the net, get output, second parameter: activation func
				loss = F.nll_loss(output, y)	# Negative log-likelihood loss
				loss.backward()					# Backprop
				optimizer.step()
			print(f"Epoch: {epoch + 1}/{EPOCHS} .... Loss: {loss:.4f}")
		test_acc = net.evaluate(testset, activation_func=activation_func)

		# Output to std.out and to the output.txt file
		output_string = f"Settings: {activation_func.__name__.rjust(11)}, {optimizer.__class__.__name__.rjust(11)}, Test accuracy: {test_acc:.4f}."
		print(output_string)
		preds = get_all_preds(testset)
		stacked = torch.stack((testset.targets,preds.argmax(dim=1)),dim=1)
		cmt = torch.zeros(100,100, dtype=torch.int64)
		for p in stacked:
    		tl, pl = p.tolist()
    		cmt[tl, pl] = cmt[tl, pl] + 1
    	print(cmt)
		with open("output.txt", "a") as output:
			output.write(output_string+"\n")
