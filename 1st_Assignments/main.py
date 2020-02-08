import torch
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets
from net import *

## TODOLIST (we don't have to do all of these):
# Comparison of Deep architectures such as AlexNet, VGG, Inception V-3, ResNet, etc.
# Using dropout, batch normalization, weight decay, etc.
# Using different activation functions such as ReLU, ELU, Leaky ReLU, PReLU, SoftPlus, Sigmoid, etc.
# Using pre-trained networks, data augmentation
# Using different optimizers such as SGD, SGD with momentum, Adam, RMSProp, Nadam, etc.

print(f"CUDA is available: {torch.cuda.is_available()}")

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True) 	# Mini batch
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

device = torch.device("cpu")
if torch.cuda.is_available():
	device = torch.device("cuda")


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 4
for epoch in range(EPOCHS):
	print(f"Epoch: {epoch}")
	for data in trainset:
		X, y = data 					# data is a batch
		net.zero_grad()					# Reset the gradient to zero
		output = net(X)					# Feed inputs to the net, get output
		loss = F.nll_loss(output, y)	# Negative log-likelihood loss
		loss.backward()					# Backprop
		optimizer.step()
	print(f"Loss: {loss}")
	test_acc = net.eval(testset)
	print(f"Test accuracy: {test_acc}")
