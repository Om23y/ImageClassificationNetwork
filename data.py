# Code snippet to load CIFAR-10 dataset using PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
# Transform the data to tensor and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Load testing data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


subset_indices = np.random.choice(len(trainset), 1000, replace=False)
small_trainset = Subset(trainset, subset_indices)

# Create the DataLoader for the small training set
small_trainloader = torch.utils.data.DataLoader(small_trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Output: trainloader, testloader, classes
# Note: The actual data loading won't execute here, but this code is ready to run in a Python environment with PyTorch.
