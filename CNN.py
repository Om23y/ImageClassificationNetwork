import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import small_trainloader, trainloader


printsPerEpoch = 24
used_trainloader = trainloader

# Define the CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)  # kernel size, stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16*5*5 comes from the dimension reduction of the image through the layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes

    def forward(self, x):
        # Apply convolutions, activation functions, and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # Apply fully connected layers with ReLU and produce final output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network
def train():
    net = Net()  # Initialize the network
    
    # Specify the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % printsPerEpoch == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printsPerEpoch:.3f}')
                running_loss = 0.0
    print('Finished Training')
    return net  # Return the trained model

if __name__ == '__main__':
    train()