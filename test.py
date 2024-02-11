import torch
from data import testloader  # Assuming you have a similar setup for your test data
from CNN import train

# Assuming `Net` class and `train` function are defined as before

def test(net):
    # Set the model to evaluation mode
    net.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for testing
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    net = train()  # Make sure your train function returns the trained model
    test(net)
