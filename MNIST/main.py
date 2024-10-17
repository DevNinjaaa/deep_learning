import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Loading the MNIST dataset
train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

# DataLoader for batching and shuffling
loader = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0),
}

# CNN model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convl1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)  # Smaller kernel size
        self.convl2 = nn.Conv2d(10, 20, kernel_size=3, padding=1) # Smaller kernel size
        self.convl3 = nn.Conv2d(20, 30, kernel_size=3, padding=1) # Smaller kernel size
        self.convl3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(30 * 3 * 3, 50)  # Adjusted input size for fully connected layer
        self.fc2 = nn.Linear(50, 10)           # Final output layer

    def forward(self, X):
        X = F.relu(F.max_pool2d(self.convl1(X), 2))  # Convolution 1 with max pooling
        X = F.relu(F.max_pool2d(self.convl2(X), 2))  # Convolution 2 with max pooling
        X = F.relu(F.max_pool2d(self.convl3_drop(self.convl3(X)), 2))  # Convolution 3 with dropout and max pooling
        X = X.view(-1, 30 * 3 * 3)  # Adjust based on output size after conv layers
        X = F.relu(self.fc1(X))      # First fully connected layer
        X = F.dropout(X, training=self.training)  # Dropout for regularization
        X = self.fc2(X)              # Final fully connected layer (logits)

        return X  # Return raw logits


# Set device for model training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, optimizer, and loss function
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader['test'])  # Average over the number of batches
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {100. * correct / len(loader["test"].dataset):.2f}%')

# Training and testing
train(1)
test()

# Model evaluation and prediction on a single image
model.eval()
data, target = test_data[0]
data = data.unsqueeze(0).to(device)  # Adding batch dimension
output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()  # Get predicted label
print(prediction)

# Displaying the image
image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap='gray')
plt.show()
