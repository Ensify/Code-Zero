import torch
import torch.nn as nn


class IntensityNet(nn.Module):
    def __init__(self,device):
        super(IntensityNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=2, padding="valid",device=device)
        self.conv2 = nn.Conv2d(10, 5, kernel_size=3, stride=1, padding="valid",device=device)
        # Fully connected layers
        self.fc1 = nn.Linear(76880, 15,device=device)
        self.fc2 = nn.Linear(15, 1,device=device)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, 3, height, width)

        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x
