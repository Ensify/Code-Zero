import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchsummary import summary

from model import IntensityNet


# Define your dataset and dataloader
# Assuming you have a dataset with clear and hazy images in separate folders
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='data', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for i,v in train_loader:
    print(i.shape)
    print(v)
    break

# Instantiate the model, define the loss function and optimizer
model = IntensityNet(torch.device("cuda:0"))
model.load_state_dict(torch.load('haze_detection_model.pth'))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary to check the architecture
summary(model, (3, 256, 256))

# Train the model
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'haze_detection_model2.pth')