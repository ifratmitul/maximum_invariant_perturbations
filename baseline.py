import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Extended CNN Architecture with 10 Layers
class ExtendedCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(ExtendedCNN, self).__init__()

        # Convolutional Layers with Batch Normalization
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)

        # ReLU Activation Function
        self.relu = nn.ReLU()

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))

        # Global Max Pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)

        # Fully Connected Layer
        x = self.fc(x)
        return x

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize Model, Loss Function, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExtendedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(200):  # 200 epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/200], Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'baselineCNN10layer_cifar10.pth')
print("Model saved to baselineCNN10layer_cifar10.pth")

# Testing Loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
