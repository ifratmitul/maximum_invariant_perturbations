import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from e2cnn import gspaces, nn as enn

class BaselineCNNWithParallelScaleAndRotEquivariance(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(BaselineCNNWithParallelScaleAndRotEquivariance, self).__init__()
        
        # Define rotational symmetry group
        self.rot_gspace = gspaces.Rot2dOnR2(N=8)  # Rotational symmetry with 8 discrete angles

        # Define field types for equivariant convolution
        self.feat_type_in_rot = enn.FieldType(self.rot_gspace, input_channels * [self.rot_gspace.trivial_repr])
        self.feat_type_out_rot = enn.FieldType(self.rot_gspace, 16 * [self.rot_gspace.regular_repr])

        # Standard convolutional layer
        self.conv_standard = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )

        # Rotational-equivariant convolutional layer
        self.conv_equivariant_rot = enn.R2Conv(self.feat_type_in_rot, self.feat_type_out_rot, kernel_size=3, padding=1)
        self.relu_equivariant_rot = enn.ReLU(self.feat_type_out_rot)

        # Simulated scale-equivariant convolutions using separate scaled inputs
        self.conv_scale_0_5 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv_scale_1_0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv_scale_2_0 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            nn.ReLU()
        )

        # Fusion layer to combine outputs of standard, rotational-equivariant, and scale-equivariant convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1),  # Adjust input channels to 192
            nn.BatchNorm2d(32),  # Add Batch Normalization
            nn.ReLU()
        )

        # Additional convolutional layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Add Batch Normalization
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply standard convolution
        x_standard = self.conv_standard(x)

        # Apply rotational-equivariant convolution
        x_rot = enn.GeometricTensor(x, self.feat_type_in_rot)  # Wrap input for rotational-equivariant layer
        x_rot = self.conv_equivariant_rot(x_rot)
        x_rot = self.relu_equivariant_rot(x_rot)
        x_rot = x_rot.tensor  # Convert back to PyTorch tensor

        # Simulated scale-equivariant convolutions
        x_scale_0_5 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_scale_0_5 = self.conv_scale_0_5(x_scale_0_5)
        x_scale_0_5 = F.interpolate(x_scale_0_5, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_scale_1_0 = self.conv_scale_1_0(x)

        x_scale_2_0 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x_scale_2_0 = self.conv_scale_2_0(x_scale_2_0)
        x_scale_2_0 = F.interpolate(x_scale_2_0, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate outputs from standard, rotational-equivariant, and scale-equivariant convolutions
        x_combined = torch.cat((x_standard, x_rot, x_scale_0_5, x_scale_1_0, x_scale_2_0), dim=1)

        # Apply fusion layer
        x = self.fusion(x_combined)

        # Pass through additional layers
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        # Global Max Pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])  # Pool to 1x1
        x = torch.flatten(x, 1)  # Flatten to feed into the FC layer

        # Fully connected layer
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

# Define device, model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNNWithParallelScaleAndRotEquivariance().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(200):  # Train for 200 epochs
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
torch.save(model.state_dict(), "parallelGCNNrotscale10layer_cifar10.pth")
print("Model saved to 'parallelGCNNrotscale10layer_cifar10.pth'.")

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
