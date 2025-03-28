import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Custom Rotate Filter Function, RotEquivConv2d, ScaleEquivConv2d, MyModel definition...

# --- Custom Rotate Filter Function ---
def rotate_filter(filter, angle, device):
    # Calculate rotation matrix
    theta = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0]
    ], dtype=torch.float).to(device)

    # Get the number of filters and the size of each filter
    N, C, H, W = filter.size()

    # Repeat and reshape theta to match the batch size of the filter tensor
    # The batch size after reshaping the filter is N * C
    theta = theta.repeat(N * C, 1, 1)

    # Adjust the shape of the filter for 4D input (combining batch and channel dimensions)
    reshaped_filter = filter.view(N * C, 1, H, W)

    # Create affine grid
    grid_size = reshaped_filter.size()
    grid = F.affine_grid(theta, grid_size, align_corners=False)

    # Apply grid sampling and reshape back to original
    rotated_filter = F.grid_sample(reshaped_filter, grid, align_corners=False)
    return rotated_filter.view(N, C, H, W)


# --- RotEquivConv2d Definition ---
class RotEquivConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_rotations=8):
        super(RotEquivConv2d, self).__init__()
        self.num_rotations = num_rotations
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        weight = self.conv.weight
        rotated_outputs = []
        for i in range(self.num_rotations):
            angle = 2 * np.pi * i / self.num_rotations
            rotated_weight = rotate_filter(weight, angle, weight.device)
            rotated_output = F.conv2d(x, rotated_weight, padding=self.conv.padding)
            rotated_outputs.append(rotated_output.unsqueeze(1))
        output = torch.cat(rotated_outputs, dim=1)
        return output.mean(dim=1)

# --- ScaleEquivConv2d Definition ---
class ScaleEquivConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scales=[1.0, 0.8, 0.6], padding=0):
        super(ScaleEquivConv2d, self).__init__()
        self.scales = scales
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.scaled_filters = self.create_scaled_filters()

    def create_scaled_filters(self):
        with torch.no_grad():
            weight = self.conv.weight
            original_size = weight.size()[2:]  # Spatial dimensions (H, W)
            scaled_filters = []

            for scale in self.scales:
                scaled_filter = F.interpolate(weight, scale_factor=scale, mode='bilinear', align_corners=False)
                resized_filter = F.interpolate(scaled_filter, size=original_size, mode='bilinear', align_corners=False)
                scaled_filters.append(resized_filter)

            # Move scaled filters to the same device as the weight
            return torch.cat(scaled_filters, dim=0).to(weight.device)

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        outputs = []
        split_size = self.out_channels

        for i in range(len(self.scales)):
            # Ensure scaled filters are on the same device as the input tensor
            scaled_filters_on_device = self.scaled_filters[i*split_size:(i+1)*split_size, :, :, :].to(device)
            output = F.conv2d(x, scaled_filters_on_device, padding=self.conv.padding)
            outputs.append(output)

        return sum(outputs)
        
        
        
# --- MyModel Definition ---

class MyModel(nn.Module):
    def __init__(self, num_classes=10, device='cuda'):
        super(MyModel, self).__init__()
        self.device = device

        # Standard Convolutional Layer with increased filters
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=2).to(device)  # Increased filters and larger kernel
        self.bn1 = nn.BatchNorm2d(128).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Rotation Equivariant Convolutional Layer with increased filters
        self.rot_equiv_conv1 = RotEquivConv2d(128, 256, kernel_size=3).to(device)  # Increased filters
        self.bn2 = nn.BatchNorm2d(256).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Scale Equivariant Convolutional Layer with increased filters
        self.scale_equiv_conv = ScaleEquivConv2d(256, 512, kernel_size=3, padding=1).to(device)  # Increased filters
        self.bn3 = nn.BatchNorm2d(512).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional Rotation Equivariant Convolutional Layer with increased filters
        self.rot_equiv_conv2 = RotEquivConv2d(512, 1024, kernel_size=3).to(device)  # Increased filters
        self.bn4 = nn.BatchNorm2d(1024).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc_input_features = 1024 * 2 * 2  # Adjust as per the final feature map size
        self.fc = nn.Linear(self.fc_input_features, num_classes).to(device)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.rot_equiv_conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.scale_equiv_conv(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.rot_equiv_conv2(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class StandardCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StandardCNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        # Fully Connected Layer
        self.fc1 = nn.Linear(512 * 2 * 2, num_classes)  # Assuming input images are 32x32 pixels

    def forward(self, x):
        # Apply conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))

        # Apply conv2 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))

        # Apply conv3 -> relu -> pool
        x = self.pool(F.relu(self.conv3(x)))

        # Apply conv4 -> relu -> pool
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, 512 * 2 * 2)  # Flatten the output

        # Apply fully connected layer
        x = self.fc1(x)

        return x

import os
def train_and_test(model, model_name, trainloader,testloader, criterion, optimizer, epochs, device):
    
    train_loss_over_epochs=[]
    test_accuracy_over_epochs=[]
    # checkpoint_path=f'checkpoints/{model_name}_checkpoint.pth'
    # checkpoint_path=f'{model_name}_model.pth'

    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # start_epoch = checkpoint['epoch']+1
    #     # train_loss_over_epochs = checkpoint['train_loss']
    #     # test_accuracy_over_epochs = checkpoint['test_accuracy']
    #     # log_model_parameters(model, "Loaded Model")
    #     # log_optimizer_state(optimizer, "Loaded Optimizer")
    #     print(f"Checkpoint found! Resuming training from epoch")
    # else:
    #     # start_epoch = 0
    #     print(f"No checkpoint found for {model_name}. Starting training from scratch.")
    
    # with open(f'{model_name}performance.txt', 'a') as file:
    #     file.write('Epoch,Train Loss,Test Accuracy\n')
    print(model_name," training started")
    test_acc=0
    for epoch in range(epochs):
        # if test_acc>81.5:
        #     break
        # model.train()
        # running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        #     inputs, labels = data
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
        # train_loss=running_loss / len(trainloader)
        # train_loss_over_epochs.append(train_loss)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc=100 * correct / total
        test_accuracy_over_epochs.append(test_acc)
        # Save model checkpoint
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_loss': train_loss_over_epochs,
        #     'test_accuracy': test_accuracy_over_epochs
        # }, checkpoint_path)
        # print(f"Epoch{epoch} Loss: {train_loss}   Accuracy: {test_acc}")
        print(f"Epoch{epoch}  Accuracy: {test_acc}")

        # with open(f'{model_name}performance.txt', 'a') as file:
        #     file.write(f'{epoch+1}, {train_loss}, {test_acc}\n')
    
    # torch.save(model,f'{model_name}_model_whole.pth')
    print('Finished Training')
    # return model


# Function to remove pixels
def remove_pixels(image, percentage=0.01):
  flattened_indices = list(range(image.nelement()))
  num_pixels_to_remove = int(len(flattened_indices) * percentage)
  pixels_to_remove = random.sample(flattened_indices, num_pixels_to_remove)

  # Create a copy of the image to avoid modifying the original
  modified_image = image.clone()

  flat_image = modified_image.flatten()
  flat_image[pixels_to_remove] = 0.0  # Set removed pixels to zero in the modified image

  return flat_image.view_as(image), len(flattened_indices)

# Function to evaluate the model
def evaluate_model(model, image, original_label):
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item() == original_label



# Define a function to display images with labels
def show_image(orgin_image, std_image, com_image,fig_name,labels, size=(6, 6)):
    """
    Displays three images horizontally in a single plot.
  
    Args:
      image1: A NumPy array representing the first image.
      image2: A NumPy array representing the second image.
      image3: A NumPy array representing the third image.
      size: A tuple representing the figure size (width, height).
    """
    orgin_image,std_image,com_image= orgin_image.to('cpu'),std_image.to('cpu'),com_image.to('cpu')
    img1 = orgin_image[0].numpy().transpose((1, 2, 0))
    img2 = std_image[0].numpy().transpose((1, 2, 0))
    img3 = com_image[0].numpy().transpose((1, 2, 0))
    # Create a new figure with the specified size
    fig, axes = plt.subplots(1, 3, figsize=size)
  
    
    # Display each image on a separate subplot
    axes[0].imshow(img1)
    axes[0].set_title(labels[0])
    axes[0].axis('off')  # Hide axis ticks and labels
  
    axes[1].imshow(img2)
    axes[1].set_title(labels[1])
    axes[1].axis('off')
  
    axes[2].imshow(img3)
    axes[2].set_title(labels[2])
    axes[2].axis('off')
  
    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    
    plt.savefig(f'max_inv_pert_data/{fig_name}.png', dpi=300)
    #plt.show()

# Example usage:
# Assuming 'images' is a tensor containing 3 images and 'labels' is a list of corresponding labels
# show_images_with_labels(images, labels)
    
delta_pixel_removal_rate=0.01
with open(f'max_inv_pert_data/max_inv{delta_pixel_removal_rate}.csv', 'a') as file:
    file.write('combined_removal_rate,standard_removal_rate\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.manual_seed_all(111)

# Dataset and DataLoader setup
transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
# transform_train = transforms.Compose([transforms.ToT 

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# Create and train the models
combined_model = torch.load('../models/custom_model_whole.pth')
standard_model=torch.load('../models/std_model_whole.pth') 
combined_model.to(device)
standard_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_custom = torch.optim.Adam(combined_model.parameters(), lr=0.001)
optimizer_standard = torch.optim.Adam(standard_model.parameters(), lr=0.001)
num_epochs=2

# train_and_test(combined_model,'custom' ,trainloader,testloader_orig, criterion, optimizer_custom,num_epochs, device)
# standard_model.load_state_dict(torch.load('std_model.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
# train_and_test(standard_model, 'std',trainloader,testloader_orig, criterion, optimizer_standard,num_epochs, device)

# combined_model.load_state_dict(torch.load('model_combined200.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
# standard_model.load_state_dict(torch.load('model_standard200.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))

# Load your pre-trained model
#combined_model = torch.load('model_combined200.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu')
#standard_model = torch.load('model_standard200.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the model is in evaluation mode
combined_model.eval()
standard_model.eval()

# Load CIFAR10 test data
# transform = transforms.Compose([transforms.ToTensor()])
# testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# comb_model_tolerances=[]
# std_model_tolerances=[]




# Test for maximum pixel removal percentage that does not change output

for i,(images, labels) in enumerate(testloader):
    
#    if i in [0,1,2]:
#        continue
    images,labels=images.to(device),labels.to(device)    
    original_label = labels.item()
    pixel_removal_rate = 0.0
    comb_model_completed=False
    std_model_completed=False

    while not comb_model_completed or not std_model_completed:
        modified_image, num_pixels_to_remove = remove_pixels(images, pixel_removal_rate)
        if not comb_model_completed:
            if not evaluate_model(combined_model, modified_image, original_label) or pixel_removal_rate>0.99:
                # result=f"Combined Model output changes at pixel removal percentage: {pixel_removal_percentage} with number of pixels removed: {num_pixels_to_remove}"
                # print(result)
                # Appending the maximum invatiant pixel removal_rate
                comb_max_inv_pert_rate=round(pixel_removal_rate-delta_pixel_removal_rate,3)
                comb_model_completed=True
                comb_images=modified_image
                

        if not std_model_completed:    
            if not evaluate_model(standard_model, modified_image, original_label) or pixel_removal_rate>0.99:
                # result=f"Standard Model output changes at pixel removal percentage: {pixel_removal_rate} with number of pixels removed: {num_pixels_to_remove}"
                # print(result)
                # Appending the maximum invatiant pixel removal_rate
                std_max_inv_pert_rate=round(pixel_removal_rate-delta_pixel_removal_rate,3)
                std_model_completed=True
                std_images=modified_image
                
                
        pixel_removal_rate += delta_pixel_removal_rate  # Incrementally increase the percentage

    with open(f'max_inv_pert_data/max_inv{delta_pixel_removal_rate}.csv', 'a') as file:
        file.write(f'{comb_max_inv_pert_rate},{std_max_inv_pert_rate}\n')
    if i<=5:
        labels_to_plot=["OrigImg", f'StdModel({std_max_inv_pert_rate})',f'CustomModel({comb_max_inv_pert_rate})']
        show_image(images,std_images,comb_images,f'Img{i}',labels_to_plot) 
        print(i,'th image tested.')
    if i%200==0:
        print(i," th image tested")

# comb_model_tolerances=np.array(comb_model_tolerances)
# std_model_tolerances=np.array(std_model_tolerances)
# avg_comb_model_tolerances=np.mean(comb_model_tolerances)
# avg_std_model_tolerances=np.mean(std_model_tolerances)
# result_comb=f"Combined Model output changes at average pixel removal rate: {avg_comb_model_tolerances} \n"
# result_std=f"Standard Model output changes at average pixel removal rate: {avg_std_model_tolerances}\n"
# with open(f'max_inv_pert_data/avg_max_inv_pert{delta_pixel_removal_rate}.txt', 'a') as file:
#     file.write(result_comb)
#     file.write(result_std)


