import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from e2cnn import gspaces, nn as enn

output_dir = 'max_inv_pert_data/'
os.makedirs(output_dir, exist_ok=True)

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



# Assuming your models are already trained and saved
# Load your pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your pre-trained models
combined_model = BaselineCNNWithParallelScaleAndRotEquivariance().to(device)
standard_model = ExtendedCNN().to(device)

# Load saved model state_dict
combined_model.load_state_dict(torch.load('/content/parallelGCNNrotscale10layer_cifar10.pth', map_location=device), strict=False)
standard_model.load_state_dict(torch.load('/content/baselineCNN10layer_cifar10.pth', map_location=device), strict=False)

# Set the models to evaluation mode
combined_model.eval()
standard_model.eval()

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

# Function to display images with labels
def show_image(orig_image, std_image, com_image, fig_name, labels, size=(6, 6)):
    orig_image, std_image, com_image = orig_image.to('cpu'), std_image.to('cpu'), com_image.to('cpu')
    img1 = orig_image[0].numpy().transpose((1, 2, 0))
    img2 = std_image[0].numpy().transpose((1, 2, 0))
    img3 = com_image[0].numpy().transpose((1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=size)

    axes[0].imshow(img1)
    axes[0].set_title(labels[0])
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title(labels[1])
    axes[1].axis('off')

    axes[2].imshow(img3)
    axes[2].set_title(labels[2])
    axes[2].axis('off')

    # plt.tight_layout()
    plt.savefig(f'./max_inv_pert_data/{fig_name}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'./max_inv_pert_data/{fig_name}.pdf', bbox_inches='tight', dpi=300, format='pdf')

# CIFAR-10 Dataset
transform_train = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # No normalization (no change to pixel values)
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# # Function to show a sample image
# def show_sample_image(index):
#     image, label = testset[index]  # Get the image and label at the given index
#     image = image.permute(1, 2, 0)  # Reorder dimensions to (height, width, channels) for display
#     plt.imshow(image)
#     plt.title(f"Label: {label}")
#     plt.axis('off')  # Hide axis
#     plt.show()

# # Show sample images at indices 10 and 12
# show_sample_image(10)
# show_sample_image(12)

# Delta for pixel removal
delta_pixel_removal_rate = 0.05
with open(f'max_inv_pert_data/max_inv{delta_pixel_removal_rate}.csv', 'a') as file:
    file.write('combined_removal_rate,standard_removal_rate\n')

# Evaluation loop
# ind = [10, 12, 7617]
# for i, (images, labels) in enumerate(testloader):
#     images, labels = images.to(device), labels.to(device)
#     original_label = labels.item()
#     pixel_removal_rate = 0.1
#     comb_model_completed = False
#     std_model_completed = False

#     while not comb_model_completed or not std_model_completed:
#         modified_image, num_pixels_to_remove = remove_pixels(images, pixel_removal_rate)

#         # Evaluate Combined Model
#         if not comb_model_completed:
#             if not evaluate_model(combined_model, modified_image, original_label) or pixel_removal_rate > 0.99:
#                 comb_max_inv_pert_rate = round(pixel_removal_rate - delta_pixel_removal_rate, 3)
#                 comb_model_completed = True
#                 comb_images = modified_image

#         # Evaluate Standard Model
#         if not std_model_completed:
#             if not evaluate_model(standard_model, modified_image, original_label) or pixel_removal_rate > 0.99:
#                 std_max_inv_pert_rate = round(pixel_removal_rate - delta_pixel_removal_rate, 3)
#                 std_model_completed = True
#                 std_images = modified_image

#         pixel_removal_rate += delta_pixel_removal_rate

#     with open(f'max_inv_pert_data/max_inv{delta_pixel_removal_rate}.csv', 'a') as file:
#         file.write(f'{comb_max_inv_pert_rate},{std_max_inv_pert_rate}\n')

#     if i in [19, 24, 22, 29, 32]:
#         labels_to_plot = ["OrigImg", f'StdModel({std_max_inv_pert_rate})', f'CustomModel({comb_max_inv_pert_rate})']
#         # labels_to_plot = ["OrigImg", f'ExtendedCNN', f'CNNScaleAndRot']
#         show_image(images, std_images, comb_images, f'Img{i}', labels_to_plot)
#         print(f'{i}th image tested.')

#     if i > 40:
#       break

#     if i % 200 == 0:
#         print(f'{i}th image tested')

for i, (images, labels) in enumerate(testloader):
    images, labels = images.to(device), labels.to(device)
    original_label = labels.item()
    pixel_removal_rate = 0.1
    comb_model_completed = False
    std_model_completed = False

    while not comb_model_completed or not std_model_completed:
        # Create a copy of the original image for each model to avoid modifying the original image
        modified_image_comb = images.clone().detach()  # For combined model
        modified_image_std = images.clone().detach()  # For standard model

        # Apply pixel removal to both models' copies
        modified_image_comb, num_pixels_to_remove_comb = remove_pixels(modified_image_comb, pixel_removal_rate)
        modified_image_std, num_pixels_to_remove_std = remove_pixels(modified_image_std, pixel_removal_rate)

        # Evaluate Combined Model
        if not comb_model_completed:
            if not evaluate_model(combined_model, modified_image_comb, original_label) or pixel_removal_rate > 0.99:
                comb_max_inv_pert_rate = round(pixel_removal_rate - delta_pixel_removal_rate, 3)
                comb_model_completed = True
                comb_images = modified_image_comb

        # Evaluate Standard Model
        if not std_model_completed:
            if not evaluate_model(standard_model, modified_image_std, original_label) or pixel_removal_rate > 0.99:
                std_max_inv_pert_rate = round(pixel_removal_rate - delta_pixel_removal_rate, 3)
                std_model_completed = True
                std_images = modified_image_std

        pixel_removal_rate += delta_pixel_removal_rate

    # Log the results for both models
    with open(f'max_inv_pert_data/max_inv{delta_pixel_removal_rate}.csv', 'a') as file:
        file.write(f'{comb_max_inv_pert_rate},{std_max_inv_pert_rate}\n')

    if i in [19, 22, 29, 30]:
        labels_to_plot = ["Original Image", f'Standard CNN({std_max_inv_pert_rate})', f'Group CNN({comb_max_inv_pert_rate})']
        show_image(images, std_images, comb_images, f'Img{i}', labels_to_plot)
        print(f'{i}th image tested.')

    if i > 50:
        break

    if i % 200 == 0:
        print(f'{i}th image tested')