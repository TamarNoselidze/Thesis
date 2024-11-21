import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, patch_size=16, num_patches=16, num_channels=3):
        super(Generator, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.noise_dim = 100
        self.output_channels=3
        
        # # Define a simple ConvTranspose2d network to generate a small patch
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(100, 64, kernel_size=4, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_channels),
            nn.Tanh()
        )

        # # 1. First Convolutional Layer: Projects random noise into a 4D tensor
        # self.conv1 = nn.ConvTranspose2d(self.noise_dim, 256, kernel_size=4, stride=1, padding=0) # Output: (256, 4, 4)
        # self.bn1 = nn.BatchNorm2d(256)
        
        # # 2. Second Convolutional Layer: Upsamples and refines the feature maps
        # self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # Output: (128, 8, 8)
        # self.bn2 = nn.BatchNorm2d(128)
        
        # # 3. Third Convolutional Layer: Further upsampling and refining
        # self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # Output: (64, 16, 16)
        # self.bn3 = nn.BatchNorm2d(64)
        
        # # 4. Fourth Convolutional Layer: Further upsampling
        # self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # Output: (32, 32, 32)
        # self.bn4 = nn.BatchNorm2d(32)
        
        # # 5. Fifth Convolutional Layer: Final upsampling and patch generation
        # self.conv5 = nn.ConvTranspose2d(32, num_patches*self.output_channels, kernel_size=4, stride=2, padding=1) # Output: (3, 64, 64)
        
        # # Thresholding layer (using tanh to restrict values between -1 and 1)
        # self.threshold = nn.Tanh()

    def forward(self, noise):
        # Generate 16 patches from noise
        patches = []
        for _ in range(self.num_patches):
            patch_noise = noise[:, :, :, :]  # Assuming noise shape is [batch_size, 100, 1, 1]
            patch = self.conv_layers(patch_noise)
            patch = torch.tanh(patch)  # Ensure patch values are in range [-1, 1]
            print(f"Generated patch size: {patch.shape}")  
            patches.append(patch)
        
        # Stack the patches to return a tensor of shape [batch_size, num_patches, num_channels, patch_size, patch_size]
        patches = torch.stack(patches, dim=1)

        return patches

    # def forward(self, x):
    #     # Pass through the layers with BatchNorm and ReLU activations
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = F.relu(self.bn4(self.conv4(x)))
        
    #     # Last convolutional layer followed by threshold (tanh)
    #     x = self.conv5(x)
    #     print(f"^^^^^^^^^^^^^^^^^^^ Output shape before reshaping: {x.shape}")
    #     x = self.threshold(x)  # Ensuring output values are limited to a specific range (-1 to 1)
    #     # Reshape to: (batch_size, num_patches, channels, patch_size, patch_size)
    #     batch_size = x.shape[0]
    #     x = x.view(batch_size, self.num_patches, self.output_channels, self.patch_size, self.patch_size)
        
    #     return x
