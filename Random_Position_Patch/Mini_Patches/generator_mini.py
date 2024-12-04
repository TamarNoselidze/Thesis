import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorMini(nn.Module):
    def __init__(self, patch_size=16, num_patches=16, num_channels=3):
        super(GeneratorMini, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.noise_dim = 100
        self.output_channels=3
        
        # # # Define a simple ConvTranspose2d network to generate a small patch
        # self.conv_layers = nn.Sequential(
        #     nn.ConvTranspose2d(100, 64, kernel_size=4, stride=3, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=3, padding=0),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=1, padding=0),
        #     nn.BatchNorm2d(num_channels),
        #     nn.Tanh()
        # )

        layers = self.build_layers(self.noise_dim, self.output_channels)
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self.weights_init)
        


    def build_layers(self, input_dim, output_dim):
        layers = []
        # First layer: always maps from (1x1) to (4x4)
        layers.append(nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=False))

        layers.append(self.make_layer(256, 128, kernel_size=4, stride=2, padding=1))  # (4x4 -> 8x8)
        layers.append(self.make_layer(128, 64, kernel_size=4, stride=2, padding=1))   # (8x8 -> 16x16)
        layers.append(self.make_layer(64, 32, kernel_size=4, stride=2, padding=1))   # (16x16 -> 32x32)
        layers.append(nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2, padding=1))  # (32x32 -> 64x64)

        return layers
    

    def make_layer(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        x = self.model(x)
        x = self.threshold(x)

        return x


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.constant_(m.bias.data, 0)

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
