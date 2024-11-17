import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, patch_size, input_dim, output_dim, k=0.5):
        super(Generator, self).__init__()
        # print(f'**************** in generator: patch size is {patch_size}')
        self.k = k
        self.patch_size = patch_size

        layers = self.build_layers(input_dim, output_dim, patch_size)
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self.weights_init)
        

    def build_layers(self, input_dim, output_dim, patch_size):
        layers = []
        
        # First layer: always maps from (1x1) to (4x4)
        layers.append(nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.ReLU(inplace=False))

        if patch_size == '48':
            # Adjusted layers for (4x4) to (48x48)
            layers.append(self.make_layer(1024, 512, kernel_size=4, stride=2, padding=1))  # (4x4 -> 8x8)
            layers.append(self.make_layer(512, 256, kernel_size=3, stride=2, padding=1))   # (8x8 -> 16x16)
            layers.append(self.make_layer(256, 128, kernel_size=3, stride=2, padding=2))   # (16x16 -> 32x32)
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=2, stride=2, padding=3))  # (32x32 -> 48x48)
            
        elif patch_size == '64':
            # Adjusted layers for (4x4) to (64x64)
            layers.append(self.make_layer(1024, 512, kernel_size=4, stride=2, padding=1))  # (4x4 -> 8x8)
            layers.append(self.make_layer(512, 256, kernel_size=4, stride=2, padding=1))   # (8x8 -> 16x16)
            layers.append(self.make_layer(256, 128, kernel_size=4, stride=2, padding=1))   # (16x16 -> 32x32)
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1))  # (32x32 -> 64x64)
            
        elif patch_size == '80':
            # (4x4) to (80x80)
            layers.append(self.make_layer(1024, 512, kernel_size=6, stride=2, padding=1))  # (4x4 -> 8x8)
            layers.append(self.make_layer(512, 256, kernel_size=5, stride=2, padding=1))   # (8x8 -> 16x16)
            layers.append(self.make_layer(256, 128, kernel_size=4, stride=2, padding=2))   # (16x16 -> 32x32)
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1))  # (128x128 -> 80x80)
        else:
            raise ValueError("Unsupported patch size! Choose between 48, 64, or 80.")

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
    
    def threshold(self, x):
        x_tanh = torch.tanh(x)
        return self.k * x_tanh + self.k

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.constant_(m.bias.data, 0)

    def reset_weights(self):
        """Reset the weights of the model."""
        print("Resetting weights...")
        self.apply(self.weights_init)
