import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, patch_size, input_dim, output_dim, k=0.5):
        super(Generator, self).__init__()
        
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
            # (4x4) -> (8x8) -> (16x16) -> (24x24) -> (48x48)
            layers.append(self.make_layer(1024, 512, 4, 2, 1))  # (4x4 -> 8x8)
            layers.append(self.make_layer(512, 256, 4, 2, 1))   # (8x8 -> 16x16)
            layers.append(self.make_layer(256, 128, 4, 2, 1))   # (16x16 -> 32x32)
            layers.append(self.make_layer(128, output_dim, 5, 2, 1))  # (32x32 -> 48x48, kernel 5 for exact 48x48)
        
        elif patch_size == '64':
            # (4x4) -> (8x8) -> (16x16) -> (32x32) -> (64x64)
            layers.append(self.make_layer(1024, 512, 4, 2, 1))  # (4x4 -> 8x8)
            layers.append(self.make_layer(512, 256, 4, 2, 1))   # (8x8 -> 16x16)
            layers.append(self.make_layer(256, 128, 4, 2, 1))   # (16x16 -> 32x32)
            layers.append(self.make_layer(128, output_dim, 4, 2, 1))  # (32x32 -> 64x64)
        
        elif patch_size == '80':
            # (4x4) -> (8x8) -> (16x16) -> (40x40) -> (80x80)
            layers.append(self.make_layer(1024, 512, 4, 2, 1))  # (4x4 -> 8x8)
            layers.append(self.make_layer(512, 256, 4, 2, 1))   # (8x8 -> 16x16)
            layers.append(self.make_layer(256, 128, 5, 2, 1))   # (16x16 -> 40x40, kernel 5 for exact step)
            layers.append(self.make_layer(128, output_dim, 5, 2, 1))  # (40x40 -> 80x80, kernel 5)

        else:
            raise ValueError("Unsupported patch size! Choose between 48, 64, or 80.")

        return layers


        # self.layer1 = nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.relu1 = nn.ReLU(inplace=False)  

        # self.layer2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.relu2 = nn.ReLU(inplace=False)  
        
        # self.layer3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(inplace=False)  
        
        # self.layer4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.relu4 = nn.ReLU(inplace=False)  
        
        # self.layer5 = nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1)
        
        # # Initialize weights
        # self.apply(self.weights_init)

    def make_layer(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
        
    def forward(self, x):
        # x = self.layer1(x)
        # # print(f"After layer 1 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        # x = self.bn1(x)
        # x = self.relu1(x)

        # x = self.layer2(x)
        # # print(f"After layer 2 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        # x = self.bn2(x)
        # x = self.relu2(x)

        # x = self.layer3(x)
        # # print(f"After layer 3 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        # x = self.bn3(x)
        # x = self.relu3(x)

        # x = self.layer4(x)
        # # print(f"After layer 4 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        # x = self.bn4(x)
        # x = self.relu4(x)

        # x = self.layer5(x)
        # # print(f"After layer 5 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")

        # x = self.threshold(x)
        # # print(f"After thresholding: {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
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

