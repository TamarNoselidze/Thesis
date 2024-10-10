import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, k=0.5):
        super(Generator, self).__init__()
        
        self.k = k
        
        self.layer1 = nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=False)  
        
        self.layer2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=False)  
        
        self.layer3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=False)  
        
        self.layer4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=False)  
        
        self.layer5 = nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1)
        
        # Initialize weights
        self.apply(self.weights_init)
        
    def forward(self, x):
        x = self.layer1(x)
        # print(f"After layer 1 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        # print(f"After layer 2 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        # print(f"After layer 3 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.layer4(x)
        # print(f"After layer 4 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.layer5(x)
        # print(f"After layer 5 (ConvTranspose2d): {x.shape} | min: {x.min().item()} | max: {x.max().item()}")

        x = self.threshold(x)
        # print(f"After thresholding: {x.shape} | min: {x.min().item()} | max: {x.max().item()}")

        return x
    
    def threshold(self, x):
        x_tanh = torch.tanh(x)
        return self.k * x_tanh + self.k

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.constant_(m.bias.data, 0)

