import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, k=0.5, image_embed_dim=128):
        super(Generator, self).__init__()
        
        self.k = k
        
        # embedding the image into a feature vector
        self.image_embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, image_embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(image_embed_dim, image_embed_dim, kernel_size=3, stride=2, padding=1),  # Additional downsampling
            nn.AdaptiveAvgPool2d((8, 8))  # Pool down to smaller size
        )        

        # fully connected 
        self.fc = nn.Linear(100 + 8192, 256)
        # self.fc = nn.Linear(input_dim + image_embed_dim, 256)
        
        # generator layers
        self.layer1 = nn.ConvTranspose2d(256, 1024, kernel_size=4, stride=1, padding=0)
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
        

        self.apply(self.weights_init)
        
    def forward(self, noise, image):
        # print("\n--- Forward Pass ---")
        # print(f"Noise shape: {noise.shape}, Image shape: {image.shape}")
        
        # embed the image
        image_features = self.image_embedding(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten the feature map
        # print(f"Image embedding shape: {image_features.shape}")
        
        noise_flat = noise.view(noise.size(0), -1)  # Flatten the noise tensor to 2D

        combined_input = torch.cat([noise_flat, image_features], dim=1)
        combined_input = self.fc(combined_input).view(-1, 256, 1, 1)
        
        x = self.layer1(combined_input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.layer5(x)

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
