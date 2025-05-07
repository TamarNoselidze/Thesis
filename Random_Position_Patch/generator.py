import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    A simple generator network (consisting of five layers) for creating adversarial patches of different sizes.

    Attributes:
        patch_size (int): Desired output patch size (e.g., 16, 32, 64, 80).
        k (float): Scaling factor used in the final tanh-based thresholding.
        model (nn.Sequential): The sequential model composed of ConvTranspose2d layers.
    """
    def __init__(self, patch_size, input_dim=100, output_dim=3, k=0.5):

        super(Generator, self).__init__()
        self.k = k
        self.patch_size = patch_size

        # Build the model dynamically based on patch size
        layers = self.build_layers(input_dim, output_dim, patch_size)
        self.model = nn.Sequential(*layers)

        self.apply(self.weights_init)   # Initialize weights
        

    def build_layers(self, input_dim, output_dim, patch_size):
        """
        Builds a series of layers based on the patch size.

        Args:
            input_dim: number of input channels.
            output_dim: number of output channels.
            patch_size:  output patch size.

        Returns:
            list of nn.Module layers.
        """
        layers = []
        
        # First layer: always maps from (1x1) to (4x4)
        layers.append(nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.ReLU(inplace=False))

        # Add layers based on the target patch size
        if patch_size == 64:
            layers.append(self.make_layer(1024, 512, kernel_size=4, stride=2, padding=1))  
            layers.append(self.make_layer(512, 256, kernel_size=4, stride=2, padding=1))   
            layers.append(self.make_layer(256, 128, kernel_size=4, stride=2, padding=1))   
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1))  
            
        elif patch_size == 80:
            layers.append(self.make_layer(1024, 512, kernel_size=6, stride=2, padding=1))  
            layers.append(self.make_layer(512, 256, kernel_size=5, stride=2, padding=1))  
            layers.append(self.make_layer(256, 128, kernel_size=4, stride=2, padding=2))   
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1)) 

        elif patch_size == 16:
            layers.append(self.make_layer(1024, 512, kernel_size=3, stride=1, padding=1))  
            layers.append(self.make_layer(512, 256, kernel_size=4, stride=2, padding=1))   
            layers.append(self.make_layer(256, 128, kernel_size=3, stride=1, padding=1))   
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1))  

        elif patch_size == 32:
            layers.append(self.make_layer(1024, 512, kernel_size=3, stride=1, padding=1))  
            layers.append(self.make_layer(512, 256, kernel_size=4, stride=2, padding=1))   
            layers.append(self.make_layer(256, 128, kernel_size=4, stride=2, padding=1))   
            layers.append(nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1))

        else:
            raise ValueError("Unsupported patch size!")

        return layers


    def make_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """ Creates a ConvTranspose2d + BatchNorm + ReLU block. """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
        
    def forward(self, x):
        """ Forward pass through the generator. """
        x = self.model(x)
        x = self.threshold(x)

        return x
    
    def threshold(self, x):
        """ A tanh-based transformation to constrain pixel values. """
        x_tanh = torch.tanh(x)
        return self.k * x_tanh + self.k

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.constant_(m.bias.data, 0)

    def reset_weights(self):
        """ Reset the weights of the model. """
        self.apply(self.weights_init)
