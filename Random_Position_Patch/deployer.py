import torch

class Deployer:
    """
    A Deployer mechanism that deploys a single adversarial patch (i.e. G-Patch) at a random position of an image.
    """
    def deploy(self, patch, image):
        """
        Applies a given adversarial patch to a random location on the input image.

        Parameters:
        - patch (torch.Tensor): The adversarial patch tensor of shape (C, P_H, P_W).
        - image (torch.Tensor): The input image tensor of shape (C, H, W).

        Returns:
        - adversarial_image (torch.Tensor): patched image tensor of shape (C, H, W).
        """
        _, H, W = image.shape
        _, P_H, P_W = patch.shape
        
        # Mask with the same shape as the image
        mask = torch.zeros_like(image)
        
        # A random position within the image for the patch
        k = torch.randint(0, H - P_H + 1, (1,)).item()
        l = torch.randint(0, W - P_W + 1, (1,)).item()

        mask[:, k:k + P_H, l:l + P_W] = patch  # Apply the patch within the mask region
        
        # Apply the patch on the image
        adversarial_image = mask + (1 - mask) * image  # Combine the mask with the original image
        
        return adversarial_image
    
    