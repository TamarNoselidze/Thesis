import torch

class Deployer:
    def deploy(self, patch, image):
        # Image and patch dimensions

        # print(f'--- SHAPE OF THE PATCH: {patch.shape}')
        _, H, W = image.shape
        _, P_H, P_W = patch.shape
        
        # mask with the same shape as the image
        M = torch.zeros_like(image)
        
        # a random position within the image for the patch
        k = torch.randint(0, H - P_H + 1, (1,)).item()
        l = torch.randint(0, W - P_W + 1, (1,)).item()

        M[:, k:k + P_H, l:l + P_W] = patch  # apply the patch within the mask region
        
        # aaply the patch on the image
        T_p_x = M + (1 - M) * image  # combine the mask with the original image
        
        return T_p_x
