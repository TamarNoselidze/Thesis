import torch
class DeployerMini:
    def __init__(self, num_patches=16, allow_overlap=False):
        self.num_patches = num_patches
        self.allow_overlap = allow_overlap


    def deploy(self, patch, image):
        _, H, W = image.shape
        _, P_H, P_W = patch.shape

        # print(f"Image size: {image.shape}")
        # print(f"Patch size: {patch.shape}")

        mask = torch.zeros_like(image)
        occupied = torch.zeros(H, W, dtype=torch.bool)
        
        deployed_count = 0
        while deployed_count < self. num_patches:

            if self.allow_overlap:
                k = torch.randint(0, H - P_H + 1, (1,)).item()
                l = torch.randint(0, W - P_W + 1, (1,)).item()

                mask[:, k:k + P_H, l:l + P_W] = patch  # apply the patch within the mask region
            
            else: 
                placed = False
                while not placed:
                    k = torch.randint(0, H - P_H + 1, (1,)).item()
                    l = torch.randint(0, W - P_W + 1, (1,)).item()

                    # Check if the area is free (not occupied)
                    if not occupied[k:k + P_H, l:l + P_W].any():
                        # Mark the area as occupied
                        occupied[k:k + P_H, l:l + P_W] = True

                        # add the patch to the mask
                        mask[:, k:k + P_H, l:l + P_W] = patch
                        placed = True
                
            deployed_count +=1
        
        adversarial_image = mask + (1 - mask) * image  # combine the mask with the original image

        return adversarial_image
