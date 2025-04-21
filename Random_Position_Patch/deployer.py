import torch

class Deployer:
    def deploy(self, patches, images):
        # # print(f'--- SHAPE OF THE PATCH: {patch.shape}')
        # _, H, W = image.shape
        # _, P_H, P_W = patch.shape
        
        # # mask with the same shape as the image
        # mask = torch.zeros_like(image)
        
        # # a random position within the image for the patch
        # k = torch.randint(0, H - P_H + 1, (1,)).item()
        # l = torch.randint(0, W - P_W + 1, (1,)).item()

        # mask[:, k:k + P_H, l:l + P_W] = patch  # apply the patch within the mask region
        
        # # apply the patch on the image
        # adversarial_image = mask + (1 - mask) * image  # combine the mask with the original image
        
        # return adversarial_image
        B, C, H, W = images.shape
        _, _, P_H, P_W = patches.shape

        adv_images = torch.empty_like(images)

        for i in range(B):
            image = images[i]
            patch = patches[i]

            # Create empty mask
            mask = torch.zeros_like(image)

            # Random position
            k = torch.randint(0, H - P_H + 1, (1,)).item()
            l = torch.randint(0, W - P_W + 1, (1,)).item()

            # Apply patch
            mask[:, k:k + P_H, l:l + P_W] = patch
            adv_images[i] = mask + (1 - mask) * image

        return adv_images