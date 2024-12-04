import torch

class DeployerMini:
    def __init__(self, patch_size=16, num_patches=16):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def deploy(self, patches, image):
        # Extract image dimensions
        _, H, W = image.shape
        P_H, P_W = self.patch_size, self.patch_size

        print(f"Image size: {image.shape}")
        print(f"Patch size: {patches.shape}")


        # Ensure patches has the correct number of patches
        # assert patches.shape[0] == self.num_patches, "Number of patches does not match expected count."

        occupied = torch.zeros(H, W, dtype=torch.bool)
        
        for patch in patches:

            print(f"Deploying patch size: {patch.shape}")  # Add this line
            # Try to place patch at a random position without overlap

            if patch.shape[1:] != (P_H, P_W):
                print(f"Error: Patch size {patch.shape[1:]} does not match expected size ({P_H}, {P_W})")
            
            placed = False
            attempts = 0
            max_attempts = 100  # Maximum attempts to find a non-overlapping position
            
            while not placed and attempts < max_attempts:
                # Generate random top-left coordinates for the patch placement
                k = torch.randint(0, H - P_H + 1, (1,)).item()
                l = torch.randint(0, W - P_W + 1, (1,)).item()

                # Check if the area is free (not occupied)
                if not occupied[k:k + P_H, l:l + P_W].any():
                    # Mark the area as occupied
                    occupied[k:k + P_H, l:l + P_W] = True

                    # Apply the patch on the image
                    image[:, k:k + P_H, l:l + P_W] = patch
                    placed = True
                
                attempts += 1
            
            # If no suitable position is found, stop trying further
            if not placed:
                print("Warning: Could not place all patches without overlap.")
                break

        return image