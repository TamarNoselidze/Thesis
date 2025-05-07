import torch
import random

class DeployerMini:
    """
    A Deployer mechanism that deploys multiple small patches (Mini-Patches) on an image.

    Parameters:
    - num_patches: Number of patches to deploy.
    - critical_points - represents the type of Mini-Patch attack:
    0 = random placement method;
    1 = corner-point method (i.e. patch is placed at token intersection locations on the ViT token grid);
    2 = token-replacement method (patch replaces token area).
    - allow_overlap: Whether overlapping patches are allowed (only for random placement).
    """

    def __init__(self, num_patches, critical_points=0, allow_overlap=False):
        self.num_patches = num_patches
        self.allow_overlap = allow_overlap
        self.critical_points = critical_points


    def deploy(self, patch, image):
        """
        Applies the mini-patches to the image.

        Parameters:
        - patch (Tensor): Patch of shape (C, P_H, P_W)
        - image (Tensor): Image of shape (C, H, W)

        Returns:
        - adversarial_image (torch.Tensor): patched image of shape (C, H, W)
        """
        _, H, W = image.shape
        _, P_H, P_W = patch.shape
        mask = torch.zeros_like(image)

        if self.critical_points == 0:         # Random placement
            deployed_count = 0
            occupied = torch.zeros(H, W, dtype=torch.bool)

            while deployed_count < self.num_patches:
                if self.allow_overlap:       # Patches can be placed anywhere
                    k = torch.randint(0, H - P_H + 1, (1,)).item()
                    l = torch.randint(0, W - P_W + 1, (1,)).item()
                    mask[:, k:k + P_H, l:l + P_W] = patch

                else:                        # Patches can't overlap
                    placed = False
                    while not placed:
                        k = torch.randint(0, H - P_H + 1, (1,)).item()
                        l = torch.randint(0, W - P_W + 1, (1,)).item()

                        if not occupied[k:k + P_H, l:l + P_W].any():
                            occupied[k:k + P_H, l:l + P_W] = True
                            mask[:, k:k + P_H, l:l + P_W] = patch
                            placed = True

                deployed_count += 1
        else: 
            # Get token-based or center-based critical locations                
            critical_areas = get_random_critical_areas(self.critical_points, self.num_patches, (H, W), (P_H, P_W))

            for (center_x, center_y) in critical_areas:
                # Top-left corner of patch placement
                top_left_x = max(0, center_x - P_H // 2)
                top_left_y = max(0, center_y - P_W // 2)

                # Bottom-right corner of the patch
                bottom_right_x = min(H, top_left_x + P_H)
                bottom_right_y = min(W, top_left_y + P_W)

                # Apply the patch
                mask[:, top_left_x:bottom_right_x, top_left_y:bottom_right_y] = patch[:, :bottom_right_x - top_left_x, :bottom_right_y - top_left_y]


        adversarial_image = mask + (1 - mask) * image  # Combine mask with original image

        return adversarial_image



def get_random_critical_areas(critical_type, numOfPoints, image_dim, patch_dim):
    """ Sample a given number of points from the full set of critical positions. """
    all_areas = get_critical_centroids(critical_type, image_dim, patch_dim)
    random_areas = random.sample(all_areas, min(numOfPoints, len(all_areas)))

    return random_areas



def get_critical_centroids(critical_type, image_dim, patch_dim):
    """ 
    Returns a list of critical positions based on the type.

    critical_type:
    - 1: corner-point method, i.e. patches are centered at exactly token intersection locations on the ViT token grid.
    - 2: token-replacement method, i.e. patches are centered at exactly the center of each token (for directly replacing tokens).
    """
    image_H, image_W = image_dim 
    patch_H, patch_W = patch_dim

    centr_coordinates = []

    for i in range(0, image_H, patch_H):
        for j in range(0, image_W, patch_W):
            if critical_type == 1:
                centr_coordinates.append((i, j))   # Use token top-left corners

            else:                                  # Use token centers
                center_x = i + patch_H // 2
                center_y = j + patch_W // 2
                centr_coordinates.append((center_x, center_y))

    return centr_coordinates

