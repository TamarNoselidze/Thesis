import torch
import random

class DeployerMini:
    def __init__(self, num_patches=16, critical_points=False, allow_overlap=False):
        self.num_patches = num_patches
        self.allow_overlap = allow_overlap
        self.critical_points = critical_points


    def deploy(self, patch, image):
        _, H, W = image.shape
        _, P_H, P_W = patch.shape
        mask = torch.zeros_like(image)

        # print(f"Image size: {image.shape}")
        # print(f"Patch size: {patch.shape}")    return random.sample(all_areas, min(numOfPoints, len(all_areas)))
        
        if self.critical_points:
            # Get random critical areas
            critical_areas = get_random_critical_areas(self.num_patches, (H, W), (P_H, P_W))

            for (center_x, center_y) in critical_areas:
                # Compute top-left corner of patch placement
                top_left_x = max(0, center_x - P_H // 2)
                top_left_y = max(0, center_y - P_W // 2)

                # Ensure patch does not go beyond image boundaries
                bottom_right_x = min(H, top_left_x + P_H)
                bottom_right_y = min(W, top_left_y + P_W)

                # Apply the patch
                mask[:, top_left_x:bottom_right_x, top_left_y:bottom_right_y] = patch[:, :bottom_right_x - top_left_x, :bottom_right_y - top_left_y]

        else:
            deployed_count = 0
            occupied = torch.zeros(H, W, dtype=torch.bool)

            while deployed_count < self.num_patches:
                if self.allow_overlap:
                    k = torch.randint(0, H - P_H + 1, (1,)).item()
                    l = torch.randint(0, W - P_W + 1, (1,)).item()
                    mask[:, k:k + P_H, l:l + P_W] = patch
                else:
                    placed = False
                    while not placed:
                        k = torch.randint(0, H - P_H + 1, (1,)).item()
                        l = torch.randint(0, W - P_W + 1, (1,)).item()

                        if not occupied[k:k + P_H, l:l + P_W].any():
                            occupied[k:k + P_H, l:l + P_W] = True
                            mask[:, k:k + P_H, l:l + P_W] = patch
                            placed = True

                deployed_count += 1

        adversarial_image = mask + (1 - mask) * image  # Combine mask with original image
        return adversarial_image


def get_random_critical_areas(numOfPoints, image_dim, patch_dim):
    all_areas = get_critical_centroids(image_dim, patch_dim)
    return random.sample(all_areas, min(numOfPoints, len(all_areas)))


def get_critical_centroids(image_dim, patch_dim):
    x_offset = y_offset = patch_dim
    # x_range, y_range = image_dim // patch_dim 
    
    centr_coordinates = []
    for x in range(x_offset, image_dim, x_offset):
        for y in range(y_offset, image_dim, y_offset):
            centr_coordinates.append((x,y))

    print(len(centr_coordinates))
    return centr_coordinates
