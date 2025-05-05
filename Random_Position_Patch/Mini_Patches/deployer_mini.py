import torch
import random

class DeployerMini:
    def __init__(self, num_patches, critical_points=0, allow_overlap=False):
        self.num_patches = num_patches
        self.allow_overlap = allow_overlap
        self.critical_points = critical_points


    def deploy(self, patch, image):
        _, H, W = image.shape
        _, P_H, P_W = patch.shape
        mask = torch.zeros_like(image)

        if self.critical_points == 0:
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
        else: 
            # Get random critical areas
            critical_areas = get_random_critical_areas(self.critical_points, self.num_patches, (H, W), (P_H, P_W))

            for (center_x, center_y) in critical_areas:
                # Compute top-left corner of patch placement
                top_left_x = max(0, center_x - P_H // 2)
                top_left_y = max(0, center_y - P_W // 2)

                print(f'top left coordinates: {top_left_x}, {top_left_y}')

                # Ensure patch does not go beyond image boundaries
                bottom_right_x = min(H, top_left_x + P_H)
                bottom_right_y = min(W, top_left_y + P_W)

                print(f'bottom right coordinates: {bottom_right_x}, {bottom_right_y}')

                # Apply the patch
                mask[:, top_left_x:bottom_right_x, top_left_y:bottom_right_y] = patch[:, :bottom_right_x - top_left_x, :bottom_right_y - top_left_y]


        adversarial_image = mask + (1 - mask) * image  # Combine mask with original image

        return adversarial_image


def get_random_critical_areas(critical_type, numOfPoints, image_dim, patch_dim):
    all_areas = get_critical_centroids(critical_type, image_dim, patch_dim)
    random_areas = random.sample(all_areas, min(numOfPoints, len(all_areas)))

    return random_areas

def get_critical_centroids(critical_type, image_dim, patch_dim):
    image_H, image_W = image_dim
    patch_H, patch_W = patch_dim

    centr_coordinates = []

    if critical_type == 1:
        for x in range(0, image_H, patch_H):
            for y in range(0, image_W, patch_W):
                centr_coordinates.append((x, y))

    else:
        for i in range(0, image_H, patch_H):
            for j in range(0, image_W, patch_W):
                center_x = i + patch_H // 2
                center_y = j + patch_W // 2
                centr_coordinates.append((center_x, center_y))

    return centr_coordinates

