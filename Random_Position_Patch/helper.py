import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os



def save_image(original, patched, label, save_dir):
    to_pil = transforms.ToPILImage()
    original_img = to_pil(original)
    patched_img = to_pil(patched)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(patched_img)
    axs[1].set_title("Patched Image")
    axs[1].axis('off')

    save_path = os.path.join(save_dir, f'{label}.png')
    plt.savefig(save_path)
    print(f"Image saved at {save_path}")
    plt.close()



def save_patch(patch, filename='adversarial_patch.png'):
    # Check if patch has 4 dimensions (batch_size, channels, height, width)
    if patch.dim() == 4:
        patch = patch[0]
    
    patch = patch.cpu().detach()

    unloader = transforms.ToPILImage()
    image = unloader(patch)
    
    # Save the image
    image.save(filename)
