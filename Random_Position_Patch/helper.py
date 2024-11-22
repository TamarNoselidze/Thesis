import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os


def load_generator(generator, checkpoint_path):
    filename = os.path.basename(checkpoint_path)  # Get the file name
    parts = filename.split('_')
    epoch = int(parts[3].split('.')[0])  # Extract epoch number
    target_class = int(parts[1])  # Assuming format: generator_{target_class}_epoch_{epoch}.pth
    
    # Load the generator state
    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()  # Set the generator to evaluation mode for testing
    
    print(f'  > Loaded generator from {checkpoint_path} (Epoch {epoch}, Target Class {target_class})')
    return generator, epoch, target_class


def save_generator(generator, epoch, target_class, output_dir='checkpoints'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    save_path = os.path.join(output_dir, f'generator_epoch_{epoch + 1}_{target_class}.pth')
    torch.save(generator.state_dict(), save_path)
    print(f'  > Saved generator at epoch {epoch + 1} to {save_path}')



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
