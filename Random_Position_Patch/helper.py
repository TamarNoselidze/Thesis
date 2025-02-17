import torch, os, random, numpy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import json

def load_random_classes(image_folder_path, num_of_classes):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    
    # Randomly select 20 class names
    selected_classes = set(random.sample(dataset.classes, num_of_classes))
    
    # Filter samples based on the selected class names
    dataset.samples = [(path, target) for path, target in dataset.samples if dataset.classes[target] in selected_classes]
    
    # Update dataset.targets to match the filtered samples
    dataset.targets = [target for _, target in dataset.samples]

    # Update dataset.classes and dataset.class_to_idx to reflect the selected classes
    dataset.classes = sorted(selected_classes)
    dataset.class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

    # Create a DataLoader for the filtered dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    classes = (set(int(class_name) for class_name in dataset.classes))

    return dataloader, classes


def get_target_classes(train_classes, total_classes, num_of_target_classes):

    all_classes = set(range(total_classes))    
    available_classes = list(all_classes - train_classes)
   
    # Randomly select num_of_target_classes from the available classes
    target_classes = random.sample(available_classes, num_of_target_classes)
    
    return target_classes


def load_generator(generator, checkpoint_filename, output_dir='checkpoints'):

    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the generator state
    generator.load_state_dict(torch.load(checkpoint_path))
    # generator.eval()  
    
    print(f'  > Loaded generator from {checkpoint_path} ')
    return generator


def save_generator(generator, epoch, target_class, output_dir='checkpoints'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the target-specific directory if it doesn't exist

    save_path = os.path.join(output_dir, f'generator_epoch_{epoch + 1}_target_{target_class}.pth')
    torch.save(generator.state_dict(), save_path)
    print(f'  > Saved generator at epoch {epoch + 1} to {save_path}')


def load_checkpoint_by_target_class(checkpoint_files):
    target_class_checkpoints = {}

    for checkpoint_file in checkpoint_files:
        parts = checkpoint_file.split('_')
        epoch = int(parts[2])  # Extract epoch number
        target_class = int(parts[4].split('.')[0])  # Extract target class
        if target_class not in target_class_checkpoints:
            target_class_checkpoints[target_class] = []
        target_class_checkpoints[target_class].append((epoch, checkpoint_file))

    return target_class_checkpoints




def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor, contrast=0.3)
    adjusted_image = brightness_transform(image)
    return adjusted_image



def get_class_name(number, json_file='class_mapping.json'):
    try:
        with open(json_file, 'r') as file:
            class_mapping = json.load(file)
        
        class_name = class_mapping.get(str(number), "Class not found")
        return class_name
    
    except FileNotFoundError:
        return "JSON file not found. Make sure the file exists and the path is correct."
    except json.JSONDecodeError:
        return "Error decoding JSON file. Ensure the file contains valid JSON."


# def display_images(images):
#     i=0
#     for original, modified in images.items():
#         i+=1    
#         image_key = f'best_epoch_img_{i}'
          
#         wandb.log({
#         image_key: [wandb.Image(original.cpu(), caption=f"Original Image {i}"), 
#                     wandb.Image(modified.cpu(), caption=f"Modified Image {i}")]
#          })