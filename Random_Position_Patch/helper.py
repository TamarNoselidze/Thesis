import torch, os, random, json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_random_classes(image_folder_path, num_of_classes):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    
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



def load_generator(generator, checkpoint_filename, output_dir='checkpoints'):

    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the generator state
    generator.load_state_dict(torch.load(checkpoint_path))
    # generator.eval()  
    
    print(f'  > Loaded generator from {checkpoint_path} ')
    return generator



def save_generator(generator_name, generator, output_dir='checkpoints'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the target-specific directory if it doesn't exist


    # f"best_iter_{iter}", best_generator, f'{checkpoint_dir}/best_generators'
    # f'generator_epoch_{epoch + 1}_iter_{iter}'
    save_path = os.path.join(output_dir, f'{generator_name}.pth')
    torch.save(generator.state_dict(), save_path)
    print(f'  > Saved generator to {save_path}')



def load_checkpoint_by_iteration(checkpoint_files, iter):
    # iter_checkpoints = {}

    # for checkpoint_file in checkpoint_files:
    #     epoch = int(checkpoint_file.split('_')[2])
    #     # iter = int(parts[4])  
    #     if iter not in iter_checkpoints:
    #         iter_checkpoints[iter] = []
    #     iter_checkpoints[iter].append((epoch, checkpoint_file))

    # return iter_checkpoints

    iter_checkpoints = []

    for checkpoint_file in checkpoint_files:
        parts = checkpoint_file.split('_')
        epoch = int(parts[2])
        checkpoint_iter = int(parts[4].split('.')[0])  

        if checkpoint_iter == iter:
            iter_checkpoints.append((epoch, checkpoint_file))

    return iter_checkpoints



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

