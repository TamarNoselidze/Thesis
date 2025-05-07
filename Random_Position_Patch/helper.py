import torch, os, json, wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def load_classes(image_folder_path):
    """
    Load images from a folder and returns a DataLoader with basic preprocessing
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader



def load_generator(generator, checkpoint_filename, output_dir='checkpoints'):
    """
    Load a generator model's weights from a .pth checkpoint. 
    """

    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator not found at {checkpoint_path}")


    generator.load_state_dict(torch.load(checkpoint_path)) # Load model weights
    generator.eval()  # Set to evaluation mode
    
    print(f'  > Loaded generator from {checkpoint_path} ')
    return generator



def save_generator(generator_name, generator, output_dir='checkpoints'):
    """ Save a generator's weights to a specified file. """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    save_path = os.path.join(output_dir, f'{generator_name}.pth')
    torch.save(generator.state_dict(), save_path)
    print(f'  > Saved generator to {save_path}')



def load_checkpoints(checkpoint_files):
    """ Parse checkpoint filenames and extract epoch numbers"""
    checkpoints = []

    for checkpoint_file in checkpoint_files:
        parts = checkpoint_file.split('_')
        epoch = int(parts[2].split('.')[0]) # filename format includes "_epochNumber.pth"
        checkpoints.append((epoch, checkpoint_file))

    return checkpoints



def get_class_name(number, json_file='class_mapping.json'):
    """ Load a human-readable class name for a given numerical class label from a JSON file """
    try:
        with open(json_file, 'r') as file:
            class_mapping = json.load(file)
        
        class_name = class_mapping.get(str(number), "Class not found")
        return class_name
    
    except FileNotFoundError:
        return "JSON file not found. Make sure the file exists and the path is correct."
    except json.JSONDecodeError:
        return "Error decoding JSON file. Ensure the file contains valid JSON."




def fetch_generators_from_wandb(entity, generator_class, project, patch_size, save_dir='downloads', max_runs=5):
    """
    For a given W&B project, load generator models from given number of runs. 
    """

    save_dir = f'{save_dir}/{project}'

    os.makedirs(save_dir, exist_ok=True)
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}', order="-created_at")
    results_dict = {}

    # Download and load generators from W&B
    for iter, run in enumerate(runs[:max_runs]):
        run_id = run.id
        gen = fetch_best_generator_from_run(generator_class, patch_size, entity, project, run_id)
        results_dict[iter] = gen
    return results_dict


def fetch_best_generator_from_run(generator_class, patch_size, entity, project, run_id, save_dir="downloads"):
    """ Helper to fetch a single generator model from a W&B run. """
    save_dir = f'{save_dir}/{project}'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        for artifact_ref in run.logged_artifacts():
            if artifact_ref.type == "model":
                print(f"Found model artifact: {artifact_ref.name}")
                artifact_dir = artifact_ref.download(root=save_dir)

                model_files = os.listdir(artifact_dir)
                generator_path = os.path.join(artifact_dir, model_files[0])
                
                # Load generator model weights
                generator = generator_class(patch_size).to(device)
                generator.load_state_dict(torch.load(generator_path, map_location=device))
                generator.eval()
                return generator

        print("No model artifact found in this run.")
        return None

    except Exception as e:
        print(f"Error fetching generator from run {run_id}: {e}")
        return None
    

