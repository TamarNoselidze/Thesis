import torch, os, random, json, wandb, requests
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from generator import Generator 

def load_classes(image_folder_path, num_of_classes):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader



def load_generator(generator, checkpoint_filename, output_dir='checkpoints'):

    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the generator state
    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()  
    
    print(f'  > Loaded generator from {checkpoint_path} ')
    return generator



def save_generator(generator_name, generator, output_dir='checkpoints'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the target-specific directory if it doesn't exist

    save_path = os.path.join(output_dir, f'{generator_name}.pth')
    torch.save(generator.state_dict(), save_path)
    print(f'  > Saved generator to {save_path}')



def load_checkpoints(checkpoint_files):

    checkpoints = []

    for checkpoint_file in checkpoint_files:
        parts = checkpoint_file.split('_')
        epoch = int(parts[2].split('.')[0])
        checkpoints.append((epoch, checkpoint_file))

    return checkpoints



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



def load_best_patch(project_name):

    api = wandb.Api()
    entity_name = "takonoselidze-charles-university"

    runs = api.runs(f"{entity_name}/{project_name}")  # Fetch all runs in the project
    patches = {}
    recent_runs = sorted(runs, key=lambda x: x.created_at, reverse=True)[:5]  # 5 most recent runs


    for run in recent_runs:
        run_id = run.id  # Get the unique run ID
        artifact_name = f"patch_*"  # Wildcard to match all patch artifacts in the run

        for artifact in run.logged_artifacts():
            if artifact.type == "patch_tensor" and artifact.name.startswith("patch_"):
                artifact_dir = artifact.download()
                noise_i = artifact.name.split("_")[-1]  # Extract noise_i from name
                tensor_path = os.path.join(artifact_dir, f"best_patch_{noise_i}.pt")

                if os.path.exists(tensor_path):
                    patches[f"{run_id}_noise_{noise_i}"] = torch.load(tensor_path)

    return patches



def fetch_generators_from_wandb(generator_class, project, patch_size, save_dir='downloads', max_runs=5):
    save_dir = f'{save_dir}/{project}'

    os.makedirs(save_dir, exist_ok=True)
    entity = "takonoselidze-charles-university"
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}', order="-created_at")
    results_dict = {}


    for iter, run in enumerate(runs[:max_runs]):
        run_id = run.id
        gen = fetch_best_generator_from_run(generator_class, patch_size, entity, project, run_id)
        results_dict[iter] = gen
    return results_dict


def fetch_best_generator_from_run(generator_class, patch_size, entity, project, run_id, save_dir="downloads"):
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
                
                # Instantiate & load model on the right device
                generator = generator_class(patch_size).to(device)
                generator.load_state_dict(torch.load(generator_path, map_location=device))
                generator.eval()
                return generator

        print("No model artifact found in this run.")
        return None

    except Exception as e:
        print(f"Error fetching generator from run {run_id}: {e}")
        return None
    

