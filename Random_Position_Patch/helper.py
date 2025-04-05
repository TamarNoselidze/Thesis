import torch, os, random, json, wandb, requests
from PIL import Image
from io import BytesIO
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
    generator.eval()  
    
    print(f'  > Loaded generator from {checkpoint_path} ')
    return generator



def save_generator(generator_name, generator, output_dir='checkpoints'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the target-specific directory if it doesn't exist

    # f"best_generator", best_generator, f'{checkpoint_dir}/best_generators'
    # f'generator_epoch_{epoch + 1}'
    save_path = os.path.join(output_dir, f'{generator_name}.pth')
    torch.save(generator.state_dict(), save_path)
    print(f'  > Saved generator to {save_path}')



def load_checkpoints(checkpoint_files):

    checkpoints = []

    for checkpoint_file in checkpoint_files:
        parts = checkpoint_file.split('_')
        epoch = int(parts[2].split('.')[0])
        # checkpoint_iter = int(parts[4].split('.')[0])  

        # if checkpoint_iter == iter: 
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


def fetch_patches_from_recent_runs(project_path, noises, save_dir='downloads', max_runs=5):

    os.makedirs(save_dir, exist_ok=True)
    
    api = wandb.Api()
    runs = api.runs(project_path, order="-created_at")
    results_dict = {}

    for iter, run in enumerate(runs[:max_runs]):
        run_id = run.id
        run_save_dir = os.path.join(save_dir, run_id)
        os.makedirs(run_save_dir, exist_ok=True)

        # --- Fetch image from history ---
        results = {}
        for i in range(noises):
            noise_i = i+1
            image_key = f"best_patches/noise_{noise_i}"
            patch_img = None
            try:
                history = run.history(keys=[image_key])
                for row in history:
                    if image_key in row:
                        img_url = row[image_key]["path"]
                        img_response = requests.get(img_url)
                        patch_img = Image.open(BytesIO(img_response.content))
                        break
            except Exception as e:
                print(f"[{run_id}] Failed to fetch image: {e}")

            # --- Fetch tensor artifact ---
            patch_tensor = None
            try:
                artifact_name = f"patch_{noise_i}:latest"
                entity = project_path.split('/')[0]
                artifact = api.artifact(f"{entity}/{artifact_name}", type="patch_tensor")
                artifact_dir = artifact.download(root=run_save_dir)
                tensor_path = os.path.join(artifact_dir, f"best_patch_{noise_i}.pt")
                patch_tensor = torch.load(tensor_path, map_location='cpu')
            except Exception as e:
                print(f"[{run_id}] Failed to fetch tensor: {e}")

            if patch_img is not None and patch_tensor is not None:
                results[noise_i] = ((patch_img, patch_tensor))

        results_dict[iter] = results

    return results_dict



results_dict = fetch_patches_from_recent_runs("takonoselidze-charles-university/RPP train gpatch =932= vgg16_bn", noises=1, max_runs=1)

for iter, results in results_dict.items():
    print(f'Results for iteration {iter}:')
    for noise_i, patches in results.items():
        print(f" Tensor shape: {patches[1].shape}")
        patches[0].show()
