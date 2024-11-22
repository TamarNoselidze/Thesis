import os, argparse, random, numpy
import wandb

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

from deployer import Deployer
from loss import AdversarialLoss
from generator import Generator
from helper import save_generator, load_generator

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights


api_key = os.getenv('WANDB_API_KEY')


def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor, contrast=0.3)
    adjusted_image = brightness_transform(image)
    return adjusted_image


def get_models(model_names, device):
    models = []
    for model_name in model_names:
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet152':
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        elif model_name == 'vgg16_bn':
            model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        elif model_name == 'vit_b_16':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif model_name == 'vit_b_32':
            model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        elif model_name == 'vit_l_16':
            model = vit_l_16(weights=ViT_L_16_Weights)
        elif model_name == 'swin_b':
            model = swin_b(weights=Swin_B_Weights)
    
        model.to(device)
        model.eval()
        models.append(model)

    return models


def display_images(images):
    i=0
    for original, modified in images.items():
        i+=1    
        image_key = f'best_epoch_img_{i}'
          
        wandb.log({
        image_key: [wandb.Image(original.cpu(), caption=f"Original Image {i}"), 
                    wandb.Image(modified.cpu(), caption=f"Modified Image {i}")]
         })
        
##### CROSS VALIDATION


def test_best_patches(dataloader, deployer, discriminators, target_models, num_of_epochs, patches, target_classes, device):
    print(f'Testing best patches of the epochs on the following discriminators:\n')

    for epoch in range(num_of_epochs):
        
        epoch_images = []
        patch = patches[epoch]
        target_class = torch.tensor(target_classes[epoch], device=device)
        # target_class = torch.tensor(target_class, device=device)
        epoch_asr = 0
        target_model_asr = 0
        for batch in dataloader:
            images, true_labels = batch
            images = images.to(device)
            batch_size = len(images)

            modified_images = []

            for i in range(batch_size):
                modified_image = deployer.deploy(patch, images[i])
                modified_images.append(modified_image)
                epoch_images.append(modified_image)

            # epoch_images.to(device)
            modified_images = torch.stack(modified_images).to(device)

            outputs = []
            for discriminator in discriminators:
                output = discriminator(modified_images)
                outputs.append(output)
            # target_class_y_prime = torch.full((batch_size,), target_class, dtype=torch.long).to(device)
            
            total_predicted = []
            for out in outputs:
                _, predicted = torch.max(out.data, 1)
                # print(type(predicted))
                total_predicted.append(predicted.cpu())
            # print(total_predicted)

            correct_counts = torch.zeros(batch_size).to(target_class.device)
            for predicted in total_predicted:
                correct_counts += (predicted.to(device) == target_class).float()

            majority_threshold = len(total_predicted) // 2
            correct = (correct_counts > majority_threshold).sum().item()

            batch_asr = correct / batch_size
            epoch_asr += batch_asr

        epoch_asr /= len(dataloader)
        print(f'The best patch of the epoch {epoch+1} with a target class {target_class} has ASR: {epoch_asr  * 100:.2f}%')
        if target_models is not None:
            print(f'Now transfering the patch to target models')
            transfer_to_target_models(target_models, images, target_class, device)
    
        # print(f'This patch has ASR: {epoch_asr  * 100:.2f}')


def transfer_to_target_models(models, images, target_class, device):

    total_images = len(images)
    
    for model in models:
        correctly_misclassified = 0
        with torch.no_grad():  # Disable gradients for evaluation
            for image in images:
                # output = model(modified.unsqueeze(0))  
                output = model(image.unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)
                
                if predicted.item() == target_class:
                    correctly_misclassified += 1
        
        # Calculate ASR (Attack Success Rate)
        avg_asr = (correctly_misclassified / total_images) * 100
        print(f"Model: {model.__class__.__name__}")
        print(f"Correctly misclassified: {correctly_misclassified}/{total_images}")
        print(f"Avg ASR: {avg_asr:.2f}%\n")


def load_random_classes(image_folder_path, num_of_classes):
    # Step 1: Load the full dataset to access `dataset.classes`
    dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    
    # Step 2: Randomly select 20 class names from the `dataset.classes`
    selected_classes = set(random.sample(dataset.classes, num_of_classes))
    # print(f'SELECTED RANDOM CLASSES: {selected_classes}')
    
    # Step 3: Filter samples based on the selected class names
    dataset.samples = [(path, target) for path, target in dataset.samples if dataset.classes[target] in selected_classes]
    
    # Step 4: Update dataset.targets to match the filtered samples
    dataset.targets = [target for _, target in dataset.samples]

    # Step 5: Update dataset.classes and dataset.class_to_idx to reflect the selected classes
    dataset.classes = sorted(selected_classes)
    dataset.class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

    # Step 6: Create a DataLoader for the filtered dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    classes = (set(int(class_name) for class_name in dataset.classes))
    # print(type(classes))

    return dataloader, classes


def get_target_classes(train_classes, total_classes, num_of_target_classes):

    # train_classes_set = set(train_classes)
    all_classes = set(range(total_classes))
    
    available_classes = list(all_classes - train_classes)

    # print(F"AVAILABLE CLSASES FOR TARGETS: {available_classes}")
    
    # Randomly select num_of_target_classes from the available classes
    target_classes = random.sample(available_classes, num_of_target_classes)
    
    return target_classes


def evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device):
    total_asr = 0

    for batch in dataloader:
        images, _ = batch
        images = images.to(device)
        batch_size = images.shape[0]
        adv_images = []
        for image in images:
            adv_image = deployer.deploy(patch, image)
            adv_images.append(adv_image)

        adv_images = torch.stack(adv_images).to(device)

        outputs = []
        for discriminator in discriminators:
            output = discriminator(adv_images)
            outputs.append(output)

        total_predicted = []
        for out in outputs:
            _, predicted = torch.max(out.data, 1)
            total_predicted.append(predicted.cpu())

        correct_counts = torch.zeros(batch_size).to(device)
        for predicted in total_predicted:
            correct_counts += (predicted.to(device) == target_class).float()

        majority_threshold = len(total_predicted) // 2
        correct = (correct_counts > majority_threshold).sum().item()

        batch_asr = correct / batch_size
        total_asr += batch_asr

    total_asr /= len(dataloader)

    return total_asr


def evaluate_saved_generators(checkpoints_dir, fixed_noise, patch_size, dataloader, deployer, discriminators, device):
    best_asr = 0
    best_epoch = -1
    best_patch = None

    # List all saved checkpoints
    checkpoint_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')])

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
        generator = Generator(patch_size)  # Initialize the generator architecture
        generator, epoch, target_class = load_generator(generator, checkpoint_path)

        # Generate the patch from fixed noise
        patch = generator(fixed_noise).detach()
        patch = patch.squeeze(0)  # Shape: [3, 64, 64]
        # Evaluate the patch
        asr = evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device)
        print(f"Epoch {epoch} - Patch ASR: {asr * 100:.2f}%")
        patch_key = f'epoch_{epoch}_best_patch'
        wandb.log({patch_key : wandb.Image(patch.cpu(), caption=f'Patch of epoch {epoch+1}')})


        if asr > best_asr:
            best_asr = asr
            best_epoch = epoch
            best_patch = patch

    print(f"Best generator found at epoch {best_epoch} with ASR: {best_asr * 100:.2f}%")
    return best_epoch, best_patch



def gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, classes, target_class, num_of_epochs, checkpoints_dir, input_dim=100):
    
    total_asr = 0
    # best_asr = 0

    for epoch in range(num_of_epochs):
        print(f'@ Epoch {epoch+1} with a target class: {target_class}')
        epoch_asr = 0
        batch_i=1

        for batch in dataloader:
            print(f'@  Batch {batch_i}')
            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)

            batch_size = images.shape[0]   # might change for the last batch
            noise = torch.randn(batch_size, input_dim, 1, 1).to(device)
            modified_images = []
            adv_patches = generator(noise)

            # deploying 
            # patch = adv_patches[0]  # one patch for all images in the batch
            # patch = adv_patches[random.randint(0, batch_size-1)]
            for i in range(batch_size):
                patch = adv_patches[i]
                # print(f'----------------- The patch is:\n{patch}')
                modified_image = deployer.deploy(patch, images[i])
                modified_images.append(modified_image)

            modified_images = torch.stack(modified_images).to(device)

            # multiple discriminators
            outputs = []
            for discriminator in discriminators:
                output = discriminator(modified_images)
                outputs.append(output)
            # print(f"   Discriminator output: {outputs.data.cpu()}")

            target_class_y_prime = torch.full((batch_size,), target_class, dtype=torch.long).to(device)
            
            criterion = AdversarialLoss(target_class=target_class_y_prime).to(device)
            # weighted loss for all the discriminators
            total_loss = sum([criterion(out) for out in outputs])
            # loss = criterion(outputs)
            loss = total_loss / len(discriminators)   # weighted loss ? ? ?
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_predicted = []
            for out in outputs:
                _, predicted = torch.max(out.data, 1)
                total_predicted.append(predicted.cpu())

            # correct = sum([(predicted == target_class).sum().item() for predicted in total_predicted])
            correct_counts = torch.zeros(batch_size).to(device)
            for predicted in total_predicted:
                correct_counts += (predicted.to(device) == target_class).float()

            # majority vote (more than half of the discriminators)
            majority_threshold = len(total_predicted) // 2
            correct = (correct_counts > majority_threshold).sum().item()

            print(f"     Loss: {loss.item()}")
            print(f"     True labels: {true_labels.cpu()}")
            print(f"     Predicted labels: {total_predicted}")
            print(f"     Correctly misclassified: {correct}")

            batch_asr = correct / batch_size
            print(f'@    Batch (number {batch_i}) has ASR: {batch_asr}')
            batch_i+=1

            # Log batch metrics to W&B
            # wandb.log({
            #     'batch_loss': loss.item(),
            #     'batch_asr': batch_asr,
            # })
            
            epoch_asr += batch_asr

        avg_epoch_asr = epoch_asr / len(dataloader)
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        wandb.log({
            'epoch_avg_asr': avg_epoch_asr,
            # 'epoch': epoch + 1
        })
      
        print(f"Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")

        # test_generator()
        # if avg_epoch_asr > best_asr:
        #     best_asr = avg_epoch_asr
        #     save_generator(generator, epoch, output_dir=output_dir)
        #     print(f"Saved new best generator with ASR: {best_asr * 100:.2f}%")
        save_generator(generator, epoch, target_class, checkpoints_dir)

    print(f'\n\n-- Average ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% --\n\n')



def start_iteration(device, patch_size, discriminators, dataloader, classes, target_classes, checkpoint_dir, num_of_epochs=40, brightness_factor=None, color_transfer=None, batch_size=32):

   
    generator = Generator(patch_size).to(device)
    generator.train()
    deployer = Deployer()

    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values

    print(f'Using device: {device}')
    print(f"Generator device: {next(generator.parameters()).device}")

    fixed_noise = torch.randn(1, 100, 1, 1).to(device)
    
    for iteration in range(len(target_classes)):
        target_class = target_classes[iteration]
        target_class = torch.tensor(target_class, device=device)
        gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, classes, target_class, num_of_epochs, checkpoint_dir)


    best_epoch, best_patch = evaluate_saved_generators(checkpoint_dir, fixed_noise, patch_size, dataloader, deployer, discriminators, device)

    print(f'Best porforming epoch: {best_epoch}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--checkpoint_folder_path', help='Path to a folder where generators will be saved', default='./checkpoints')
    parser.add_argument('--transfer_mode', choices=['source-to-target', 'ensemble', 'cross-validation'], 
                        help='Choose the transferability approach: source-to-target, ensemble, or cross-validation', default='source-to-target')
    parser.add_argument('--training_models', type=str)
                        # nargs='+', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='List of training models')
    parser.add_argument('--target_models', type=str)
    # , nargs='+', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='List of target models')    
    parser.add_argument('--patch_size', choices=['48', '64', '80'], help='Size of the adversarial patch', default=64)
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--num_of_train_classes', type=int, help='Number of (random) classes to train the generator on', default=100)
    parser.add_argument('--num_of_target_classes', type=int, help='Number of (random) target classes to misclassify images as', default=10)
    parser.add_argument('--brightness', help='Brightness level for the patch')
    parser.add_argument('--color_transfer', help='Color transfer value for the patch')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_folder_path
    transfer_mode = args.transfer_mode
    training_model_names = args.training_models.split(' ') if args.training_models else None
    target_model_names = args.target_models.split(' ') if args.target_models else None
    intra_model_attack = False
    cross_validation = False

    if training_model_names is None:
        raise ValueError('You should specify training models.')  
    
    if transfer_mode == 'source-to-target':
        if target_model_names is None:
            intra_model_attack = True
        else: 
            raise ValueError("For the source-to-target attack you cannot have additional target models.")

    else:
        if transfer_mode == 'cross-val':
            cross_validation = True
        if target_model_names is None:
            raise ValueError("For the ensemble/cross-validation attack you should specify target models.")
        
    patch_size = args.patch_size

    try:
        brightness_factor = float(args.brightness)
    except:
        brightness_factor = None

    try:
        color_transfer = float(args.color_transfer)
    except:
        color_transfer = None

    discriminators = get_models(training_model_names, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # dataset = datasets.ImageFolder(args.image_folder_path, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataloader, classes = load_random_classes(args.image_folder_path, args.num_of_train_classes)
    num_of_classes = len(classes)
    print(f'CLASSES: {classes}')
    print(f'IN TOTAL: {len(classes)}')
    num_of_epochs = args.epochs
    num_of_target_classes = args.num_of_target_classes

    target_classes = get_target_classes(classes, 1000, num_of_target_classes)
    print(f'TARGET CLASSES: {target_classes}')

    

    project_name = (
        f'RPP {transfer_mode} ' +
        f'train-{",".join(training_model_names)} ' + 
        (f'target-{",".join(target_model_names)}' if target_model_names else '') +
        (f' (br {brightness_factor})' if brightness_factor else '') + 
        (f' _col-tr{color_transfer}' if color_transfer else '')
    )

    # Initialize W&B
    wandb.init(project=project_name, entity='takonoselidze-charles-university', config={
        'epochs': num_of_epochs,
        'classes' : num_of_classes,
        'target_classes' : num_of_target_classes,
    })

    epoch_patches = start_iteration(device, patch_size, discriminators, dataloader, classes, target_classes, checkpoint_dir, num_of_epochs, brightness_factor, color_transfer)

    target_models = None
    # if not intra_model_attack: 
    # # if target_model_names is not None:
    #     target_models = get_models(target_model_names, device)
    # test_best_patches(dataloader, deployer, discriminators, target_models, num_of_epochs, epoch_patches, target_classes, device)

    wandb.finish()
