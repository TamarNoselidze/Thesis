import os, argparse, random, numpy
import wandb

import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from loss import AdversarialLoss
from deployer import Deployer
from Mini_Patches.deployer_mini import DeployerMini
from generator import Generator
from helper import save_generator, load_generator, load_checkpoint_by_target_class, load_random_classes, get_target_classes

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights


api_key = os.getenv('WANDB_API_KEY')


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



        
def test_best_patch(dataloader, target_models, target_model_names, patch, target_class, device):
    deployer = Deployer()
    missclassified_counts = {model : 0 for model in target_model_names}
    total_image_count = len(dataloader.dataset)
    
    for batch in dataloader:
        images, _ = batch
        images = images.to(device)
        batch_size = images.shape[0]
        for i in range(batch_size):
            modified_image = deployer.deploy(patch, images[i])
            for model, name in zip(target_models, target_model_names):
                # with torch.no_grad():  # Disable gradients for evaluation
                output = model(modified_image.unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)       
                if predicted.item() == target_class:
                    missclassified_counts[name] +=1

    
    for name, count in missclassified_counts.items():
        print(f'The patch of taget class {target_class} on model {name} had {count} misclassifications')    
        asr = count / total_image_count
        print(f'ASR for target model {name}: {asr * 100:.2f}%')    



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



def evaluate_saved_generators(checkpoint_dir, fixed_noise, patch_size, dataloader, deployer, discriminators, device):
    best_asr = 0
    best_epoch = -1
    best_patch = None

    # Iterate over subfolders in the checkpoints directory
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    results = {}

    print(f'\n{"="*38} Evaluating generators {"="*38}')
    # Group checkpoints by target class
    target_class_checkpoints = load_checkpoint_by_target_class(checkpoint_files)

    print(f'checkpoint files: {checkpoint_files}')
    print(f'by target class: {target_class_checkpoints}')

    for target_class, checkpoints in target_class_checkpoints.items():
        print(f'Results for target class {target_class}')
        print("-"*100)

        for epoch, checkpoint_file in checkpoints:
            generator = Generator(patch_size).to(device)
            generator = load_generator(generator, checkpoint_file)

            # Generate a patch and evaluate
            patch = generator(fixed_noise).detach().squeeze(0)
            asr = evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device)
            print(f"Epoch {epoch} -<>- Patch ASR: {asr * 100:.2f}%")

            patch_key = f'epoch_{epoch}_best_patch'
            wandb.log({patch_key : wandb.Image(patch.cpu(), caption=f'Patch of epoch {epoch} target class {target_class}')})
                
        print("-"*100)

        if asr > best_asr:
            best_asr = asr
            best_epoch = epoch
            best_patch = patch
        

        print(f"Best generator found at epoch {best_epoch} with ASR: {best_asr * 100:.2f}%")
        print("-"*100)

        results[target_class] = {
            'best_asr': best_asr,
            'best_epoch': best_epoch,
            'best_patch': best_patch,
        }

    print("="*100)

    return results



def gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, classes, target_class, num_of_epochs, checkpoints_dir, input_dim=100):
    
    total_asr = 0

    for epoch in range(num_of_epochs):
        print(f'@ Epoch {epoch+1} for a target class: {target_class}')
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
            for i in range(batch_size):
                patch = adv_patches[i]
                modified_image = deployer.deploy(patch, images[i])
                modified_images.append(modified_image)

            modified_images = torch.stack(modified_images).to(device)

            # multiple discriminators
            outputs = []
            for discriminator in discriminators:
                output = discriminator(modified_images)
                outputs.append(output)

            target_class_y_prime = torch.full((batch_size,), target_class, dtype=torch.long).to(device)
            
            criterion = AdversarialLoss(target_class=target_class_y_prime).to(device)
            # weighted loss for all the discriminators
            total_loss = sum([criterion(out) for out in outputs])
            loss = total_loss / len(discriminators)  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_predicted = []
            for out in outputs:
                _, predicted = torch.max(out.data, 1)
                total_predicted.append(predicted.cpu())

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

            
            epoch_asr += batch_asr

        avg_epoch_asr = epoch_asr / len(dataloader)
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        wandb.log({
            'epoch_avg_asr': avg_epoch_asr,
        })
      
        print(f"Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")

        save_generator(generator, epoch, target_class, checkpoints_dir)

    print(f'\n\nAverage ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% \n')



def start_iteration(device, attack_type, patch_size, discriminators, dataloader, classes, target_classes, checkpoint_dir, num_of_epochs=40, num_of_patches=8, brightness_factor=None, color_transfer=None):
    if attack_type == 'mini':
        deployer = DeployerMini(patch_size, num_of_patches)
    else:
        deployer = Deployer()

    for iteration in range(len(target_classes)):
        target_class = target_classes[iteration]
        print(f'{"-"*30} Iteration {iteration+1} for target class: {target_class} {"-"*30}')
        generator = Generator(patch_size).to(device)
        generator.train()

        optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))   ## try different values
        target_class = torch.tensor(target_class, device=device)
        gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, classes, target_class, num_of_epochs, checkpoint_dir)
        print(f'{"-"*100}\n\n')

    fixed_noise = torch.randn(1, 100, 1, 1).to(device)
    results = evaluate_saved_generators(checkpoint_dir, fixed_noise, patch_size, dataloader, deployer, discriminators, device)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--checkpoint_folder_path', help='Path to a folder where generators will be saved', default='./checkpoints')
    parser.add_argument('--attack_mode', choices=['gpatch', 'mini'], default='gpatch')
    parser.add_argument('--number_of_patches', type=int, help='Number of patches. 1 for G-patch attack, and more than 1 for mini-patch attack', default=1)
    parser.add_argument('--transfer_mode', choices=['source-to-target', 'ensemble', 'cross-validation'], 
                        help='Choose the transferability approach: source-to-target, ensemble, or cross-validation', default='source-to-target')
    parser.add_argument('--training_models', type=str)
                        # nargs='+', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='List of training models')
    parser.add_argument('--target_models', type=str)
    # , nargs='+', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='List of target models')    
    parser.add_argument('--patch_size', type=int, help='Size of the adversarial patch', default=64)
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

    patch_size = args.patch_size
    attack_mode = args.attack_mode
    # if attack_mode == 'mini':
    num_of_patches = args.number_of_patches

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
        

    try:
        brightness_factor = float(args.brightness)
    except:
        brightness_factor = None

    try:
        color_transfer = float(args.color_transfer)
    except:
        color_transfer = None


    discriminators = get_models(training_model_names, device)


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
        f'{attack_mode}_attack' +
        f'train-{",".join(training_model_names)} ' + 
        (f'target-{",".join(target_model_names)} ' if target_model_names else '') +
        f'{num_of_target_classes} iters ' +
        (f'(br {brightness_factor}) ' if brightness_factor else '') + 
        (f'_col-tr{color_transfer}' if color_transfer else '')
    )

    # Initialize W&B
    wandb.init(project=project_name, entity='takonoselidze-charles-university', config={
        'epochs': num_of_epochs,
        'classes' : num_of_classes,
        'target_classes' : num_of_target_classes,
    })

    results = start_iteration(device, attack_mode, patch_size, discriminators, dataloader, classes, target_classes, checkpoint_dir, num_of_epochs, num_of_patches, brightness_factor, color_transfer)


    target_models = None
    if not intra_model_attack: 
        target_models = get_models(target_model_names, device)
        for target, result in results.items():
            # best_asr = result['best_asr']
            # best_epoch = result['best_epoch']
            best_patch = result['best_patch']
            
            test_best_patch(dataloader, target_models, target_model_names, best_patch, target_class=target, device=device)

    wandb.finish()
