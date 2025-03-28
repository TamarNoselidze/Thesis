import os, argparse

import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from loss import AdversarialLoss
from deployer import Deployer
from Mini_Patches.deployer_mini import DeployerMini
from generator import Generator
from helper import save_generator, load_generator, load_checkpoint_by_iteration, load_random_classes, get_class_name
from wandb_logger import WandbLogger

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights, resnet101, ResNet101_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights


api_key = os.getenv('WANDB_API_KEY')


def get_models(model_names, device):
    models = []
    for model_name in model_names:
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
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



        
def test_best_patch(iter, noise_i, dataloader, attack_mode, target_models, target_model_names, train_model_names, patch, target_class, device, logger, iteration):
    
    attack_type = attack_mode.split('_')
    if attack_type[0] == 'gpatch':
        deployer = Deployer()
    else:
        deployer = DeployerMini(num_of_patches, critical_points=int(attack_type[1]))

    misclassified_counts = {model : {'misclassified' : 0, 'total' : 0} for model in target_model_names}

    # total_image_count = len(dataloader.dataset)
    
    image_i = 0
    total_valid_images = 0  # Keep track of valid images (not skipped)
    for batch in dataloader:
        images, true_labels = batch
        images = images.to(device)
        true_labels = true_labels.to(device)
        
        # Filter out images with true_label == target_class
        valid_indices = true_labels != target_class
        if not valid_indices.any():
            continue  # Skip this batch if all images are to be excluded

        images = images[valid_indices]
        true_labels = true_labels[valid_indices]

        batch_size = images.shape[0]
        total_valid_images += batch_size

        for i in range(batch_size):
            misclassified = False
            modified_image = deployer.deploy(patch, images[i])
            for model, name in zip(target_models, target_model_names):
                with torch.no_grad():  # Disable gradients for evaluation
                    output = model(modified_image.unsqueeze(0).to(device))
                    _, predicted = torch.max(output.data, 1)       
                    if predicted.item() == target_class:
                        misclassified_counts[name]["misclassified"] += 1
                        misclassified = True
                    misclassified_counts[name]["total"] += 1
                
            if image_i % 500 == 0:   # displaying one in every 500 modified images
                # wandb.log({f"modified image_{image_i}" : wandb.Image(modified_image.cpu(), caption=f'target class "{get_class_name(target_class)}" ({target_class})')})
                logger.log_modified_image(iter, noise_i, image_i, modified_image, misclassified)

            image_i +=1
    print(f'WE ARE TESTING THE BEST GENERATOR OF ITERATION {iter}, PATCH GENERATED FROM RANDOM NOISE NUBMER {noise_i}')
    print(f'The generator was trained on: {", ".join(train_model_names)}.')
    for name, results in misclassified_counts.items():
        count = results["misclassified"]
        total_image_count = results["total"]
        print(f'Target class: "{get_class_name(target_class)}" ({target_class})')
        print(f'The generated adversarial patch on model {name} had {count} misclassifications')    
        asr = count / total_image_count
        print(f'ASR for target model {name}: {asr * 100:.2f}%')    

        # Log to wandb
        logger.log_target_model_results(
            iter, noise_i, target_model_names, train_model_names, count, total_image_count, asr
        )
    
    return misclassified_counts

def evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device):
    total_correct_count = 0
    total_valid_images = 0

    with torch.no_grad():  # No gradient tracking
        for batch in dataloader:
            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)
                    
            # Filter out images with true_label == target_class
            valid_indices = true_labels != target_class
            if not valid_indices.any():
                continue  # Skip this batch if all images are to be excluded

            images = images[valid_indices]
            true_labels = true_labels[valid_indices]

            batch_size = images.shape[0]
            total_valid_images += batch_size
            
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

            total_correct_count += correct

        total_asr = total_correct_count / total_valid_images

    return total_asr



def evaluate_saved_generators(iter, target_class, checkpoint_dir, fixed_noises, patch_size, dataloader, deployer, discriminators, device, logger):
    best_avg_asr = 0
    best_epoch = -1
    best_patches = []
    best_generator = None

    last_checkpoint  = None
    # Iterate over subfolders in the checkpoints directory
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    results = {}

    print(f'\n{"="*30} Evaluating generators for iteration {iter}/5 {"="*30}')
    # Group checkpoints by the iteration number
    iter_checkpoints = load_checkpoint_by_iteration(checkpoint_files, iter)

    # for epoch, checkpoints in iter_checkpoints:

    for epoch, checkpoint_file in iter_checkpoints:
            
        generator = Generator(patch_size).to(device)
        generator = load_generator(generator, checkpoint_file)

        total_asr = 0
        patches = []
        for j, fixed_noise in enumerate(fixed_noises):
            noise_i = j+1
            # Generate a patch and evaluate
            patch = generator(fixed_noise).detach().squeeze(0)
            asr = evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device)
            print(f"    ->  Epoch {epoch}, Noise #{noise_i}  -<>- Patch ASR: {asr * 100:.2f}%")

            total_asr += asr
            patches.append(patch)

            logger.log_generator_evaluation(iter, noise_i, epoch, asr)
            logger.log_patch_image(iter, noise_i, epoch, patch)

        avg_asr = total_asr / len(fixed_noises)
        last_checkpoint = checkpoint_file

        if avg_asr > best_avg_asr:
            best_avg_asr = avg_asr
            best_epoch = epoch
            best_patches = patches
            best_generator = generator

        
    if best_epoch == -1:
        generator = Generator(patch_size).to(device)
        generator = load_generator(generator, last_checkpoint)
        save_generator("best_iter_-1", generator, f'{checkpoint_dir}/best_generators')
    else:
        save_generator(f"best_iter_{iter}", best_generator, f'{checkpoint_dir}/best_generators')
        logger.log_best_generator(iter, best_epoch, best_avg_asr)
        # logger.log_best_patch(iter, noise_i, best_epoch, best_patch)

    print(f"At iteration {iter}, best generator found at epoch {best_epoch} with avg ASR: {best_avg_asr * 100:.2f}%")

    results = {
        'best_avg_asr': best_avg_asr,
        'best_epoch': best_epoch,
        'best_patches': best_patches,
    }

    print("="*100)

    return results



def gan_attack(iter, device, generator, optimizer, deployer, discriminators, dataloader, classes, target_class, num_of_epochs, checkpoint_dir, logger, input_dim=100):
    
    total_asr = 0
    # total_valid_images = 0  # Keep track of valid images (not skipped)

    for epoch in range(num_of_epochs):
        print(f'@ Epoch {epoch+1} for a target class: {target_class}')
        epoch_correct_count = 0
        epoch_valid_images = 0  # Count valid images for this epoch
        batch_i=1
        batch_loss = 0

        for batch in dataloader:

            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)

            # Filter out images with true_label == target_class
            valid_indices = true_labels != target_class
            if not valid_indices.any():
                print(f"     Batch {batch_i} skipped: all images have true label == target class.")
                batch_i += 1
                continue  # Skip this batch if all images are to be excluded

            images = images[valid_indices]
            true_labels = true_labels[valid_indices]
            batch_size = images.shape[0]
            epoch_valid_images += batch_size
            # total_valid_images += batch_size

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
            batch_asr = correct / batch_size
            batch_loss += loss.item()


            logger.log_batch_metrics(iter, epoch+1, loss.item(), batch_asr)

            batch_i+=1
            
            # epoch_asr += batch_asr
            epoch_correct_count += correct

        avg_epoch_asr = epoch_correct_count / epoch_valid_images
        avg_epoch_loss = batch_loss / (batch_i - 1) if batch_i > 1 else 0
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        logger.log_epoch_metrics(iter, avg_epoch_loss, avg_epoch_asr)
      
        print(f"Epoch [{epoch+1}/{num_of_epochs}], Avg loss: {avg_epoch_loss}, Avg ASR: {avg_epoch_asr * 100:.2f}%")

        generator_name = f'generator_epoch_{epoch + 1}_iter_{iter}'
        save_generator(generator_name, generator, checkpoint_dir)
        

    print(f'\n\nAverage ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% \n')



def start_iteration(device, attack_mode, patch_size, discriminators, dataloader, classes, target_class, checkpoint_dir, num_of_epochs, num_of_patches, logger):
    attack_type = attack_mode.split('_')
    if attack_type[0] == 'gpatch':
        deployer = Deployer()
    else:
        deployer = DeployerMini(num_of_patches, critical_points=int(attack_type[1]))

    # results_list = []  # Store results from all 5 generators
    results_dict = {}
    # fixed_noise = torch.randn(1, 100, 1, 1).to(device)
    fixed_noises = [torch.randn(1, 100, 1, 1).to(device) for _ in range(2)]

    for i in range(2):  # Train generators separately
        iter = i+1
        print(f'{"-"*30} Training Generator {iter}/5 {"-"*30}')

        generator = Generator(patch_size).to(device)
        generator.train()

        optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))   ## try different values
        target_class = torch.tensor(target_class, device=device)
        gan_attack(iter, device, generator, optimizer, deployer, discriminators, dataloader, classes, target_class, num_of_epochs, checkpoint_dir, logger)
        print(f'{"-"*26} Finished Training Generator {iter}/5 {"-"*26}')

    # for noise_i, fixed_noise in enumerate(fixed_noises):
        results = evaluate_saved_generators(iter, target_class, checkpoint_dir, fixed_noises, patch_size, dataloader, deployer, discriminators, device, logger)
        results_dict[iter] = results

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--checkpoint_folder_path', help='Path to a folder where generators will be saved.', default='./checkpoints')
    parser.add_argument('--attack_mode', choices=['gpatch', 'mini'], default='gpatch')
    parser.add_argument('--num_of_patches', type=int, help='Number of patches. 1 for G-patch attack, and more than 1 for mini-patch attack', default=1)
    parser.add_argument('--transfer_mode', choices=['src-tar', 'ensemble', 'cross-val'],  
                        help='Choose the transferability approach: src-tar (for source-to-target), ensemble, or cross-val (for cross-validation)', default='src-tar')
    parser.add_argument('--training_models', type=str, help='Name(s) of the models the generator will be trained on.')
    parser.add_argument('--target_models', type=str)
    parser.add_argument('--patch_size', type=int, help='Size of the adversarial patch', default=64)
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--num_of_train_classes', type=int, help='Number of (random) classes to train the generator on', default=100)
    parser.add_argument('--target_class', type=int, help='Target class to misclassify images as', default=932)
    parser.add_argument('--mini_type', type=int, default=0, help='Type of mini-attack: 0 is for normal, 1&2 are for critical, but crosspoints and within the patches correspondingly.' )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_folder_path
    transfer_mode = args.transfer_mode
    training_model_names = args.training_models.split(' ') if args.training_models else None
    target_model_names = args.target_models.split(' ') if args.target_models else None
    intra_model_attack = False
    cross_validation = False

    patch_size = args.patch_size
    num_of_patches = args.num_of_patches
    attack_mode = args.attack_mode

    if attack_mode == 'mini':
        attack_mode += f'_{args.mini_type}'

    if training_model_names is None:
        raise ValueError('Please, specify training models.')  
    
    if transfer_mode == 'src-tar':
        if target_model_names is None:
            intra_model_attack = True
    else:
        if transfer_mode == 'cross-val':
            cross_validation = True
        if target_model_names is None:
            raise ValueError("For the ensemble/cross-validation attack please, specify target models.")
        

    discriminators = get_models(training_model_names, device)

    dataloader, classes = load_random_classes(args.image_folder_path, args.num_of_train_classes)
    num_of_classes = len(classes)
    num_of_epochs = args.epochs

    target_class = args.target_class 

    project_name = (
        f't>RPP {transfer_mode} ' +
        f'{attack_mode} ' +
        f'{num_of_patches}_patches ' + 
        f'={target_class}= ' +
        f'train-{",".join(training_model_names)} ' + 
        (f'target-{",".join(target_model_names)} ' if target_model_names else '')
    )

    print(f'PROJECT: {project_name}')
    
    # Initialize W&B
    logger = WandbLogger(project_name, target_class, {
        'epochs': num_of_epochs,
        'classes': num_of_classes,
        'target_class': target_class,
        'attack_mode': attack_mode,
        'patch_size': patch_size,
        'num_of_patches': num_of_patches,
        'transfer_mode': transfer_mode,
        'training_models': training_model_names,
        'target_models': target_model_names
    })


    results_dict = start_iteration(device, attack_mode, patch_size, discriminators, dataloader, classes, target_class, checkpoint_dir, num_of_epochs, num_of_patches, logger)
    
    if intra_model_attack: 
        target_models = discriminators
        target_model_names = training_model_names
    else:
        target_models = get_models(target_model_names, device)

    # for results in results_list:
    for iter, results in results_dict.items():
        # for result in results.items():
            # best_patch = result['best_patch']
        best_patches = results["best_patches"]
        for i, patch in enumerate(best_patches):
            noise_i = i+1
            test_best_patch(iter, noise_i, dataloader, attack_mode, target_models, target_model_names, training_model_names, patch, target_class, device=device, logger=logger, iteration=iter)

    logger.finalize()
