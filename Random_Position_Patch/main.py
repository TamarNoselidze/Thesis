import os, argparse

import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from loss import AdversarialLoss
from deployer import Deployer
from Mini_Patches.deployer_mini import DeployerMini
from generator import Generator
from helper import save_generator, load_generator, load_checkpoints, load_random_classes, get_class_name, fetch_generators_from_wandb
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



        
def test_best_patch(training_model_names, patch, patch_i, dataloader, target_class, deployer, target_model, target_model_name, device, logger):
    # print(f'---------------------------- tc: {target_class}')
    # print(type(target_class))
    misclassified_counts = 0
    image_i = 0
    total_valid_images = 0  # Keep track of valid images (not skipped)

    with torch.no_grad():  # Disable gradients for evaluation
        for batch in dataloader:
            images, true_labels, indices = batch
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
            # print(f'plus {batch_size}')
            # print(f'so total is: {total_valid_images}')

            for i in range(batch_size):
                misclassified = False
                modified_image = deployer.deploy(patch, images[i])
                # for model, name in zip(target_models, target_model_names):
                output = target_model(modified_image.unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)       
                if predicted.item() == target_class:
                    misclassified_counts += 1
                    misclassified = True

                original_index = indices[valid_indices][i]  

                if image_i % 200 == 0:   # displaying one in every 200 modified images

                    logger.log_modified_image(patch_i, image_i, modified_image, misclassified, true_labels[original_index], target_model_name)

                image_i +=1

  
        # print(f'Target class: "{get_class_name(target_class)}" ({target_class})')
        print(f'The generated adversarial patch on target model had {misclassified_counts} misclassifications')    
        asr = misclassified_counts / total_valid_images
        print(f'ASR for target model: {asr * 100:.2f}%')    

        # Log to wandb
        logger.log_target_model_results(
            training_model_names, patch_i, target_model_name, misclassified_counts, total_valid_images, asr
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
            for idx, image in enumerate(images):
                adv_image = deployer.deploy(patch, image)
                adv_images.append(adv_image)

            adv_images = torch.stack(adv_images).to(device)


            # === Majority Voting ===
            # outputs = []
            # for discriminator in discriminators:
            #     output = discriminator(adv_images)
            #     outputs.append(output)

            # total_predicted = []
            # for out in outputs:
            #     _, predicted = torch.max(out.data, 1)
            #     total_predicted.append(predicted.cpu())

            # correct_counts = torch.zeros(batch_size).to(device)
            # for predicted in total_predicted:
            #     correct_counts += (predicted.to(device) == target_class).float()

            # majority_threshold = len(total_predicted) // 2
            # correct = (correct_counts > majority_threshold).sum().item()


            # === Soft Voting ===
            sum_probs = None
            for discriminator in discriminators:
                output = discriminator(adv_images)
                probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs

            ensemble_predicted = torch.argmax(sum_probs, dim=1)
            # Count how many were predicted as the target class (i.e., "successful attack")
            correct = (ensemble_predicted == target_class).sum().item()
            total_correct_count += correct

        total_asr = total_correct_count / total_valid_images

    return total_asr



def evaluate_saved_generators(target_class, checkpoint_dir, fixed_noises, patch_size, dataloader, deployer, discriminators, device, logger):
    best_avg_asr = 0
    best_epoch = -1
    best_patches = []
    best_generator = None

    last_checkpoint  = None
    # Iterate over subfolders in the checkpoints directory
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    results = {}

    print(f'\n{"="*30} Evaluating generators {"="*30}')
    checkpoints = load_checkpoints(checkpoint_files)

    for epoch, checkpoint_file in checkpoints:
            
        gen = Generator(patch_size).to(device)
        generator = load_generator(gen, checkpoint_file)

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

            logger.log_generator_evaluation(noise_i, epoch, asr)
            logger.log_patch_image(noise_i, epoch, patch)

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
        save_generator("best_-1", generator, f'{checkpoint_dir}/best_generators')
    else:
        generator_name = f"best"
        save_generator(generator_name, best_generator, f'{checkpoint_dir}/best_generators')
        logger.log_best_generator(generator_name, best_epoch, best_avg_asr)
        for noise_i, patch in enumerate(best_patches):
            logger.log_best_patch(noise_i, patch)

    print(f"Best generator found at epoch {best_epoch} with avg ASR: {best_avg_asr * 100:.2f}%")

    results = {
        'best_avg_asr': best_avg_asr,
        'best_epoch': best_epoch,
        'best_patches': best_patches,
    }

    print("="*100)

    return results



def gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, target_class, num_of_epochs, checkpoint_dir, logger, input_dim=100):
    print(f'{"-"*30} Training Generator {"-"*30}')
    
    total_asr = 0
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

            # soft voting
            sum_probs = None
            for out in outputs:
                probs = torch.softmax(out, dim=1)  # Convert logits to probabilities
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs
            ensemble_predicted = torch.argmax(sum_probs, dim=1)

            correct = (ensemble_predicted == target_class).sum().item()


            # # majority voting (more than half of the discriminators)
            # total_predicted = []
            # for out in outputs:
            #     _, predicted = torch.max(out.data, 1)
            #     total_predicted.append(predicted.cpu())

            # correct_counts = torch.zeros(batch_size).to(device)
            # for predicted in total_predicted:
            #     correct_counts += (predicted.to(device) == target_class).float()

            # majority_threshold = len(total_predicted) // 2
            # correct = (correct_counts > majority_threshold).sum().item()


            batch_asr = correct / batch_size
            batch_loss += loss.item()

            logger.log_batch_metrics(epoch+1, loss.item(), batch_asr, batch_i)
            batch_i+=1
            epoch_correct_count += correct

        avg_epoch_asr = epoch_correct_count / epoch_valid_images
        avg_epoch_loss = batch_loss / (batch_i - 1) if batch_i > 1 else 0
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        logger.log_epoch_metrics(avg_epoch_loss, avg_epoch_asr, epoch)
      
        print(f"Epoch [{epoch+1}/{num_of_epochs}], Avg loss: {avg_epoch_loss}, Avg ASR: {avg_epoch_asr * 100:.2f}%")

        generator_name = f'generator_epoch_{epoch + 1}'
        save_generator(generator_name, generator, checkpoint_dir)
        
    print(f'\n\nAverage ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% \n')
    print(f'{"-"*26} Finished Training Generator {"-"*26}')




def start_training(device, attack_mode, patch_size, discriminators, dataloader, target_class, checkpoint_dir, num_of_epochs, num_of_patches, logger):
    attack_type = attack_mode.split('_')
    if attack_type[0] == 'gpatch':
        deployer = Deployer()
    else:
        deployer = DeployerMini(num_of_patches, critical_points=int(attack_type[1]))

    fixed_noises = torch.load("fixed_noises.pt")
    fixed_noises = [noise.to(device) for noise in fixed_noises]

    generator = Generator(patch_size).to(device)
    generator.train()

    optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))   ## try different values
    target_class = torch.tensor(target_class, device=device)
    gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, target_class, num_of_epochs, checkpoint_dir, logger)

    # for noise_i, fixed_noise in enumerate(fixed_noises):
    results = evaluate_saved_generators(target_class, checkpoint_dir, fixed_noises, patch_size, dataloader, deployer, discriminators, device, logger)

    return results

def start_testing(train_model_names, device, dataloader, target_class, num_of_patches, patches, target_models, target_model_names, attack_mode, logger):
    attack_type = attack_mode.split('_')
    if attack_type[0] == 'gpatch':
        deployer = Deployer()
    else:
        deployer = DeployerMini(num_of_patches, critical_points=int(attack_type[1]))


    for iter, generator in patches.items():
        noise = torch.randn(1, 100, 1, 1).to(device)
        patch = generator(noise).detach().squeeze(0)
        logger.log_best_patch(f'{train_model_names}/iter {iter+1}', patch, testing=True)

        for target_model, target_model_name in zip(target_models, target_model_names):
            test_best_patch(train_model_names, patch, f'iter {iter+1}', dataloader, target_class, deployer, target_model, target_model_name, device, logger)
           
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--checkpoint_folder_path', help='Path to a folder where generators will be saved.', default='./checkpoints')
    parser.add_argument('--run_mode', choices=['train', 'test'])
    parser.add_argument('--attack_mode', choices=['gpatch', 'mini_0', 'mini_1', 'mini_2'], default='gpatch')
    parser.add_argument('--training_models', type=str, help='Name(s) of the models the generator will be trained on.')
    parser.add_argument('--target_models', type=str)
    parser.add_argument('--target_class', type=int, help='Target class to misclassify images as', default=932)
    parser.add_argument('--patch_size', type=int, help='Size of the adversarial patch', default=64)
    parser.add_argument('--num_of_patches', type=int, help='Number of patches. 1 for G-patch attack, and more than 1 for mini-patch attack', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=40)
    parser.add_argument('--num_of_train_classes', type=int, help='Number of (random) classes to train the generator on', default=100)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_folder_path
    run_mode = args.run_mode
    attack_mode = args.attack_mode
    training_model_names = args.training_models.split(' ') if args.training_models else None
    target_model_names = args.target_models.split(' ') if args.target_models else None
    target_class = args.target_class 
    patch_size = args.patch_size
    num_of_patches = args.num_of_patches
    num_of_epochs = args.epochs

    att = f'{attack_mode} ' if patch_size!=80 else f'{attack_mode}-80 '


    project_name = (
        f'F {run_mode} ' +
        att +
        # f'{num_of_patches}_patches ' + 
        f'={target_class}= ' +  
        f' {",".join(training_model_names)} ' + 
        (f' > {",".join(target_model_names)}' if target_model_names else '')
    )

    logger = WandbLogger(run_mode, project_name, target_class, {
        'run_mode' : run_mode,
        'epochs': num_of_epochs,
        # 'classes': num_of_classes,
        'target_class': target_class,
        'attack_mode': attack_mode,
        'patch_size': patch_size,
        'num_of_patches': num_of_patches,
        'training_models': training_model_names,
        'target_models': target_model_names
    })

    discriminators = get_models(training_model_names, device)

    dataloader, classes = load_random_classes(args.image_folder_path, args.num_of_train_classes)
    num_of_classes = len(classes)

    print(f'PROJECT: {project_name}')
    # print(f'CLASSES: {classes}')
    if run_mode == 'train':
        start_training(device, attack_mode, patch_size, discriminators, dataloader, target_class, checkpoint_dir, num_of_epochs, num_of_patches, logger)
    else:
        train_project_name = f'F train ' + att + f'={target_class}= ' + f' {",".join(training_model_names)} '
        print(f'train project name: {train_project_name}')
        target_models = get_models(target_model_names, device)

        # generators = fetch_generators_from_wandb(lambda: Generator(patch_size), train_project_name, patch_size=patch_size)
        generators = fetch_generators_from_wandb(Generator, train_project_name, patch_size=patch_size)
        start_testing(args.training_models, device, dataloader, target_class, num_of_patches, generators, target_models, target_model_names, attack_mode, logger)
        
    
    logger.finalize()
