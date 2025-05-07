import os, argparse

import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from Random_Position_Patch.loss import AdversarialLoss
from Random_Position_Patch.Mini_Patches.deployer_mini import DeployerMini
from Random_Position_Patch.deployer import Deployer
from Random_Position_Patch.generator import Generator
from Random_Position_Patch.helper import save_generator, load_generator, load_checkpoints, load_classes, get_class_name, fetch_generators_from_wandb
from Random_Position_Patch.wandb_logger import WandbLogger

# from loss import AdversarialLoss
# from deployer import Deployer
# from Mini_Patches.deployer_mini import DeployerMini
# from generator import Generator
# from helper import save_generator, load_generator, load_checkpoints, load_classes, get_class_name, fetch_generators_from_wandb
# from wandb_logger import WandbLogger

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights, resnet101, ResNet101_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights

# === Load W&B API key from environment ===
api_key = os.getenv('WANDB_API_KEY')


def get_models(model_names, device):
    """
    Given a list of model names, load pretrained models and set them to eval mode.

    Args:
        model_names: list of string model names.
        device: 'cuda' or 'cpu'.

    Returns:
        List of pretrained PyTorch models.
    """
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
    """
    Evaluate a single adversarial patch on a target model not used during training.

    Args:
        training_model_names: names of models used in training.
        patch: adversarial patch to evaluate.
        patch_i: patch index (for logging).
        dataloader: dataset for evaluation.
        target_class: numerical label of the taarget class.
        deployer: patch deployment strategy.
        target_model: target model (with weights) for evaluation.
        target_model_name: name of the target model.
        device: 'cuda' or 'cpu'.
        logger: optional W&B logger.
    """
    misclassified_counts = 0
    image_i = 0
    total_valid_images = 0  # Keep track of valid images (not skipped)

    with torch.no_grad():  # Disable gradients for evaluation
        for batch in dataloader:
            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)
            
            # Filter out images with true_label == target_class
            valid_indices = true_labels != target_class
            if not valid_indices.any():
                continue  # skip this batch if all images should be excluded

            images = images[valid_indices]
            true_labels = true_labels[valid_indices]

            batch_size = images.shape[0]       # Update batch size
            total_valid_images += batch_size   # Update the number of valid images

            # Deploying and evaluating patch success
            for i in range(batch_size):
                misclassified = False
                modified_image = deployer.deploy(patch, images[i])

                output = target_model(modified_image.unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)       
                if predicted.item() == target_class:
                    misclassified_counts += 1
                    misclassified = True

                if logger and image_i % 500 == 0:   # displaying one in every 500 modified images
                    logger.log_modified_image(patch_i, image_i, modified_image, misclassified, target_model_name)

                image_i +=1

                if device=='cuda':
                    del image, modified_image, output, predicted
                    torch.cuda.empty_cache()

            if device=='cuda':
                del images, true_labels
                torch.cuda.empty_cache()

  
        print(f'The generated adversarial patch on target model had {misclassified_counts} misclassifications')    
        asr = misclassified_counts / total_valid_images
        print(f'ASR for target model: {asr * 100:.2f}%')    

       
        if logger:
            logger.log_target_model_results(
                training_model_names, patch_i, target_model_name, misclassified_counts, total_valid_images, asr
            )
    
    return misclassified_counts


def evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device):
    """
    Evaluates the performance (ASR) of a given adversarial patch on a dataset.

    Args:
        patch: the adversarial patch to be applied to images.
        dataloader: dataLoader providing input images.
        model_list: list of models (ensemble) used for classification.
        device: 'cuda' or 'cpu'.
        target_class_y_prime: The target class label the attack aims to induce.
        logger (optional): a W&B  logger for logging metrics.

    Returns:
        the attack success rate over the dataset.
    """
    total_correct_count = 0
    total_valid_images = 0

    with torch.no_grad():  
        for batch in dataloader:
            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)
                    
            # Filter out images with true_label == target_class
            valid_indices = true_labels != target_class
            if not valid_indices.any():
                continue  
            images = images[valid_indices]
            true_labels = true_labels[valid_indices]

            batch_size = images.shape[0]      # Update batch size
            total_valid_images += batch_size  # Update the number of valid images
            
            # Deploying patches
            adv_images = []
            for idx, image in enumerate(images):
                adv_image = deployer.deploy(patch, image)
                adv_images.append(adv_image)

            adv_images = torch.stack(adv_images).to(device)


            # === Soft Voting ===
            sum_probs = None
            for discriminator in discriminators:
                output = discriminator(adv_images)
                probs = torch.softmax(output, dim=1)  
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs

            ensemble_predicted = torch.argmax(sum_probs, dim=1)  # Final prediction via soft-voting

            correct = (ensemble_predicted == target_class).sum().item()  # Number of "correctly misclassified" images
            total_correct_count += correct   

            if device=='cuda':
                del adv_images, images
                torch.cuda.empty_cache()

        total_asr = total_correct_count / total_valid_images

    return total_asr



def evaluate_saved_generators(target_class, checkpoint_dir, fixed_noises, patch_size, dataloader, deployer, discriminators, device, logger):
    """ 
    Load saved generators of each training epoch (in the checkpoint folder). 
    Identify the best-performing generator and save it as a separate file.       

    Args:
        generator_class: the generator class used to load the saved models.
        generator_path_list: list of file paths to saved generator state_dicts.
        dataloader: dataLoader providing input images.
        model_list: llist of models used to classify patched images.
        device: 'cuda' or 'cpu'.
        target_class_y_prime: target class that the generator aims to fool the models into predicting.

    """

    best_avg_asr = 0
    best_epoch = -1
    best_patches = []
    best_generator = None

    last_checkpoint  = None  # Track the checked generators

    # iterate over subfolders in the checkpoints directory
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    checkpoints = load_checkpoints(checkpoint_files)
    
    results = {}
    print(f'\n{"="*30} Evaluating generators {"="*30}')

    for epoch, checkpoint_file in checkpoints:
            
        gen = Generator(patch_size).to(device)
        generator = load_generator(gen, checkpoint_file)

        total_asr = 0
        patches = []
        for j, fixed_noise in enumerate(fixed_noises):
            noise_i = j+1
            # Generate a patch and evaluate its performance
            patch = generator(fixed_noise).detach().squeeze(0)
            asr = evaluate_patch(patch, dataloader, target_class, deployer, discriminators, device)
            print(f"    ->  Epoch {epoch}, Noise #{noise_i}  -<>- Patch ASR: {asr * 100:.2f}%")

            total_asr += asr
            patches.append(patch)  # Save the patch generated from a corresponding fixed random noise vector
            if logger:
                logger.log_generator_evaluation(noise_i, epoch, asr)
                logger.log_patch_image(noise_i, epoch, patch)

        avg_asr = total_asr / len(fixed_noises)
        last_checkpoint = checkpoint_file   # Update the last checked generator

        if avg_asr > best_avg_asr:  # Update the best-performing generator, with its patches and ASR
            best_avg_asr = avg_asr
            best_epoch = epoch
            best_patches = patches
            best_generator = generator

        
    if best_epoch == -1:   # Every generator had 0 ASR, so we save the last checked generator. This is simply to not have an empty model.
        generator = Generator(patch_size).to(device)
        generator = load_generator(generator, last_checkpoint)
        save_generator("best_-1", generator, f'{checkpoint_dir}/best_generators') # Name convention to easily identify a generator with 0 ASR
    else:
        generator_name = f"best"
        save_generator(generator_name, best_generator, f'{checkpoint_dir}/best_generators')
        if logger:
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
    """
    Begin the training of a generator to produce adversarial patches.

    Args:
        device: computation device, e.g., 'cuda' or 'cpu'.
        generator: generator model that produces adversarial patches from noise.
        optimizer: optimizer used to update generator weights.
        deployer: mechanism responsible for applying patches to input images.
        discriminators: victim model(s) (discriminators) to be fooled by adversarial patches.
        dataloader: Dataloader providing batches of (image, true_label) pairs.
        target_class: numerical label of the target class the attack aims to force the model to predict.
        num_of_epochs: number of epochs for training.
        checkpoint_dir: directory to save generator checkpoints after each epoch.
        logger (optional): W&B logger for tracking training metrics.
    """    
    
    print(f'{"-"*30} Training Generator {"-"*30}')
    
    total_asr = 0    # Total ASR accumulated over all epochs

    for epoch in range(num_of_epochs):
        print(f'@ Epoch {epoch+1} for a target class: {target_class}')
        epoch_correct_count = 0   # Number of successful target class predictions in the epoch
        epoch_valid_images = 0    # Number of images excluding those with the same label as the target class

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
                continue  

            images = images[valid_indices]
            true_labels = true_labels[valid_indices]
            batch_size = images.shape[0]
            epoch_valid_images += batch_size    

            # Generate random noise for the generator
            noise = torch.randn(batch_size, input_dim, 1, 1).to(device)  # Every batch has its own unique random noise vector


            # === Deploying ===

            modified_images = []
            adv_patches = generator(noise)
            for i in range(batch_size):
                patch = adv_patches[i]
                modified_image = deployer.deploy(patch, images[i])
                modified_images.append(modified_image)
            modified_images = torch.stack(modified_images).to(device)

            # === Evaluate modified images using all discriminators ===
            outputs = []
            for discriminator in discriminators:
                output = discriminator(modified_images)
                outputs.append(output)

            target_class_y_prime = torch.full((batch_size,), target_class, dtype=torch.long).to(device)
            

            criterion = AdversarialLoss(target_class=target_class_y_prime).to(device)  # Compute adversarial loss 
            # Weighted loss for all the discriminators
            total_loss = sum([criterion(out) for out in outputs])
            loss = total_loss / len(discriminators)  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # === Compute ASR via ensemble soft-voting ===
            sum_probs = None
            for out in outputs:
                probs = torch.softmax(out, dim=1) 
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs
            ensemble_predicted = torch.argmax(sum_probs, dim=1)

            correct = (ensemble_predicted == target_class).sum().item()

            batch_asr = correct / batch_size
            batch_loss += loss.item()

            if logger: 
                # Log batch-level metrics
                logger.log_batch_metrics(epoch+1, loss.item(), batch_asr, batch_i)
            

            if device=='cuda':
                del images, true_labels, noise, adv_patches, modified_images, outputs, sum_probs, ensemble_predicted, target_class_y_prime, loss
                torch.cuda.empty_cache()
            batch_i+=1
            epoch_correct_count += correct

        avg_epoch_asr = epoch_correct_count / epoch_valid_images
        avg_epoch_loss = batch_loss / (batch_i - 1) if batch_i > 1 else 0
        total_asr += avg_epoch_asr

        if logger:
            # log epoch-level metrics to wandb
            logger.log_epoch_metrics(avg_epoch_loss, avg_epoch_asr, epoch)
      
        print(f"Epoch [{epoch+1}/{num_of_epochs}], Avg loss: {avg_epoch_loss}, Avg ASR: {avg_epoch_asr * 100:.2f}%")

        generator_name = f'generator_epoch_{epoch + 1}'
        save_generator(generator_name, generator, checkpoint_dir)
        
    print(f'\n\nAverage ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% \n')
    print(f'{"-"*26} Finished Training Generator {"-"*26}')




def start_training(device, attack_mode, patch_size, discriminators, dataloader, target_class, checkpoint_dir, num_of_epochs, num_of_patches, logger):
    """
    Initialize the training process.
    This function initializes the generator and correct deployer (for G-Patches and Mini-Patches)
    """
    
    attack_type = attack_mode.split('_')
    if attack_type[0] == 'gpatch':
        deployer = Deployer()
    else:  # Mini-Patch 
        deployer = DeployerMini(num_of_patches, critical_points=int(attack_type[1]))


    generator = Generator(patch_size).to(device)
    generator.train()

    optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))  
    target_class = torch.tensor(target_class, device=device)
    # Start the actual training, and save generators after each epoch
    gan_attack(device, generator, optimizer, deployer, discriminators, dataloader, target_class, num_of_epochs, checkpoint_dir, logger)

    # Load the fixed random noise vectors for evaluation
    fixed_noises = torch.load("./Random_Position_Patch/fixed_noises.pt")
    fixed_noises = [noise.to(device) for noise in fixed_noises]
    # Evaluate each generator using these noises
    results = evaluate_saved_generators(target_class, checkpoint_dir, fixed_noises, patch_size, dataloader, deployer, discriminators, device, logger)

    return results



def start_testing(train_model_names, device, dataloader, target_class, num_of_patches, patches, target_models, target_model_names, attack_mode, logger):
    """
    Initialize the testing process.
    This function initializes the generator and correct deployer (for G-Patches and Mini-Patches)
    """
    
    attack_type = attack_mode.split('_')
    print(attack_type)
    if attack_type[0] == 'gpatch':
        deployer = Deployer()
    else: # Mini-Patch
        deployer = DeployerMini(num_of_patches, critical_points=int(attack_type[1]))


    # Test each generator on a new random noise vector (not using the fixed vectors used during evaluation)
    for iter, generator in patches.items():
        noise = torch.randn(1, 100, 1, 1).to(device)
        patch = generator(noise).detach().squeeze(0)
        
        if logger:
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
    parser.add_argument('--wandb_entity', help='Entity name of the users WANDB account')
    parser.add_argument('--local_generator', help='Path to the locally saved pre-trained generator, if not loading from W&B')
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

    discriminators = get_models(training_model_names, device)

    dataloader = load_classes(args.image_folder_path)

    if attack_mode == "gpatch":
        attack_mode = f'{args.attack_mode}_{patch_size}'

    logger = None
    wandb_entity = args.wandb_entity   # Set up the W&B project configuration, if the entity is given
    if wandb_entity:
        project_name = (
            f'F {run_mode} ' +
            f'{attack_mode} ' +
            f'={target_class}= ' +  
            f'{",".join(training_model_names)}' + 
            (f' > {",".join(target_model_names)}' if target_model_names else '')
        )

        logger = WandbLogger(wandb_entity, run_mode, project_name, target_class, config={
            'run_mode' : run_mode,
            'attack_mode': attack_mode,
            'target_class': target_class,
            'patch_size': patch_size,
            'num_of_patches': num_of_patches,
            'training_models': training_model_names,
            'target_models': target_model_names
        })


    if run_mode == 'train':
        start_training(device, attack_mode, patch_size, discriminators, dataloader, target_class, checkpoint_dir, num_of_epochs, num_of_patches, logger)
    else:
        target_models = get_models(target_model_names, device)

        if args.local_generator:    # Loading generator from a locally saved file, so we don't need to download it from W&B
            local_path = args.local_generator
            assert os.path.exists(local_path), f"Could not locate the generator {local_path}" 
            generator = Generator(patch_size).to(device)
            generator.load_state_dict(torch.load(local_path, map_location=device))
            generator.eval()

            generators={0 : generator}

        else:     # Local generator path wasn't provided, so we try to access the train project of corresponding configuration and download best performing generators(from each run)
            assert logger is not None, f"For testing either a path to local generator, or W&B entity should be provided." 
            train_project_name = f'F train ' + attack_mode + f' ={target_class}= ' + f'{",".join(training_model_names)}'
            generators = fetch_generators_from_wandb(wandb_entity, Generator, train_project_name, patch_size=patch_size)

        start_testing(args.training_models, device, dataloader, target_class, num_of_patches, generators, target_models, target_model_names, attack_mode, logger)
    
    if logger: 
        logger.finalize()