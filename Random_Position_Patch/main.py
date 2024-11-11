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

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights

# from dotenv import load_dotenv


# seed = 42  #fixed seed for reproducibility
# random.seed(seed)
# numpy.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


# load_dotenv(os.path.join(os.getenv('SCRATCHDIR', '.'), '.env'))
api_key = os.getenv('WANDB_API_KEY')
# print(f"WANDB_API_KEY: {api_key}")  


def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor, contrast=0.3)
    adjusted_image = brightness_transform(image)
    return adjusted_image

def start(device, generator, optimizer, deployer, discriminators, dataloader, num_of_classes, num_of_epochs=40, brightness_factor=None, color_transfer=None, input_dim=100):

    best_epoch_asr = 0  
    best_epoch_images = {}
    total_asr = 0

    possible_targets = [random.randint(800, 900) for _ in range(40)]

    generator.reset_weights()
    optimizer = optim.Adam(generator.parameters(), lr=0.001)
    for epoch in range(num_of_epochs):

        print(f'@ Processing epoch {epoch+1}')
        target_class = possible_targets[epoch]
        target_class = 813

        epoch_total_asr = 0
        # best_epoch_patch = None   
        best_batch_asr = 0
        # best_batch_images = {}

        epoch_images = {}

        temp_count=1
        for batch in dataloader:

            print(f'@  working on batch {temp_count}')

            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)

            batch_size = images.shape[0]   # might change for the last batch
            noise = torch.randn(batch_size, input_dim, 1, 1).to(device)
            modified_images = []
            adv_patches = generator(noise)

            # deploying 
            for i in range(batch_size):
                modified_image = deployer.deploy(adv_patches[i], images[i])
                modified_images.append(modified_image)

            modified_images = torch.stack(modified_images).to(device)

            # multiple discriminators
            outputs = []
            for discriminator in discriminators:
                output = discriminator(modified_images)
                outputs.append(output)
            # print(f"   Discriminator output: {outputs.data.cpu()}")

            
            # target_class_y_prime = torch.randint(0, 1000, (batch_size,)).to(device)
            target_class_y_prime = torch.full((batch_size,), target_class, dtype=torch.long).to(device)
            # target_class_y_prime[target_class_y_prime == true_labels] = (target_class_y_prime[target_class_y_prime == true_labels] + 1) % num_of_classes
            
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
                # print(type(predicted))
                total_predicted.append(predicted.cpu())
            print(total_predicted)

            # correct = sum([(predicted == target_class).sum().item() for predicted in total_predicted])
            correct_counts = torch.zeros(batch_size).to(target_class.device)
            for predicted in total_predicted:
                correct_counts += (predicted == target_class).float()

            # majority vote (more than half of the discriminators)
            majority_threshold = len(total_predicted) // 2
            correct = (correct_counts > majority_threshold).sum().item()

            print(f"     Loss: {loss.item()}")
            print(f"     True labels: {true_labels.cpu()}")
            print(f"     Predicted labels: {total_predicted}")
            print(f"     Target class is: {target_class}")
            print(f"     Correctly misclassified: {correct}")

            batch_asr = correct / batch_size
            print(f'@    this batch (number {temp_count}) has ASR: {batch_asr}')
            print(f'@    best_batch_asr so far: {best_batch_asr}')
            temp_count+=1


            # Log batch metrics to W&B
            wandb.log({
                'batch_loss': loss.item(),
                'batch_asr': batch_asr,
                'epoch': epoch + 1
            })
            
            batch_images = {}
            for i in range(batch_size):
                batch_images[images[i].cpu()] = modified_images[i].cpu()

            epoch_images.update(batch_images)
            epoch_total_asr += batch_asr

            if batch_asr > best_batch_asr:
                best_batch_asr = batch_asr
                print(f'@       current batch asr is better than best_batch_asr ({best_batch_asr}) so we are updating its value.')

        print(f'@  best batch ASR in this epoch is: {best_batch_asr}')
        print(f'   best batch ASR in the previous epoch was: {best_epoch_asr}')

        if best_batch_asr > best_epoch_asr:   # if the best batch outperforms the best batch in prev epoch, update the best_epoch_images
            best_epoch_asr = best_batch_asr
            print(f'   we changed best_epoch_asr value to: {best_batch_asr}')
            best_epoch_images.clear()
            best_epoch_images = epoch_images

        avg_epoch_asr = epoch_total_asr / len(dataloader)
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        wandb.log({
            'epoch_avg_asr': avg_epoch_asr,
            'epoch_best_batch_asr': best_batch_asr,
            'epoch': epoch + 1
        })

        print(f"Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")
        print(f'|___ ASR of the best performing batch: {best_batch_asr * 100:.2f}%')

    print(f'\n\n-- Average ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% --\n\n')
    print(f'\n\nResults saved.\nBest ASR achieved over {num_of_epochs} epochs: {best_epoch_asr * 100:.2f}%')

    return best_epoch_images


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
        # print(f'displayed: original and modified of {image_key}')

       
    # wandb.finish()


def transfer_to_target_models(models, images, target_class, device):
    total_images = len(images)
    
    for model in models:
        correctly_misclassified = 0
        with torch.no_grad():  # Disable gradients for evaluation
            for original, modified in images.items():
                # output = model(modified.unsqueeze(0))  
                output = model(modified.unsqueeze(0).to(device))
                _, predicted = torch.max(output.data, 1)
                
                if predicted.item() == target_class:
                    correctly_misclassified += 1
        
        # Calculate ASR (Attack Success Rate)
        avg_asr = (correctly_misclassified / total_images) * 100
        print(f"Model: {model.__class__.__name__}")
        print(f"Correctly misclassified: {correctly_misclassified}/{total_images}")
        print(f"Avg ASR: {avg_asr:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--transfer_mode', choices=['source-to-target', 'ensemble', 'cross-validation'], 
                        help='Choose the transferability approach: source-to-target, ensemble, or cross-validation', default='source-to-target')
    parser.add_argument('--training_models', nargs='+', 
                        choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'],
                        help='List of models for ensemble or cross-validation. Use space to separate models, e.g., --model_list resnet50 vit_b_16 swin_b')
    parser.add_argument('--target_models', nargs='+', 
                    choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'],
                    help='List of models to attack and validate trained attack on. Use space to separate models, e.g., --model_list resnet50 vit_b_16 swin_b')
    parser.add_argument('--patch_size', choices=['48', '64', '80'], help='Size of the adversarial patch', default=64)
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--brightness', help='Brightness level for the patch')
    parser.add_argument('--color_transfer', help='Color transfer value for the patch')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    transfer_mode = args.transfer_mode
    training_model_names = args.training_models
    target_model_names = args.target_models
    intra_model_attack = False
    cross_validation = False
    if training_model_names is None:
        raise ValueError('You should specify training models.')  
    
    if transfer_mode == 'source-to-target':
        if target_model_names is None:
            intra_model_attack = True
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

    input_dim = 100  
    output_dim = 3      
    k = 0.5
    
    generator = Generator(patch_size, input_dim, output_dim, k).to(device)
    generator.train()
    deployer = Deployer()

    discriminators = get_models(training_model_names, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(args.image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_of_classes = len(dataset.classes)
    print(f'NUMBER OF CLASSES: {num_of_classes}')
    num_of_epochs = int(args.epochs) if args.epochs else 40

    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values

    print(f'Using device: {device}')
    print(f"Generator device: {next(generator.parameters()).device}")
    # print(f"Discriminator device: {next(discriminator.parameters()).device}")
    
    project_name = f'Random Position Patch_tmp_{ "(br "+str(brightness_factor)+")" if brightness_factor else ""}{ "_col-tr"+color_transfer if color_transfer else ""}'


    # Initialize W&B
    wandb.init(project=project_name, entity='takonoselidze-charles-university', config={
        'epochs': num_of_epochs,
        'classes' : num_of_classes,
        'input_dim': input_dim
    })

    best_ep_imgs = start(device, generator, optimizer, deployer, discriminators, dataloader, num_of_classes, num_of_epochs, brightness_factor, color_transfer)
    display_images(best_ep_imgs)

    if target_model_names is not None:
        target_models = get_models(target_model_names, device)
        transfer_to_target_models(target_models, best_ep_imgs, target_class=813, device=device)

    wandb.finish()
