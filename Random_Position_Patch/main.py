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



# seed = 42  #fixed seed for reproducibility
# random.seed(seed)
# numpy.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


api_key = os.getenv('WANDB_API_KEY')


def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor, contrast=0.3)
    adjusted_image = brightness_transform(image)
    return adjusted_image

def start(device, generator, optimizer, deployer, discriminators, dataloader, classes, num_of_epochs=40, brightness_factor=None, color_transfer=None, input_dim=100):

    best_epoch_asr = 0  
    # epoch_images = {}
    epoch_patches = []
    target_classes = []
    total_asr = 0

    possible_targets = [random.randint(800, 900) for _ in range(40)]
    # possible_targets = random.sample([i for i in range(1000) if i not in classes], 40)

    optimizer = optim.Adam(generator.parameters(), lr=0.001)

    for epoch in range(num_of_epochs):

        print(f'@ Processing epoch {epoch+1}')
        # generator.reset_weights()
        target_class = possible_targets[epoch]

        epoch_total_asr = 0
        best_epoch_patch = None   
        best_batch_asr = 0
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
            patch = adv_patches[0]  # one patch for all images in the batch
            # patch = adv_patches[random.randint(0, batch_size-1)]
            for i in range(batch_size):
                # patch = adv_patches[i]
                # print(f'----------------- The patch is:\n{patch}')
                modified_image = deployer.deploy(patch, images[i])
                modified_images.append(modified_image)

                epoch_images.update({images[i] : modified_image})

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
            # print(total_predicted)

            # correct = sum([(predicted == target_class).sum().item() for predicted in total_predicted])
            target_class = torch.tensor(target_class, device=device)
            correct_counts = torch.zeros(batch_size).to(target_class.device)
            for predicted in total_predicted:
                correct_counts += (predicted.to(device) == target_class).float()

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
            
            # batch_images = {}
            # for i in range(batch_size):
            #     batch_images[images[i].cpu()] = modified_images[i].cpu()

            # epoch_images.update(batch_images)
            epoch_total_asr += batch_asr

            if batch_asr > best_batch_asr:
                best_batch_asr = batch_asr
                best_epoch_patch = patch
                print(f'@       current batch asr is better than best_batch_asr ({best_batch_asr}) so we are updating its value.')

            if best_epoch_patch is None:
                best_epoch_patch = patch   # just to assign something

        print(f'@  best batch ASR in this epoch is: {best_batch_asr}')
        print(f'   best batch ASR in the previous epoch was: {best_epoch_asr}')
        
        
        if best_batch_asr > best_epoch_asr:   # if the best batch outperforms the best batch in prev epoch, update the best_epoch_images
            best_epoch_asr = best_batch_asr
            print(f'   we changed best_epoch_asr value to: {best_batch_asr}')
            # best_epoch_images.clear()
            # best_epoch_images = epoch_images

        epoch_patches.append(best_epoch_patch)
        target_classes.append(target_class)
        avg_epoch_asr = epoch_total_asr / len(dataloader)
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        wandb.log({
            'epoch_avg_asr': avg_epoch_asr,
            'epoch_best_batch_asr': best_batch_asr,
            'epoch': epoch + 1
        })

        # i=1
        # for original, modified in epoch_images.items():
        #     image_key = f'epoch_{epoch+1}_img_{i}'
            
        #     wandb.log({
        #     image_key: [wandb.Image(original.cpu(), caption=f"Original Image {i} (epoch {epoch+1})"), 
        #                 wandb.Image(modified.cpu(), caption=f"Modified Image {i} (epoch {epoch+1})")]
        #     })
        #     i+=1    

        patch_key = f'epoch_{epoch+1}_best_patch'
        wandb.log({patch_key : wandb.Image(best_epoch_patch.cpu(), caption=f'Best performing patch for target class {target_class}')})

        print(f"Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")
        print(f'|___ ASR of the best performing batch: {best_batch_asr * 100:.2f}%')

    print(f'\n\n-- Average ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% --\n\n')
    print(f'\n\nResults saved.\nBest ASR achieved over {num_of_epochs} epochs: {best_epoch_asr * 100:.2f}%')

    return epoch_patches, target_classes


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


def load_random_classes(image_folder_path, num_of_classses):
    # Step 1: Load the full dataset to access `dataset.classes`
    dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    
    # Step 2: Randomly select 20 class names from the `dataset.classes`
    selected_classes = set(random.sample(dataset.classes, num_of_classses))
    print(f'SELECTED RANDOM CLASSES: {selected_classes}')
    
    # Step 3: Filter samples based on the selected class names
    dataset.samples = [(path, target) for path, target in dataset.samples if dataset.classes[target] in selected_classes]
    
    # Step 4: Update dataset.targets to match the filtered samples
    dataset.targets = [target for _, target in dataset.samples]

    # Step 5: Update dataset.classes and dataset.class_to_idx to reflect the selected classes
    dataset.classes = sorted(selected_classes)
    dataset.class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

    # Step 6: Create a DataLoader for the filtered dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader, dataset.classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='./imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--transfer_mode', choices=['source-to-target', 'ensemble', 'cross-validation'], 
                        help='Choose the transferability approach: source-to-target, ensemble, or cross-validation', default='source-to-target')
    parser.add_argument('--training_models', type=str)
                        # nargs='+', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='List of training models')
    parser.add_argument('--target_models', type=str)
    # , nargs='+', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='List of target models')    
    parser.add_argument('--patch_size', choices=['48', '64', '80'], help='Size of the adversarial patch', default=64)
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--num_of_classes', type=int, help='Number of (random) classes to train the generator on', default=100)
    parser.add_argument('--brightness', help='Brightness level for the patch')
    parser.add_argument('--color_transfer', help='Color transfer value for the patch')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
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

    # dataset = datasets.ImageFolder(args.image_folder_path, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataloader, classes = load_random_classes(args.image_folder_path, args.num_of_classes)
    num_of_classes = len(classes)
    print(f'CLASSES: {classes}')
    print(f'IN TOTAL: {len(classes)}')
    num_of_epochs = int(args.epochs) if args.epochs else 40

    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values

    print(f'Using device: {device}')
    print(f"Generator device: {next(generator.parameters()).device}")
    

    project_name = (
        f'RPP_notreset 1st_patch{transfer_mode} ' +
        f'train-{",".join(training_model_names)} ' + 
        (f'target-{",".join(target_model_names)}' if target_model_names else '') +
        (f' (br {brightness_factor})' if brightness_factor else '') + 
        (f' _col-tr{color_transfer}' if color_transfer else '')
    )

    # Initialize W&B
    wandb.init(project=project_name, entity='takonoselidze-charles-university', config={
        'epochs': num_of_epochs,
        'classes' : num_of_classes,
        'input_dim': input_dim
    })

    epoch_patches, target_classes = start(device, generator, optimizer, deployer, discriminators, dataloader, classes, num_of_epochs, brightness_factor, color_transfer)


    target_models = None
    if not intra_model_attack: 
    # if target_model_names is not None:
        target_models = get_models(target_model_names, device)
    test_best_patches(dataloader, deployer, discriminators, target_models, num_of_epochs, epoch_patches, target_classes, device)

    wandb.finish()
