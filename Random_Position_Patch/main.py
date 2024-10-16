import os, argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

from deployer import Deployer
from helper import save_image, save_patch
from loss import AdversarialLoss

from dotenv import load_dotenv


load_dotenv(os.path.join(os.getenv('SCRATCHDIR', '.'), '.env'))
api_key = os.getenv('WANDB_API_KEY')
# print(f"WANDB_API_KEY: {api_key}")  


def start(device, generator, deployer, discriminator, attack_type, dataloader, num_of_epochs=40, input_dim=100):
    # print(f'-------------- Attack type: {attack_type}')
    import wandb

    # Initialize W&B
    wandb.init(project='Random Position Patch Attacks', entity='takonoselidze-charles-university', config={
        'epochs': num_of_epochs,
        'attack_type': 'Including the image embeddings' if attack_type=='0' else 'Without the image embeddings',
        'input_dim': input_dim
    })

    num_classes = 200  
    best_epoch_asr = 0  
    best_epoch_images = {}
    total_asr = 0


    for epoch in range(num_of_epochs):

        print(f'@@ Processing epoch {epoch+1}')

        epoch_total_asr = 0
        # best_epoch_patch = None   
        best_batch_asr = 0
        best_batch_images = {}

        epoch_images = {}

        temp_count=1
        for batch in dataloader:
            # print(type(batch))
            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)

            batch_size = images.shape[0]   # might change for the last batch

            noise = torch.randn(batch_size, input_dim, 1, 1).to(device)
            modified_images = []
            adv_patches = None

            if attack_type == '0':
                adv_patches = generator(noise)
            else:
                # generating patches for each image
                adv_patches = []  # we need to store generated patch for each image

                for i in range(batch_size):
                    adv_patch = generator(noise[i].unsqueeze(0).to(device), images[i].unsqueeze(0).to(device))
                    adv_patches.append(adv_patch)

                adv_patches = torch.cat(adv_patches, dim=0).to(device)  # Stack generated patches

            # deploying 
            for i in range(batch_size):
                modified_image = deployer.deploy(adv_patches[i], images[i])
                modified_images.append(modified_image)

                # epoch_images[images[i]] = modified_image
                # epoch_images.update({images[i] : modified_image})

            
            modified_images = torch.stack(modified_images).to(device)

            outputs = discriminator(modified_images)
            
            target_class_y_prime = torch.randint(0, num_classes, (batch_size,)).to(device)
            target_class_y_prime[target_class_y_prime == true_labels] = (target_class_y_prime[target_class_y_prime == true_labels] + 1) % num_classes
            
            criterion = AdversarialLoss(target_class=target_class_y_prime).to(device)
            loss = criterion(outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            
            correct = (predicted == true_labels).sum().item()
            batch_asr = (batch_size - correct) / batch_size
            print(f'@@@@ batch {temp_count} has ASR: {batch_asr}')
            print(f'@@@@ best_batch_asr so far: {best_batch_asr}')
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
                print(f'@@@@@@ current batch asr is better than best_batch_asr ({best_batch_asr}) so we are updating its value.')
                # best_patch = adv_patch.clone().detach()  # Save the best performing patch
                # best_epoch_patches = adv_patches.clone

                # Save the original and modified images for the best batch
                # best_batch_images = {}
                # for i in range(batch_size):
                #     best_batch_images[images[i].cpu()] = modified_images[i].cpu()  

        print(f'@@ best batch ASR in this epoch is: {best_batch_asr}')
        print(f'   best batch ASR in the previous epoch was: {best_epoch_asr}')

        if best_batch_asr > best_epoch_asr:   # if the best batch outperforms the best batch in prev epoch, update the best_epoch_images
            best_epoch_asr = best_batch_asr
            print(f'   we changed best_epoch_asr value to: {best_batch_asr}')
            best_epoch_images = epoch_images

        avg_epoch_asr = epoch_total_asr / len(dataloader)
        total_asr += avg_epoch_asr

        # Log epoch-level metrics to W&B
        wandb.log({
            'epoch_avg_asr': avg_epoch_asr,
            'epoch_best_batch_asr': best_batch_asr,
            'epoch': epoch + 1
        })

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")
        print(f'|___ ASR of the best performing batch: {best_batch_asr * 100:.2f}%')

    print(f'\n\n-- Average ASR over all {num_epochs} epochs: {total_asr * 100:.2f}% --\n\n')


    pics_dir = os.path.join(os.getenv('SCRATCHDIR', '.'), 'pics')
    os.makedirs(pics_dir, exist_ok=True)

    i=0
    for original, modified in best_epoch_images.items():
        # save_image(original, modified, f"res_{i}", pics_dir)
        i+=1    
        image_key = f'best_epoch_img_{i}'
        wandb.log({
                f'Original Image {image_key}': wandb.Image(original.cpu()),
                f'Modified Image {image_key}': wandb.Image(modified.cpu()),
                })

    print(f'\n\nResults saved.\nBest ASR achieved over {num_epochs} epochs: {best_epoch_asr * 100:.2f}%')

        
    wandb.finish()


def save_results():
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')

    parser.add_argument('--attack_type', choices=['0', '1'], help='0 for the normal patch generation, 1 to include image embedding into the patch training')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--model', choices=['vit_b_16', 'vit_b_32'], help='Model to attack')
    parser.add_argument('--epochs', help='Number of epochs')
    # parser.add_argument('--brightness', help='Brightness value, between 0 (black image) and 2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    attack_type = args.attack_type
    if attack_type == '0':            # without image 
        from generator import Generator
    elif attack_type == '1':
        from gen import Generator
    else:
        raise ValueError('Invalid attack type. \nOptions: 0, 1.')


    if args.model == 'vit_b_16':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16
        weights = ViT_B_16_Weights.DEFAULT

    elif args.model == 'vit_b_32':
        from torchvision.models import vit_b_32, ViT_B_32_Weights
        model = vit_b_32
        weights = ViT_B_32_Weights.DEFAULT

    else:
        raise ValueError('Invalid model type.\nOptions: vit_b_16, vit_b_32')


    input_dim = 100  
    output_dim = 3      
    k = 0.5
    
    generator = Generator(input_dim, output_dim, k).to(device)
    generator.train()

    deployer = Deployer()

    discriminator = model(weights).to(device)  # moving model to the appropriate device
    discriminator.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])



    dataset = datasets.ImageFolder(args.image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_epochs = int(args.epochs) if args.epochs else 40

    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values
    # print(f'-------------- Attack type: {attack_type}')
    start(device, generator, deployer, discriminator, attack_type, dataloader, num_epochs)
