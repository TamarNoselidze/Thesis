import os, argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


torch.autograd.set_detect_anomaly(True)

# from generator import Generator
# from gen import Generator
from deployer import Deployer
from helper import save_image, save_patch
from loss import AdversarialLoss

def start(device, generator, deployer, discriminator, dataloader, num_of_epochs=40, input_dim=100):

    num_classes = 200  
    best_epoch_asr = 0  
    best_epoch_images = {}
    total_asr = 0


    for epoch in range(num_of_epochs):

        epoch_total_asr = 0
        best_epoch_patch = None 
        best_batch_asr = 0
        best_batch_images = {}

        for batch in dataloader:
            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)

            batch_size = images.shape[0]   # might change for the last batch

            noise = torch.randn(batch_size, input_dim, 1, 1).to(device)

            # generating patches for each image
            adv_patches = []  # we need to store generated patch for each image

            for i in range(batch_size):
                adv_patch = generator(noise[i].unsqueeze(0).to(device), images[i].unsqueeze(0).to(device))
                adv_patches.append(adv_patch)

            adv_patches = torch.cat(adv_patches, dim=0).to(device)  # Stack generated patches

            # deploying 
            modified_images = []
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

            epoch_total_asr += batch_asr

            if batch_asr > best_batch_asr:
                best_batch_asr = batch_asr
                # best_patch = adv_patch.clone().detach()  # Save the best performing patch
                # best_epoch_patches = adv_patches.clone

                # Save the original and modified images for the best batch
                best_batch_images = {}
                for i in range(batch_size):
                    best_batch_images[images[i].cpu()] = modified_images[i].cpu()  

        if best_batch_asr > best_epoch_asr:   # if the best batch outperforms the best batch in prev epoch, update the best_epoch_images
            best_epoch_asr = best_batch_asr
            best_epoch_images = best_batch_images

        avg_epoch_asr = epoch_total_asr / len(dataloader)
        total_asr += avg_epoch_asr
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")
        print(f'|___ ASR of the best performing batch: {best_batch_asr * 100:.2f}%')

    print(f'\n\n-- Average ASR over all {num_epochs} epochs: {total_asr * 100:.2f}% --\n\n')


    pics_dir = os.path.join(os.getenv('SCRATCHDIR', '.'), 'pics')
    os.makedirs(pics_dir, exist_ok=True)

    i=0
    for original, modified in best_epoch_images.items():
        save_image(original, modified, f"res_{i}", pics_dir)
        i+=1    

    print(f'\n\nResults saved.\nBest ASR achieved over {num_epochs} epochs: {best_epoch_asr * 100:.2f}%')

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')

    parser.add_argument('--attack_type', choices=['0', '1'], help='0 for the normal patch generation, 1 to include image embedding into the patch training')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--model', choices=['vit_b_16', 'vit_tb_32'], help='Model to attack')
    parser.add_argument('--epochs', help='Number of epochs')
    # parser.add_argument('--brightness', help='Brightness value, between 0 (black image) and 2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.attack_type == '0':            # without image 
        from generator import Generator
        # generator = Generator
    elif args.attack_type == '1':
        from gen import Generator
    else:
        pass

    input_dim = 100  
    output_dim = 3      
    k = 0.5
    
    generator = Generator(input_dim, output_dim, k).to(device)
    generator.train()


    if args.model == 'vit_b_16':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16
        weights = ViT_B_16_Weights.DEFAULT
    # else: 
    #     exit

    discriminator = model(weights).to(device)  # moving model to the appropriate device
    discriminator.eval()

    deployer = Deployer()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values


    dataset = datasets.ImageFolder(args.image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_epochs = args.epochs if args.epochs else 40

    start(device, generator, deployer, dataloader, num_epochs)