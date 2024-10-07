import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.autograd.set_detect_anomaly(True)

# from generator import Generator
from gen import Generator
from deployer import Deployer
from helper import save_image, save_patch
from loss import AdversarialLoss

input_dim = 100  
output_dim = 3      
k = 0.5

generator = Generator(input_dim, output_dim, k).to(device)  # moving generator to the appropriate device (if gpu not available, cpu!!!)
generator.train()

deployer = Deployer()

discriminator = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)  # moving model to the appropriate device
discriminator.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset = datasets.ImageFolder('./imagenetv2-top-images/imagenetv2-top-images-format-val', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


num_epochs = 40
batch_size = 32
num_classes = 1000  

# best_patch = None
# best_asr = 0  

optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values

# noise = torch.randn(batch_size, input_dim, 1, 1)  # Single noise vector to evolve
# adv_patch = generator(noise)  # Create the patch


# with open('results.txt', 'w') as f:

best_epoch_asr = 0  
best_epoch_images = {}

for epoch in range(num_epochs):

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
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")
    print(f'|___ ASR of the best performing batch: {best_batch_asr}')


    # save_patch(best_epoch_patch, f'./pics/best_patch_epoch_{epoch+1}.png')
    

# saving the best results
# save_patch(best_patch, './pics/best_adversarial_patch.png')


i=0
for original, modified in best_epoch_images.items():
    save_image(original, modified, f"res_{i}")
    i+=1    

print(f'\n\nResults saved.\nBest ASR achieved over {num_epochs} epochs: {best_epoch_asr * 100:.2f}%')

# f.write(f'\n\nBest ASR achieved over {num_epochs} epochs: {best_asr * 100:.2f}%')
    