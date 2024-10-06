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

# discriminator = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
discriminator = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)  # moving model to the appropriate device
discriminator.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


#dataset = datasets.ImageFolder('../imagenetv2-top-images/imagenetv2-top-images-format-val', transform=transform)
dataset = datasets.ImageFolder('./imagenetv2-top-images/imagenetv2-top-images-format-val', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



num_epochs = 40
batch_size = 32
num_classes = 1000  

best_patch = None
best_asr = 0  
best_example_images = {}

optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values

# noise = torch.randn(batch_size, input_dim, 1, 1)  # Single noise vector to evolve
# adv_patch = generator(noise)  # Create the patch


# with open('results.txt', 'w') as f:

for epoch in range(num_epochs):

    epoch_images = {}
    best_epoch_patch = None 

    for batch in dataloader:
        # print('aaaaaa')
        images, true_labels = batch
        batch_size = images.shape[0]

        noise = torch.randn(batch_size, input_dim, 1, 1)
        # adv_patch = generator(noise)  # add an image

        adv_patches = []  # Store generated patches for each image

        #deploying 
        # modified_images = []
        # torch.empty_like(images)
        for i in range(batch_size):
            # modified_images[i] = deployer.deploy(adv_patch[i], images[i])
            print(F'------------------------------ noise shape: {noise[i].unsqueeze(0).shape}')
            adv_patch = generator(noise[i].unsqueeze(0), images[i].unsqueeze(0))

            # modified_images.append(deployer.deploy(adv_patch[i], images[i]))
            # epoch_images[images[i]] = modified_images[i]
            adv_patches.append(adv_patch)
            # save_image(images[i], modified_images[i], f"image_{i}_epoch_{epoch+1}")

        adv_patches = torch.cat(adv_patches, dim=0)  # Stack generated patches
        # print([t.shape for t in modified_images])
        # modified_images = torch.stack(modified_images)




        modified_images = []

        for i in range(batch_size):
            modified_image = deployer.deploy(adv_patches[i], images[i])
            modified_images.append(modified_image)
        
        modified_images = torch.stack(modified_images)






        outputs = discriminator(modified_images)
        
        target_class_y_prime = torch.randint(0, num_classes, (batch_size,))
        target_class_y_prime[target_class_y_prime == true_labels] = (target_class_y_prime[target_class_y_prime == true_labels] + 1) % num_classes
        
        criterion = AdversarialLoss(target_class=target_class_y_prime)
        loss = criterion(outputs)
        
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


        _, predicted = torch.max(outputs.data, 1)
        
        correct = (predicted == true_labels).sum().item()
        asr = (batch_size - correct) / batch_size

        # if asr > best_epoch_asr:
        #     best_epoch_asr = asr
        #     best_epoch_patch = adv_patch

        if asr > best_asr:
            best_asr = asr
            best_patch = adv_patch.clone().detach()  # Save the best performing patch

        

    # with open('results.txt', 'w') as f:
    # f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    # f.write(f'The best ASR for the epoch: {asr * 100:.2f}%')

# saving the best results
save_patch(best_patch, './pics/best_adversarial_patch.png')

i=0
for original, modified in best_example_images.items():
    save_image(original, modified, f"res_{i}")
    i+=1    

print(f'\n\n Results saved.\nBest ASR achieved over {num_epochs} epochs: {best_asr * 100:.2f}%')

# f.write(f'\n\nBest ASR achieved over {num_epochs} epochs: {best_asr * 100:.2f}%')
