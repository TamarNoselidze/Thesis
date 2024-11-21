import os, argparse, random, numpy
import wandb

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

from deployer_mini import Deployer
# from loss import AdversarialLoss
from generator_mini import Generator

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights


api_key = os.getenv('WANDB_API_KEY')



def start(device, generator, optimizer, deployer, discriminator, dataloader, num_of_epochs=40):
    for epoch in range(num_of_epochs):

        print(f'@ Processing epoch {epoch+1}')
        epoch_images = {}
        for batch in dataloader:

            # print(f'@  working on batch {temp_count}')

            images, true_labels = batch
            images = images.to(device)
            true_labels = true_labels.to(device)

            batch_size = images.shape[0]   # might change for the last batch
            noise = torch.randn(num_of_patches, 100, 1, 1).to(device)
            # noise = torch.randn(1, 100, 1, 1).to(device)
            # noise = torch.randn(batch_size, 100, 1, 1).to(device)
            modified_images = []
            adv_patches = generator(noise)
            # adv_patches = adv_patches.squeeze(0)  # Remove the batch dimension -> [num_patches, 3, patch_size, patch_size]

            # patches = generator(noise)

            # # Apply the patches to the image
            # adv_images = apply_patches_to_image(images.clone(), patches)


            # deploying 
            # patch = adv_patches[0]  # one patch for all images in the batch
            for i in range(batch_size):
                # patch = adv_patches[i]
                # print(f'----------------- The patch is:\n{patch}')
                modified_image = deployer.deploy(adv_patches, images[i])
                modified_images.append(modified_image)

                epoch_images.update({images[i] : modified_image})
            
                image_key = f'epoch_{epoch+1}_img_{i}'
                
                wandb.log({
                image_key: [wandb.Image(images[i].cpu(), caption=f"Original Image {i} (epoch {epoch+1})"), 
                            wandb.Image(modified_image.cpu(), caption=f"Modified Image {i} (epoch {epoch+1})")]
                })


            modified_images = torch.stack(modified_images).to(device)

            # multiple discriminators
            # outputs = []
            # for discriminator in discriminators:
            output = discriminator(modified_images)
                # outputs.append(output)
            print(f"   Discriminator output: {output.data.cpu()}")







def get_model(model_name, device):
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

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Mini-Patch Attack')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../../imagenetv2-top-images/imagenetv2-top-images-format-val')
    # # parser.add_argument('--transfer_mode', choices=['source-to-target', 'ensemble', 'cross-validation'], 
    # #                     help='Choose the transferability approach: source-to-target, ensemble, or cross-validation', default='source-to-target')
    # parser.add_argument('--training_models', type=str)
    # parser.add_argument('--target_models', type=str)
    parser.add_argument('--model', default='vit_b_16')
    parser.add_argument('--patch_size', type=int, help='Size of the mini adversarial patch', default=16)
    parser.add_argument('--num_of_patches', type=int, help='Number of mini adversarial patches', default=16)
    parser.add_argument('--epochs', help='Number of epochs', default=20)
    parser.add_argument('--num_of_classes', type=int, help='Number of (random) classes to train the generator on', default=100)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    discriminator = get_model(args.model, device)
    patch_size = args.patch_size
    num_of_patches = args.num_of_patches
    num_of_epochs = args.epochs
    

    project_name = (
        f"Mini Patch Attack_{args.model}"
    )

    # Initialize W&B
    wandb.init(project=project_name, entity='takonoselidze-charles-university', config={
        'epochs': num_of_epochs,
        # 'classes' : num_of_classes,
        # 'input_dim': input_dim
    })

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(args.image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    generator = Generator(patch_size, num_of_patches).to(device)
    generator.train()

    deployer = Deployer(patch_size, num_of_patches)

    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))   ## try different values


    image = torch.rand(3, 224, 224)  # Dummy image of shape [C, H, W]
    patches = torch.rand(16, 3, 16, 16)  # 16 patches of shape [num_patches, C, patch_size, patch_size]

    start(device, generator, optimizer, deployer, discriminator, dataloader, num_of_epochs)

    wandb.finish()