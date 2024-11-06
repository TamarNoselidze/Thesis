import os, argparse, random, numpy

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

from deployer import Deployer
from loss import AdversarialLoss

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights, vit_b_32, ViT_B_32_Weights, vgg16_bn, VGG16_BN_Weights, swin_b, Swin_B_Weights

# from dotenv import load_dotenv


seed = 42  # Use a fixed seed for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# load_dotenv(os.path.join(os.getenv('SCRATCHDIR', '.'), '.env'))
api_key = os.getenv('WANDB_API_KEY')
# print(f"WANDB_API_KEY: {api_key}")  


def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor, contrast=0.3)
    adjusted_image = brightness_transform(image)
    return adjusted_image

def start(project_name, device, generator, optimizer, deployer, discriminator, attack_type, dataloader, num_of_classes, num_of_epochs=40, brightness_factor=None, color_transfer=None, input_dim=100):
    import wandb

    # Initialize W&B
    # wandb.init(project=project_name, entity='takonoselidze-charles-university', config={
    #     'epochs': num_of_epochs,
    #     'classes' : num_of_classes,
    #     'attack_type': 'Including the image embeddings' if attack_type=='0' else 'Without the image embeddings',
    #     'input_dim': input_dim
    # })


    best_epoch_asr = 0  
    best_epoch_images = {}
    total_asr = 0


    for epoch in range(num_of_epochs):

        print(f'@ Processing epoch {epoch+1}')

        epoch_total_asr = 0
        # best_epoch_patch = None   
        best_batch_asr = 0
        best_batch_images = {}

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
            adv_patches = None

            if attack_type == '0':
                adv_patches = generator(noise)
            else:
                # generating patches for each image
                adv_patches = []  # we need to store generated patch for each image

                for i in range(batch_size):
                    # print(f'*************** calling from start')
                    adv_patch = generator(noise[i].unsqueeze(0).to(device), images[i].unsqueeze(0).to(device))
                    adv_patches.append(adv_patch)

                adv_patches = torch.cat(adv_patches, dim=0).to(device)  # Stack generated patches

            # print(f"   Generated adversarial patch: {adv_patches[0].cpu()}")  # Log a generated patch

            # deploying 
            for i in range(batch_size):
                modified_image = deployer.deploy(adv_patches[i], images[i])
                modified_images.append(modified_image)

                # epoch_images[images[i]] = modified_image
                # epoch_images.update({images[i] : modified_image})
            
            modified_images = torch.stack(modified_images).to(device)

            outputs = discriminator(modified_images)
            # print(f"   Discriminator output: {outputs.data.cpu()}")

            
            target_class_y_prime = torch.randint(0, 1000, (batch_size,)).to(device)
            # print(target_class_y_prime)
            target_class_y_prime[target_class_y_prime == true_labels] = (target_class_y_prime[target_class_y_prime == true_labels] + 1) % num_of_classes
            
            criterion = AdversarialLoss(target_class=target_class_y_prime).to(device)
            loss = criterion(outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            
            correct = (predicted == true_labels).sum().item()
            # correct = (predicted == true_labels).sum()


            print(f"     Loss: {loss.item()}")

            # If you have the criterion, you can also print its input values
            # print(f"   Criterion input (outputs): {outputs.data.cpu()}")
            # print(f"   Criterion target: {target_class_y_prime.cpu()}")


            print(f"     True labels: {true_labels.cpu()}")
            print(f"     Predicted labels: {predicted.cpu()}")

            print(f"     Correct predictions: {correct}")

            batch_asr = (batch_size - correct) / batch_size
            print(f'@    this batch (number {temp_count}) has ASR: {batch_asr}')
            print(f'@    best_batch_asr so far: {best_batch_asr}')
            temp_count+=1


            # Log batch metrics to W&B
            # wandb.log({
            #     'batch_loss': loss.item(),
            #     'batch_asr': batch_asr,
            #     'epoch': epoch + 1
            # })
            
            batch_images = {}
            for i in range(batch_size):
                batch_images[images[i].cpu()] = modified_images[i].cpu()  

            epoch_images.update(batch_images)
            epoch_total_asr += batch_asr

            if batch_asr > best_batch_asr:
                best_batch_asr = batch_asr
                print(f'@       current batch asr is better than best_batch_asr ({best_batch_asr}) so we are updating its value.')
                # best_patch = adv_patch.clone().detach()  # Save the best performing patch
                # best_epoch_patches = adv_patches.clone

                # Save the original and modified images for the best batch
                # best_batch_images = {}
                # for i in range(batch_size):
                #     best_batch_images[images[i].cpu()] = modified_images[i].cpu()  

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
        # wandb.log({
        #     'epoch_avg_asr': avg_epoch_asr,
        #     'epoch_best_batch_asr': best_batch_asr,
        #     'epoch': epoch + 1
        # })

        print(f"Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item()}, Avg ASR: {avg_epoch_asr * 100:.2f}%")
        print(f'|___ ASR of the best performing batch: {best_batch_asr * 100:.2f}%')

    print(f'\n\n-- Average ASR over all {num_of_epochs} epochs: {total_asr / num_of_epochs * 100:.2f}% --\n\n')



    i=0
    for original, modified in best_epoch_images.items():
        i+=1    
        image_key = f'best_epoch_img_{i}'
          
        # wandb.log({
        # image_key: [wandb.Image(original.cpu(), caption=f"Original Image {i}"), 
        #             wandb.Image(modified.cpu(), caption=f"Modified Image {i}")]
        #  })
        # print(f'displayed: original and modified of {image_key}')

    print(f'\n\nResults saved.\nBest ASR achieved over {num_of_epochs} epochs: {best_epoch_asr * 100:.2f}%')

        
    wandb.finish()

def get_model(model_name):
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
    
    # model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')

    parser.add_argument('--attack_type', choices=['0', '1'], help='0 for the normal patch generation, 1 to include image embedding into the patch training')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--model', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'swin_b'], help='Model to attack')
    parser.add_argument('--patch_size', choices=['48', '64', '80'], help='Size of the adversarial patch')
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--brightness', help='Brightness level for the patch')
    parser.add_argument('--color_transfer', help='Color transfer value for the patch')

    # parser.add_argument('--brightness', help='Brightness value, between 0 (black image) and 2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attack_type = args.attack_type
    if attack_type == '0':            # without image 
        from generator import Generator
    elif attack_type == '1':
        from gen import Generator
    else:
        raise ValueError('Invalid attack type. \nOptions: 0, 1.')

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

    model_name = args.model
    discriminator = get_model(model_name).to(device)  # moving model to the appropriate device
    discriminator.eval()

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
    print(f"Discriminator device: {next(discriminator.parameters()).device}")
    
    project_name = f'Random Position Patch_{args.attack_type}-{args.model}{ "(br "+str(brightness_factor)+")" if brightness_factor else ""}{ "_col-tr"+color_transfer if color_transfer else ""}'
    start(project_name, device, generator, optimizer, deployer, discriminator, attack_type, dataloader, num_of_classes, num_of_epochs, brightness_factor, color_transfer)