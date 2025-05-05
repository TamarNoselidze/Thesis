import os, argparse
import numpy as np


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_32, ViT_L_32_Weights

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from wandb_logger import WandbLogger


def transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return transform

def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return image_tensor * std + mean


def get_attack_info(attack_name, epsilon, target_class=None):
    attack = None
    attack_params = {}
    targeted=False

    if target_class:
        targeted = True


    if attack_name == 'FGSM':
        attack = fast_gradient_method
        attack_params = {
            'eps' : epsilon,
            'norm' : np.inf,    #np.inf, 1, 2
            'y' : target_class if targeted else None,
            'targeted' : targeted,
            'sanity_checks' : False,
        }
        
    elif attack_name == 'PGD':
        attack = projected_gradient_descent
        attack_params = {
            'eps': epsilon,
            'eps_iter': 0.01,
            'nb_iter': 15,
            'norm': np.inf,  
            'y' : target_class if targeted else None,
            'targeted': targeted,
        }
    else:
        raise ValueError("Unsupported attack")
        
    return attack, attack_params


def start_attack(attack_name, attack_params, model, dataloader, logger, target_class, device='cpu'):

    total_mismatched = 0
    image_i = 0

    for i, batch in enumerate(dataloader):
        images, _ = batch
        images = images.to(device)
        batch_size = images.shape[0]
        print(f'Working with batch {i+1}')
        batch_asr = 0 
        mismatched = 0

        for image in images:
            original_image = image.clone().detach().to(device)  
            image = image.unsqueeze(0)  #

            adversarial_image = attack_name(model, image, **attack_params).detach().to(device)
            original_logits = model(image)
            adversarial_logits = model(adversarial_image)
            _, orig_predicted_class = original_logits.max(1)
            _, adv_predicted_class = adversarial_logits.max(1)

            misclassified = False
            label = orig_predicted_class.item()
            if target_class is None: 
                if orig_predicted_class.item() != adv_predicted_class.item():
                    mismatched += 1  
                    total_mismatched += 1
                    misclassified = True
                    label = adv_predicted_class.item()
            else:
                if adv_predicted_class.item() == target_class.item():  # SUCCESS for targeted attack
                    mismatched += 1  
                    total_mismatched += 1
                    misclassified = True


                    label = adv_predicted_class.item()

            if image_i % 500 == 0:

                adversarial_image = adversarial_image.squeeze(0)
                original = denormalize(original_image.cpu())  # add denormalize
                modified = adversarial_image.cpu()

                logger.log_images(original, modified, misclassified, label)

            image_i += 1

        batch_asr = mismatched / batch_size * 100

        print(f'Adversarial images for batch {i+1} generated successfully')
        print(f'This batch has an ASR: {batch_asr}\n')

    asr = total_mismatched / image_i

    logger.log_asr(asr)

    return asr 


def get_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == 'resnet152':
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    elif model_name == 'vit_l_32':
        model = vit_l_32(ViT_L_32_Weights.DEFAULT)
    
    model.eval()
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CleverHans Attacks')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--attack', choices=['FGSM', 'PGD'], help='The attack to perform.')
    parser.add_argument('--model', choices=['resnet50', 'resnet152', 'vit_b_16', 'vit_l_32'], help='Model to attack')
    parser.add_argument('--target', help='The target class.')
    parser.add_argument('--epsilon', type=float, help='The perturbation budget, controlling the maximum amount of change allowed to the input. In the range [0,1]', default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_folder = args.image_folder_path
    assert os.path.exists(image_folder), f"Could not locate the image dataset at {image_folder}" 
    dataset = datasets.ImageFolder(image_folder, transform=transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    target_class_tensor = None
    targeted = False
    target_class = None
    if args.target:
        target_class_tensor = torch.tensor([int(args.target)]).to(device)
        target_class = args.target
        targeted = True


    model_name = args.model
    model = get_model(model_name).to(device)

    
    epsilon=args.epsilon
    attack_name, attack_params = get_attack_info(attack_name=args.attack, epsilon=epsilon, target_class=target_class_tensor)


    project_name = (
        f'Cleverhans {args.attack} '
    )

    config = {
        'model_name': model_name,
        'attack_name': args.attack,
        'epsilon': epsilon,
    }


    logger = WandbLogger(project_name, config)

    asr = start_attack(attack_name, attack_params, model, dataloader, logger, target_class_tensor, device)


    print(f'Success rate of {args.attack} attack on {model_name} model: {asr*100 :.2f}%\n{"="*60}\n')
    logger.log_metrics(model_name, target_class, epsilon, asr)
    
    logger.finalize()