import os, argparse, wandb
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_32, ViT_L_32_Weights, vgg16_bn, VGG16_BN_Weights

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack 
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent


def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor, contrast=0.3)
    adjusted_image = brightness_transform(image)
    return adjusted_image


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


def get_attack_info(attack_name, epsilon):
    attack = None
    attack_params = {}

    if attack_name == 'FGSM':
        attack = fast_gradient_method
        # epsilon, np.inf, targeted=False
        attack_params = {
            'eps' : epsilon,
            'norm' : np.inf,
            # 'clip_min' : None,
            # 'clip_max' : None,
            # 'y' : None,
            'targeted' : False,
            'sanity_checks' : False,
        }
    elif attack_name == 'PGD':
        attack = projected_gradient_descent
        attack_params = {
            'eps': 0.01,
            'eps_iter': 0.01,
            'nb_iter': 15,
            'norm': np.inf,  # Or 1, 2 for l1, l2 norms
            'clip_min': 0,
            'clip_max': 1,
            'targeted': False,
            'sanity_checks' : False
        }
    elif attack_name == 'CWl2':
        attack = carlini_wagner_l2
        attack_params = {
            'confidence': 0,
            'targeted': False,
            'binary_search_steps': 9,
            'max_iterations': 1000,
            'clip_min': 0,
            'clip_max': 1,
            'n_classes': 1000
        }
    elif attack_name == 'HOP-SKIP':
        attack = hop_skip_jump_attack
        attack_params = {
            'norm' : np.inf,   # 2 or np.inf
            'max_num_evals' : 2000,  # default is 10000
            'constraint': 2,  # 'l2', 'l1', or 'linf'
            # 'stepsize': 0.5,
            # 'steps': 40,
            # 'n_samples': 20,
            # 'targeted': False,
            'clip_min': 0,
            'clip_max': 1
        }
    elif attack_name == 'Sl1D':
        attack = sparse_l1_descent
        attack_params = {
            'eps': 4.0,
            'nb_iter': 40,
            'eps_iter': 0.08,
            # 'q': 80,  # q controls sparsity
            'clip_min': 0,
            'clip_max': 1,
            'targeted': False
        }
    else:
        raise ValueError("Unsupported attack")
        
    return attack, attack_params


def attack(attack_name, attack_params, model, dataloader, brightness_factor=None, device='cpu'):
    model.to(device)
    adv_images, orig_predicted_classes, adv_predicted_classes = [], [], []

    for i, batch in enumerate(dataloader):
        images, _ = batch
        images = images.to(device)
        print(f'Working with batch {i+1}')

        for image in images:
            original_image = image.clone().detach().to(device)  # Capture original image

            if brightness_factor != None:
                image = adjust_brightness(image.cpu(), brightness_factor).to(device)

            image = image.unsqueeze(0)  # Add batch dimension

            adversarial_image = attack_name(model, image, **attack_params).detach().to(device)
            original_logits = model(image)
            adversarial_logits = model(adversarial_image)
            _, orig_predicted_class = original_logits.max(1)
            _, adv_predicted_class = adversarial_logits.max(1)

            # Log images based on brightness adjustment
            if brightness_factor != None:
                wandb.log({"original_image": wandb.Image(denormalize(original_image.cpu())), 
                            "image_adjusted_brightness" : wandb.Image(image.cpu()),
                            "adversarial_image": wandb.Image(adversarial_image.cpu())})
            else:
                wandb.log({"original_image": wandb.Image(image.cpu()), 
                            "adversarial_image": wandb.Image(adversarial_image.cpu())})

            adv_images.append(adversarial_image)
            orig_predicted_classes.append(orig_predicted_class)
            adv_predicted_classes.append(adv_predicted_class)
        print(f'Adversarial images for batch {i+1} generated successfully\n')

    return adv_images, orig_predicted_classes, adv_predicted_classes


def evaluate_predictions(original_preds, adversarial_preds):

    mismatched = 0
    for i in range(len(original_preds)):
        or_pred = original_preds[i].item()
        adv_pred = adversarial_preds[i].item()
        if or_pred != adv_pred:
            mismatched += 1
    return mismatched


def get_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == 'resnet152':
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    elif model_name == 'vit_l_32':
        model = vit_l_32(ViT_L_32_Weights.DEFAULT)
    
    model.eval()
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CleverHans Attacks')

    parser.add_argument('--attack', choices=['FGSM', 'CWl2', 'HOP-SKIP', 'PGD', 'Sl1D'], help='The attack to perform.')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    parser.add_argument('--model', choices=['resnet50', 'resnet152', 'vgg16_bn', 'vit_b_16', 'vit_l_32'], help='Model to attack')
    # parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--brightness', help='Brightness value, between 0 (black image) and 2')
    parser.add_argument('--additional', help='Any additional comment that will be added to the project name for logging in wandb')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_folder = args.image_folder_path
    assert os.path.exists(image_folder), f"Could not locate the image dataset at {image_folder}" 
    dataset = datasets.ImageFolder(image_folder, transform=transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # No use of multiple epochs, since Cleverhans has non-training methods - but could be used for future improvements.
    # assert isinstance(args.epochs, int), "Epochs should be an integer"
    # epochs = int(args.epochs)

    try:
        brightness_factor = float(args.brightness)
    except:
        brightness_factor = None
    # brightness_factor = None if args.brightness is None or '' else float(args.brightness)
    additional_comment = args.additional
    attack_name, attack_params = get_attack_info(attack_name=args.attack, epsilon=0.09)
    model_name = args.model
    model = get_model(model_name).to(device)
    
    num_of_images = 2000
    total_mismatch = 0

    project_name = f'Cleverhans {args.attack}-{model_name}{ "(br "+str(brightness_factor)+")" if brightness_factor else ""}{ "_"+additional_comment if additional_comment else ""}'
    wandb.init(project=project_name, config= {
        'model_name': model_name,
        'brightness': args.brightness,
        'number_of_images' : num_of_images,
        'attack_name': args.attack,
        **attack_params  # Log the attack parameters
    })

    adv_images, original_preds, adv_preds = attack(attack_name, attack_params, model, dataloader, brightness_factor, device)
    mismatched = evaluate_predictions(original_preds, adv_preds)
    total_mismatch += mismatched
    wandb.log({f'Number of mistmatches on {model_name} model': mismatched})
    print(f'\n{"="*60}\nNumber of mistmatches on {model_name} model: {mismatched}')

    success_rate = (total_mismatch /  num_of_images) * 100
    wandb.log({f'Success rate of {args.attack} attack on {model_name} model': success_rate})
    print(f'Success rate of {args.attack} attack on {model_name} model: {success_rate}\n{"="*60}\n')
    
    wandb.finish()