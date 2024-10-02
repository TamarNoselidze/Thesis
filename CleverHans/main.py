from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50,ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights

from torch.autograd import Variable

# from cleverhans.torch.attacks import fast_gradient_method, projected_gradient_descent
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack 
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent

# import PIL
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import argparse


def preprocess_image(raw_image_path):

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(raw_image_path)
    image_tensor = preprocess(image)    # for saving the preprocessed image
    image_tensor_unsqueezed = image_tensor.unsqueeze(0)  # Add a batch dimension   # for attacking

    return image_tensor, image_tensor_unsqueezed

def adjust_brightness(image, brightness_factor):
    brightness_transform = transforms.ColorJitter(brightness=brightness_factor)
    image = brightness_transform(image)
    return image

def transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform

def save_images(original_image, adversarial_image, result_dir, image_name, ):
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    original_image_normalized = inv_normalize(original_image)
    adversarial_image = adversarial_image.squeeze(0)  # Remove batch dimension
    adversarial_image_normalized = inv_normalize(adversarial_image)
    adversarial_image_normalized = adversarial_image_normalized.clamp(0, 1)  # Clip to make sure the values are valid for an image

    to_pil_image = transforms.ToPILImage()
    original_image_pil = to_pil_image(original_image_normalized)
    adv_image_pil = to_pil_image(adversarial_image_normalized)


    original_image_pil.save(f'{result_dir}/{image_name}_processed.jpg')
    adv_image_pil.save(f'{result_dir}/{image_name}_adversarial.jpg')


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
            'eps': epsilon,
            'eps_iter': 0.01,
            'nb_iter': 40,
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
            'max_num_evals' : 100,

            'constraint': 2,  # 'l2', 'l1', or 'linf'
            # 'stepsize': 0.5,
            # 'steps': 40,
            # 'n_samples': 20,
            # 'targeted': False,
            'clip_min': 0,
            'clip_max': 1
        }
    elif attack_name == 'SparseL1':
        attack = sparse_l1_descent
        attack_params = {
            'eps': epsilon,
            'nb_iter': 40,
            'eps_iter': 0.01,
            'q': 80,  # q controls sparsity
            'clip_min': 0,
            'clip_max': 1,
            'targeted': False
        }
    else:
        raise ValueError("Unsupported attack")
        

    return attack, attack_params

def attack(attack_name, attack_params, model, dataloader, brightness_factor=None, epsilon=0.03):

    # adv_image = fast_gradient_method(model, image, epsilon, np.inf, targeted=False)
    # adv_image = sparse_l1_descent(model, image, **attack_params)
    adv_images = []
    orig_predicted_classes = []
    adv_predicted_classes = []

    for i, batch in enumerate(dataloader):
        images, _ = batch
        batch_size = images.shape[0]
        print(f'Working with batch {i+1}. Size of this batch is: {batch_size}')

        # for j in range(batch_size):
        for image in images:
            if brightness_factor != None:
                image = adjust_brightness(image, brightness_factor)
            image = image.unsqueeze(0)  # Add batch dimension

            adversarial_image = attack_name(model, image, **attack_params)
            original_logits = model(image)
            adversarial_logits = model(adversarial_image)
            _, orig_predicted_class = original_logits.max(1)
            _, adv_predicted_class = adversarial_logits.max(1)
            adv_images.append(adversarial_image)
            orig_predicted_classes.append(orig_predicted_class)
            adv_predicted_classes.append(adv_predicted_class)

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
    # models = []
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
    elif model_name == 'vit_b_32':
        model = vit_b_32(ViT_B_32_Weights.DEFAULT)
    
    model.eval()
    # models.append(model)

    ## add more models
        
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CleverHans Attacks')

    parser.add_argument('--attack', choices=['FGSM', 'CWl2', 'HOP-SKIP', 'PGD', 'Sl1D'], help='The attack to perform.')
    parser.add_argument('--image_folder_path', help='Image dataset to perturb', default='../imagenetv2-top-images/imagenetv2-top-images-format-val')
    # parser.add_argument('--model', choices=['resnet50', 'vit_b_16'], help='Model to attack')
    parser.add_argument('--models', nargs='+', choices=['resnet50', 'vit_b_16', 'vit_b_32'], help='Models to attack')
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--brightness', help='Brightness value, between 0 (black image) and 2')
    args = parser.parse_args()

    models = args.models
    image_folder = args.image_folder_path
    brightness_factor = None if args.brightness is None else float(args.brightness)
    # attack_name = args.attack
    dataset = datasets.ImageFolder(image_folder, transform=transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    attack_name, attack_params = get_attack_info(attack_name=args.attack, epsilon=0.09)

    epochs = int(args.epochs)
    output_filename = f'./results{args.attack}-VS-{"&".join(models)}.txt'
    num_of_images = 100

    with open(output_filename, 'w') as f:
        total_mismatch = {x : 0 for x in models}
        # for i in range(epochs):
        #     f.write(f'{"-"*30} Results for epoch {i+1} {"-"*30}\n')
        #     print(f'{"-"*30} Results for epoch {i+1} {"-"*30}')

        for model_name in models:
            model = get_model(model_name)
            adversarial_images, original_preds, adversarial_preds = attack(attack_name, attack_params, model, dataloader, brightness_factor)
            mismatched = evaluate_predictions(original_preds, adversarial_preds)
            f.write(f'The attack {args.attack} managed to fool the model `{model_name}` {mismatched} out of {num_of_images} times.\n')
            print(f'The attack {args.attack} managed to fool the model `{model_name}` {mismatched} out of {num_of_images} times.')
            total_mismatch[model_name] += mismatched
        f.write(f'{"-"*85}\n\n')
        
        for mod_name, Nmis in total_mismatch.items():
            f.write(f'\nAverage success rate of the attack on {mod_name} model: {Nmis/num_of_images* 100:.2f}%\n')
        f.flush()
    # save_images(image_tensor, adversarial_image, result_dir, image_name)