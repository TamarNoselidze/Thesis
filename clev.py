import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet50_Weights
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



def attack(attack_name, image, model, epsilon=0.03):
    if attack_name == 'FGSM':
        adv_image = fast_gradient_method(model, image, epsilon, np.inf, targeted=False)
       
    elif attack_name =='CWl2':

        attack_params = {
            'confidence': 0,
            'targeted': False,
            'binary_search_steps': 9,
            'max_iterations': 1000,
            'clip_min': 0,
            'clip_max': 1,
            'n_classes': 1000
        }
        adv_image = carlini_wagner_l2(model, image, **attack_params)

    elif attack_name == 'HOP-SKIP':
        # Hop Skip Jump Attack
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
        adv_image = hop_skip_jump_attack(model, image, **attack_params)

    elif attack_name == 'PGD':
        # Projected Gradient Descent
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
        adv_image = projected_gradient_descent(model, image, **attack_params)

    elif attack_name == 'SparseL1':
        # Sparse L1 Descent
        attack_params = {
            'eps': epsilon,
            'nb_iter': 40,
            'eps_iter': 0.01,
            'q': 80,  # q controls sparsity
            'clip_min': 0,
            'clip_max': 1,
            'targeted': False
        }
        adv_image = sparse_l1_descent(model, image, **attack_params)

    else:
        raise ValueError("Unsupported attack")

    # Run model prediction on both original and adversarial images
    original_logits = model(image)
    adversarial_logits = model(adv_image)


    _, orig_predicted_class = original_logits.max(1)
    _, adv_predicted_class = adversarial_logits.max(1)

    return adv_image, orig_predicted_class.item(), adv_predicted_class.item()



def get_model(model_name):
    if model_name == 'ResNet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    ### ADD MORE MODELS
        
    model.eval()  # Set the model to evaluation mode
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CleverHans Attacks')

    parser.add_argument('--attack', choices=['FGSM', 'CWl2', 'HOP-SKIP', 'PGD', 'Sl1D'], help='The attack to perform.')
    parser.add_argument('--image_path', help='The image to pertub')
    parser.add_argument('--model', choices=['ResNet50', ''], help='Model to attack')
    args = parser.parse_args()

    image_name = args.image_path.split('/')[-1].split('.')[0]
    image_tensor, image_unsq = preprocess_image(args.image_path)

    model = get_model(args.model)

    adversarial_image, pred_original, pred_adversarial = attack(args.attack, image_unsq, model, epsilon=0.09)

    print(f'Original image prediction: {pred_original}')
    print(f'Adversarial image prediction: {pred_adversarial}')

    result_dir = f"./Processed_Adversarial/{args.attack}"
    save_images(image_tensor, adversarial_image, result_dir, image_name)
