# Introduction

This repository contains the code and experiments conducted for my bachelor thesis on Adversarial Attacks Against Vision Transformers. It includes two main components:

1. CleverHans Experiments: in this section we evaluate standard adversarial attack methods on multiple models. 
2. Random Position Patch Experiments: here, we provide an implementation of ["Designing Physical-World Universal Attacks on Vision Transformers"](https://openreview.net/forum?id=DqBPk7887N) paper, as well as our novel proposed approach of "Mini-Patch Attacks".


For both experimental setups we use the [ImageNetV2](https://github.com/modestyachts/ImageNetV2) dataset. This dataset contains 1000 classes, with 10 images in each. 


We discuss how to use the scripts provided in each section below.

# CleverHans Experiments

This module evaluates the robustness of modern vision models against common white-box adversarial attacks using the CleverHans library. The attacks are applied to a subset of ImageNetV2 images, and the results are logged via Weights & Biases. 


Our goal is to show how vulnerable standard architectures like ResNet and Vision Transformers (ViT) are to:
- Fast Gradient Sign Method (FGSM);
- Projected Gradient Descent (PGD).

For both methods we can run the experiments in two types of settings: targeted and untargeted attack. 
For the untargeted attack, user does not have to provide any additional arguments, and the attack will be conducted to simply mislead the victim model into classifying input images as anything but their true label. On the other hand, for the targeted setting, user needs to provide a numerical label for the desired target class. In this case, the attack is successful if it manages to fool the model into classifying the adversarial image as the provided target class.

Additionally, the user can specify epsilon values for each attack, i.e., perturbation magnitude. Intuitively, low epsilon values guarantee that images will be modified minimally and the difference between the original and adversarial examples will be essentially unnoticeable.


The models used within this framework are:
- ResNet50 and ResNet152 from the CNN family
- ViT-B/16 and ViT-L/32 from the ViT family.

Each model is evaluated under identical conditions to compare adversarial susceptibility.

To run experiments in this framework, move to the `CleverHans` directory and run the following command:

`python main.py --image_folder_path <path to image dataset> --attack <attack mode>  --model <model name> --target <target class (numerical value)> --epsilon <perturbation magnitude>`



# Random Position Patch Experiments



qsub -v "RUN_MODE=train,ATTAgitCK_MODE=mini_0,TRAINING_MODELS='swin_b',TARGET_CLASS=932,PATCH_SIZE=16,PATCHES=8,EPOCHS=20,CLASSES=700" job.sh

qsub -v "RUN_MODE=test,ATTACK_MODE=mini_0,TRAINING_MODELS='swin_b',TARGET_MODELS='vit_b_16',TARGET_CLASS=932,PATCH_SIZE=16,PATCHES=8" job.sh


