# Introduction

This repository contains the code and experiments conducted for my bachelor thesis on **Adversarial Attacks Against Vision Transformers**. It includes two main components:

1. **CleverHans Experiments** - evaluation of standard adversarial attack methods on multiple models. 
2. **Random Position Patch Experiments** - implementation of the paper ["Designing Physical-World Universal Attacks on Vision Transformers"](https://openreview.net/forum?id=DqBPk7887N), as well as our novel **Mini-Patch Attacks**.

All experiments use the [ImageNetV2](https://github.com/modestyachts/ImageNetV2) dataset, which contains 1000 classes with 10 images each. 


Instructions on how to use each component are detailed below.

---


# CleverHans Experiments

This module evaluates the robustness of modern vision models against common white-box adversarial attacks using the **CleverHans** library. Attacks are applied to ImageNetV2 images, and the results can be optionally logged using **Weights & Biases** (WANDB). 

### Attack methods used in this framework
- Fast Gradient Sign Method (FGSM);
- Projected Gradient Descent (PGD).

### Evaluated models
- ResNet50 and ResNet152 from the CNN family;
- ViT-B/16 and ViT-L/32 from the ViT family.

### Attack types
- Untargeted Attack:
  The model is fooled into predicting *any class* other than the true label.  
  No additional parameters are needed.
  
- Targeted Attack:
  The model is forced to classify the image as a *specific target class*.  
  The user must supply a numerical label for the target class using the `--target` argument.

### Parameters:

- `--image_folder_path`: path to the ImageNetV2 dataset
- `--attack`: attack type (`FGSM` or `PGD`)
- `--model`: model name (`resnet50`, `resnet152`, `vit_b_16`, `vit_l_32`.)
- `--epsilon`: perturbation magnitude
- `--target`: *(optional)* target class for targeted attack
- `--wandb_entity`: *(optional)* WANDB entity for logging


To run experiments in this framework, move to the `CleverHans` directory and run:

`python main.py --image_folder_path '../imagenetv2-top-images/imagenet-imagenetv2-top-images-format-val' --attack FGSM -- odel vit_b_16 --epsilon 0.05`

For untargeted attacks, simply omit the `--target` argument. 

#### Example Commands:

**Untargeted FGSM on ViT-B/16:**
```bash
python main.py \
  --image_folder_path '../imagenetv2-top-images/imagenet-imagenetv2-top-images-format-val' \
  --attack FGSM \
  --model vit_b_16 \
  --epsilon 0.05
```

## Outputs

Results of the experiments are simply printed out to the standard output with:
- Per-batch success rates
- Final succesrate over the entire dataset

To enable W&B logging for better visualization, ensure:
- you have a `.env` file outside the `CleverHans` directory containing:
```
WANDB_API_KEY=your_api_key_here
```
- and you use the optional `--wandb_entity` argument when running the experiments.


* The ImageNetV2 dataset is not included in the repository due to storage limits. The script expects it to be in the same directory as `CleverHans` by default.


---

# Random Position Patch Experiments



qsub -v "RUN_MODE=train,ATTAgitCK_MODE=mini_0,TRAINING_MODELS='swin_b',TARGET_CLASS=932,PATCH_SIZE=16,PATCHES=8,EPOCHS=20,CLASSES=700" job.sh

qsub -v "RUN_MODE=test,ATTACK_MODE=mini_0,TRAINING_MODELS='swin_b',TARGET_MODELS='vit_b_16',TARGET_CLASS=932,PATCH_SIZE=16,PATCHES=8" job.sh


