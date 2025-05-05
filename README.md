# Adversarial Attacks

This repository contains the code and experiments conducted for my bachelor thesis on Adversarial Attacks Against Vision Transformers. It includes two main components:

1. CleverHans Experiments: in this section we evaluate standard adversarial attack methods on multiple models. 
2. Random Position Patch Experiments: here, we provide an implementation of [1] "Designing Physical-World Universal Attacks on Vision Transformers" paper [G-patch](https://openreview.net/forum?id=DqBPk7887N)

we train a generator to produce adversarial patches applied at random positions.


We discuss how to use the scripts provided in each section below.

## Cleverhans Experiments



Supported Attacks:
    FGSM

    PGD

Models: 
    ResNet50

    ResNet152

    ViT-B/16

    ViT-L/32






## Random Position Patch Experiments







All scripts are compatible with MetaCentrum job submission system.

Results are reproducible via W&B logging.


qsub -v "RUN_MODE=train,ATTAgitCK_MODE=mini_0,TRAINING_MODELS='swin_b',TARGET_CLASS=932,PATCH_SIZE=16,PATCHES=8,EPOCHS=20,CLASSES=700" job.sh

qsub -v "RUN_MODE=test,ATTACK_MODE=mini_0,TRAINING_MODELS='swin_b',TARGET_MODELS='vit_b_16',TARGET_CLASS=932,PATCH_SIZE=16,PATCHES=8" job.sh


