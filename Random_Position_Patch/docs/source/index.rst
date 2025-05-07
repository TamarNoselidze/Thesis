Random Position Patch Documentation
===================================

Welcome to the documentation of Random Position Patch attacks.

This module implements several patch-based adversarial methods:
The two main types of attacks we explore are **G-Patch** (proposed in the paper https://openreview.net/forum?id=DqBPk7887N) and **Mini-Patch** attacks.

In both attack types, we leverage a GAN-like architecture, where we train a **Generator** that crafts adversarial patches, and then a **Deployer** applies the patches at random positions within the images.
For G-Patches we provide the following:

   - `main.py`: Entry point for training and evaluation.
   - `generator.py`: Defines the patch generator network.
   - `deployer.py`: Handles applying the patch to input images.
   - `helper.py`: Utilities and support functions.
   - `loss.py`: Contains custom loss functions.
   - `wandb_logger.py`: Integrates Weights & Biases logging.

For Mini-Patches, we only need a modified deployer mechanism, so we include:
   - `Mini_Patches/` directory with `deployer_mini.py` to provide deployment logic for mini-patch variants.



Additionally, the experiments can be conducted on the following models:

- ResNet50, ResNet152 and VGG16-BN from the CNN family;
- ViT-B/16, ViT-B/32, ViT-L/16 and Swin-B from the ViT family.





.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/Random_Position_Patch.generator
   api/Random_Position_Patch.deployer
   api/Random_Position_Patch.helper
   api/Random_Position_Patch.main
   api/Random_Position_Patch.loss
   api/Random_Position_Patch.wandb_logger
   api/Random_Position_Patch.Mini_Patches

  
