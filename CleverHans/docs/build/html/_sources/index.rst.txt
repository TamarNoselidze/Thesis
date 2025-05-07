CleverHans Experiments documentation
====================================


Welcome to the documentation of our experiments with CleverHans attacks.

This module evaluates the robustness of modern vision models against common white-box adversarial attacks using the **CleverHans** library. 
Attacks are applied to ImageNetV2 images, and the results can be optionally logged using **Weights & Biases** (W&B). 


In this framework, we provide implementations of the following attacks:

   - Fast Gradient Sign Method (FGSM);
   - Projected Gradient Descent (PGD).

In each setting, the scripts support both, targeted and untargeted attacks.

Additionally, the experiments can be conducted on the following models:

   - ResNet50 and ResNet152 from the CNN family;
   - ViT-B/16 and ViT-L/32 from the ViT family.




.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/CleverHans.main
   api/CleverHans.wandb_logger
