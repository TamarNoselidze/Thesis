## Rough plan of the contents

1. Introduction

  Background and Motivation: 
   - explanation of deep learning
   - vulnerability to adversarial examples and overview of what adversarial attacks do
   - the rise of ViTs in image classification
   - weaknesses of ViTs -> patch attacks
  
  Problem Statement:
   - definition of adversarial patch attacks
  Goals of the thesis: 
  - studying patch attacks
  - enhancing their transferability across models (ViTs, CNNs, etc.)

  abstract:
  Deep learning models are currently state of the art in the area of image classification and object detection, however, they can be easily confused by so called adversarial examples. 
  These are specifically crafted inputs that resemble typical clean inputs, however, the models classify them incorrectly. 
  Adversarial examples have been widely studied in the literature and there are many different ways of creating them. 
  This thesis will focus on so called patch attacks against vision transformers with the goal to make the attacks general and transferable to other models.

  The student will study literature on vision transformers and adversarial attacks with focus on patch attacks. 
  Based on the obtained information, she will test existing attacks and implement new attacks with focus on transferability of the attacks between different image classification and object detection models.

  
Machine learning (/deep learning) has revolutionized the field of image classification and object detection, enabling machines to achieve performance levels comparable to, and in some cases even surpassing, human ability. Especially, the rise of neural networks resulted in an incredibly powerful tool - Convolutional Neural Networks (CNNs), which remained dominant in image processing tasks until not even a decade ago, that is, when Vision Transformers (ViT) emerged as a state-of-the-art architecture. ViTs have outperformed traditional CNNs in several computer vision tasks, due to their self-attention mechanisms which enable modeling(?) long-range dependencies in images.

However, despite their success, these models are still vulnerable to adversarial examples — i.e. input images that have been subtly manipulated to fool the model into making incorrect classifications. Adversarial examples have been extensively studied in the field, leading to a variety of attack methods aimed at confusing different models. Most of these attacks focus on adding small perturbations to an image, making it imperceptibly different from the original. Adversarial examples often appear very similar to clean inputs but are crafted in such a way that they exploit the specific model's weaknesses.

There also exist so-called patch attacks, which, unlike traditional perturbations, involve altering specific regions of an image with a noticeable patch that can still deceive the model. These attacks are not only effective but also practical, as they could be applied in real-world settings without requiring pixel-level precision.

The goal of this thesis is to explore different types of adversarial attacks, especially in the context of Vision Transformers. While much research has been conducted on adversarial attacks targeting CNNs, the behavior of Vision Transformers under adversarial conditions remains less explored. 

~ Since ViTs use patches of input images as the tokens, we would like to specifically focus on patch-based attacks and later on making it general and transferable across different models, both within the ViT family and across other architectures like CNNs.

The transferability of adversarial attacks is a critical aspect because an attack that works across multiple models is far more dangerous than one that is model-specific. By improving the transferability of patch attacks, we can gain deeper insights into the shared vulnerabilities of deep learning models, and ultimately, work towards more robust defenses.





This phenomenon poses significant security risks, especially in applications where image classification is critical, such as autonomous driving, healthcare diagnostics, and surveillance systems. Therefore, it is rather crucial, more than interesting, to research the topic and

2. Theoretical Background
  - CNNs:
    A brief overview
  - ViTs:
    Overview of Vision Transformer architecture and their application in image classification.
    Maybe comparing to CNNs? 
  - Adversarial Attacks
    Types of existing adversarial attacks: white box, black box. 
    Cleverhans attacks ?
    Briefly about patch attacks.
  - Transferability of Adversarial Attacks
    Challenges and significance of attack transferability across models.
  - Related Work
    Review of related work on adversarial attacks, focusing on patch attacks.

3. Experimenting with existing attacks
  - Cleverhans
  - Square attack
  - ...

4. Patch Attacks
  - Briefly: importance of patches in ViTs
  - Existing Patch Attack Methods
    Modifications required to adapt existing patch attacks to Vision Transformers.
  - Random Position Patch attack - overview of the paper 
       my implementation of the paper + additional features (?)

5. Transferability

  -----

6. Methodology
  - Research and Implementation Framework
    Tools, frameworks, and datasets used for patch attack implementation.
  - Experiment Setup
    Overview of the setup, including attack generation, testing across different models, and evaluation criteria.
  - Transferability Testing
    Description of experiments designed to test the transferability of patch attacks across multiple models.


7. Results
  - Evaluation Metrics
    Explanation of metrics used for evaluating attack success and transferability.  for example, ASR
  - Attack Performance on Vision Transformers
    Results of applying patch attacks to Vision Transformers.
  - Transferability to Other Models
    Results of transferring the attacks to other models (e.g., CNNs) and comparison of performance.
  - Analysis of Generalization
    Discussion of the factors contributing to transferability and generalization of patch attacks.


8. Conclusion
  - Summary of Findings
    Recap of the main results and conclusions drawn from the research.
  - Contributions of the Thesis
    Contributions made to the field, including new approaches to patch attack transferability.


9. Bibliography

