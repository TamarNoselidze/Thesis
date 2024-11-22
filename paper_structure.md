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


    Convolutional Neural Networks (CNNs) are a class of deep learning models mainly used for image processing tasks, such as image classification, object detection, etc. They consist of layers that apply convolutional operations, which detect local patterns and features in an image, followed by pooling layers to down-sample and reduce dimensionality. CNNs are very effective in learning spatial hierarchies and have become the standard for many computer vision tasks, especially image classification, due to their ability to effectively extract relevant features.


  - ViTs:

    Overview of Vision Transformer architecture and their application in image classification.
    Maybe comparing to CNNs? 

    Vision Transformers (ViTs) are quite different from traditional convolution-based approaches, and they are transformer-based architectures for image classification tasks. Unlike CNNs, which use convolutional layers to detect local patterns, ViTs treat an image as a sequence of patches, applying transformer layers (originally designed for natural language processing tasks) to learn dependencies across these patches. The image is split into fixed-size patches, which are then linearly embedded and processed using self-attention mechanisms. ViTs have shown impressive results in image classification tasks, especially when trained on large datasets, and offer advantages in terms of capturing long-range dependencies in images.
   
    Comparison to CNNs:

    While CNNs are effective at learning local patterns in images, ViTs excel at modeling global relationships due to their self-attention mechanism. However, ViTs often require larger datasets and computational resources to outperform CNNs. CNNs are still more efficient in scenarios where labeled data is limited, while ViTs can potentially achieve higher performance on large-scale datasets.

    
  - Adversarial Attacks
    Types of existing adversarial attacks: white box, black box. 

    Adversarial attacks are methods used to manipulate the input data in a way that causes machine learning models, particularly deep learning models, to make incorrect predictions. These attacks are a significant challenge in the deployment of AI systems, particularly in safety-critical applications.

    White Box Attacks: In white box attacks, the attacker has complete knowledge of the target model, including its architecture, weights, and parameters. This allows the attacker to craft precise perturbations to the input data that are likely to mislead the model. Examples of white box attacks include the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

    Black Box Attacks: In black box attacks, the attacker has no access to the model's internal structure or parameters. Instead, they can only observe the model's outputs for given inputs. Despite this limitation, black box attacks can still be effective, often using techniques like query-based attacks, where the attacker generates adversarial examples by repeatedly querying the model with slightly modified inputs. Examples include the Boundary Attack and the Transfer Attack, where adversarial examples from one model are used to attack another.


    Briefly about patch attacks.
  - Transferability of Adversarial Attacks
    Challenges and significance of attack transferability across models.



    The transferability of adversarial attacks refers to the phenomenon where adversarial examples generated for one machine learning model (typically a source model) can also mislead a different model (typically a target model). This is a significant aspect of adversarial robustness, as it implies that an adversarial example crafted for one specific model can still be effective against models with different architectures or training procedures.

    Challenges:

    Model Diversity: Models can vary in terms of their architectures (e.g., CNNs vs. ViTs), training data, hyperparameters, and regularization techniques, making it difficult to create universal adversarial examples that successfully attack all models. The behavior of adversarial examples across models is not always predictable, and the transferability rate may depend on the similarity between the models.

    Robustness Variations: Different models exhibit different levels of robustness to adversarial examples. Some models might be more resistant to certain types of attacks, making the transferability less effective. Understanding and overcoming these variations is an ongoing challenge.

    Optimization Overhead: Generating transferable adversarial examples typically requires iterating through various attack strategies, models, and parameters to ensure that the perturbation works across multiple target models. This can be computationally expensive and time-consuming.
  - Related Work
    Review of related work on adversarial attacks, focusing on patch attacks.

Existing attacks

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

