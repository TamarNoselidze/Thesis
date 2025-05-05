import wandb
import torch
import numpy as np
from PIL import Image as PILImage

from torchvision import transforms

class WandbLogger:
    def __init__(self, project_name, config):
        self.run = wandb.init(
            project=project_name, 
            entity='takonoselidze-charles-university', 
            config=config
        )

        self.results_table = wandb.Table(
                columns=["victim model", "target class", "epsilon", "ASR"]
        )

    def tensor_to_numpy(self, tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()

        return tensor.numpy()
    

    def tensor_to_pil(self, tensor):

        # denormalizing
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        denormalized = tensor * std + mean
        
        # clipping
        # clipped = torch.clamp(denormalized, 0.0, 1.0)
        
        return denormalized



    def log_images(self, original, adversarial, misclassified, label):
        to_pil = transforms.ToPILImage()

        original_denormalized = self.tensor_to_pil(original)
        original_pil = to_pil(original_denormalized)
        

        clipped = torch.clamp(adversarial, 0.0, 1.0)
        adversarial_denormalized = self.tensor_to_pil(clipped)
        adversarial_pil = to_pil(adversarial_denormalized)
        # adversarial_pil = to_pil(adversarial_denormalized)
        

        wandb.log({
            "original": wandb.Image(original_pil),
            "adversarial": wandb.Image(adversarial_pil, caption=f'{f"misclassified, label: {label}" if misclassified else "not misclassified"}')
        })


    def log_metrics(self, victim_model, target_class, epsilon, asr):
        self.results_table.add_data(
            victim_model, target_class, epsilon, f'{asr * 100:.2f}%'
        )

    def log_asr(self, asr):
        wandb.log({f'total ASR' : asr * 100})

    def finalize(self):
        wandb.log({"results": self.results_table})
        wandb.finish()