import wandb, os, torch

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


    def log_images(self, original, modified, misclassified):
        wandb.log({"original images" : wandb.Image(original),
                   f"modified images" : wandb.Image(modified, caption=f'{"misclassified" if misclassified else "not misclassified"}')
                   })
        # wandb.log({k: wandb.Image(v) for k, v in image_dict.items()})

    def log_metrics(self, victim_model, target_class, epsilon, asr):
        self.results_table.add_data(
            victim_model, target_class, epsilon, f'{asr * 100:.2f}%'
        )

    def log_asr(self, asr):
        wandb.log({f'total ASR' : asr * 100
                   })

    def finalize(self):
        wandb.log({"results": self.results_table})
        wandb.finish()
