import wandb

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



    def log_images(self, original, adversarial, misclassified, label):
        
        wandb.log({
            "original": wandb.Image(original),
            "adversarial": wandb.Image(adversarial, caption=f'{f"misclassified, label: {label}" if misclassified else "not misclassified"}')
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