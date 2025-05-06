import wandb

class WandbLogger:
    """
    A simple wrapper for logging CleverHans experiments to Weights & Biases.
    Supports logging of images, metrics (for example, ASR), and summary tables.
    """
    def __init__(self, entity, project_name, config):
        self.run = wandb.init(
            project=project_name, 
            entity=entity, 
            config=config
        )

        # Table with predefined columns to track attack results
        self.results_table = wandb.Table(
            columns=["victim model", "target class", "epsilon", "ASR"]
        )



    def log_images(self, original, adversarial, misclassified, label):
        """
        Log the original and adversarial images.
        """
      
        wandb.log({
            "original": wandb.Image(original),
            "adversarial": wandb.Image(adversarial, caption=f'{f"misclassified, label: {label}" if misclassified else "not misclassified"}')
        })


    def log_metrics(self, victim_model, target_class, epsilon, asr):
        """
        Log a row of metrics into the results table.
        """

        self.results_table.add_data(
            victim_model, target_class, epsilon, f'{asr * 100:.2f}%'
        )

    def log_asr(self, asr):
        """
        Log the overall attack success rate as a scalar metric.
        """
        wandb.log({f'total ASR' : asr * 100})
        

    def finalize(self):
        """
        Log the results table and finish the session.
        """
        wandb.log({"results": self.results_table})
        wandb.finish()