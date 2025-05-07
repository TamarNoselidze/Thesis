import wandb, os, torch
from .helper import get_class_name

class WandbLogger:
    """
    A simple wrapper for logging experiments to Weights & Biases.
    Supports logging of images, metrics (for example, ASR), and summary tables.
    """

    def __init__(self, entity, run_mode, project_name, target_class, config):
        self.run = wandb.init(
            project=project_name, 
            entity=entity, 
            config=config
        )
        self.run_mode = run_mode
        self.target_class_name = get_class_name(target_class)
        self.create_tables()
        
        
    def create_tables(self):
        """ Tables for structured logging """

        if self.run_mode == "train":
            # Table to track metrics of the best-performing generators 
            self.best_generators_table = wandb.Table(
                columns=["target class_name", "best epoch", "best ASR"]
            )

            # Table for tracking generator evaluation results
            self.generator_eval_table = wandb.Table(
                columns=["target class name", "random noise #", "epoch", "ASR"]
            )
            
        else:
            # Table for tracking test results on target models
            self.target_model_results_table = wandb.Table(
                columns=["train model(s)", "target class", "target model",
                        "patch from" , "misclassified", "total", "ASR"]
            )
        
    

    def log_generator_evaluation(self, noise_i, epoch, asr):
        """ Log individual generator evaluation result. """

        self.generator_eval_table.add_data(self.target_class_name, noise_i, epoch, f'{asr * 100:.2f}%')
        # Also log as metrics for easier time-series visualization
        wandb.log({
            f"eval/noise_{noise_i}/epoch_ASR": asr * 100,
        })
        
    
    def log_patch_image(self, noise_i, epoch, patch):
        """ Log adversarial patch for a given epoch. """
        wandb.log({
            f"patches/noise_{noise_i}/epoch_{epoch}": 
                wandb.Image(patch.cpu(), caption=f'Patch for epoch {epoch}')
        })
    

    def log_modified_image(self, patch_i, image_idx, modified_image, is_misclassified, target_model):
        """ Log modified image with adversarial patch (only during testing). """
        wandb.log({
            f"{target_model}_images/image_{image_idx}": 
                wandb.Image(modified_image.cpu(), caption=f'Patch {patch_i}, {"misclassified" if is_misclassified else "not misclassified"}')
        })
    

    def log_batch_metrics(self, epoch, loss, batch_asr, batch):
        """ Log batch-level metrics """
        log_dict = {
            f"training_detailed/epoch_{epoch}/loss": loss,
            f"training_detailed/epoch_{epoch}/ASR": batch_asr * 100,
        }

        if batch == 1:
            log_dict["batch"] = batch

        wandb.log(log_dict)    


    def log_epoch_metrics(self, loss, avg_asr, epoch):
        """ Log epoch-level metrics. """
        wandb.log({
            f"training/avg_loss": loss,
            f"training/avg_ASR": avg_asr * 100,
            "epoch" : epoch
        })

    
    def log_target_model_results(self, train_model_names, patch_i, target_model_name, misclassified, total, asr):
        """ Log test results on target models. """
        
        self.target_model_results_table.add_data(
            train_model_names, self.target_class_name, target_model_name, patch_i, misclassified, total, f'{asr * 100:.2f}%'
        )
 
    
    def log_best_generator(self, generator_name, epoch, asr):
        """ Log best generator info for an iteration. """
        self.best_generators_table.add_data(self.target_class_name, epoch, f'{asr * 100:.2f}%')
        
        wandb.log({
            f"summary/best_epoch": epoch,
            f"summary/best_ASR": asr * 100,
        })

        output_dir = "checkpoints/best_generators"
        generator_path = os.path.join(output_dir, f'{generator_name}.pth')

        # Log model as an artifact
        artifact = wandb.Artifact(name=f"best_epoch_{epoch}", type="model")
        artifact.add_file(generator_path)
        self.run.log_artifact(artifact)

        print(f"Generator saved to W&B as {generator_name}!")    



    def log_best_patch(self, noise_i, patch, testing=False):
        """ (Different from `log_patch_image`) Log the best-performing patch for each noise in a separate section, for better visualization. """
        if testing:
            wandb.log({
                f"{noise_i}": 
                    wandb.Image(patch.cpu(), caption=f'Best patch for {noise_i}')
            })
            
        else:  # Training
            wandb.log({
                f"best_patches/noise_{noise_i}": 
                    wandb.Image(patch.cpu(), caption=f'Best patch for noise #{noise_i}')
            })

            tensor_path = os.path.join('checkpoints', f"best_patch_{noise_i}.pt")
            torch.save(patch.cpu(), tensor_path)

            # Save the best patches as artifacts
            artifact = wandb.Artifact(name=f"patch_{noise_i}", type="patch_tensor")
            artifact.add_file(tensor_path)
            self.run.log_artifact(artifact)    
        
    
    def finalize(self):
        """ Log final tables and finish the run. """
        if self.run_mode == "train":
            wandb.log({"generator_evaluation_results": self.generator_eval_table})
            wandb.log({"best_generators_summary": self.best_generators_table})
        else:
            wandb.log({"target_model_test_results": self.target_model_results_table})
        wandb.finish()