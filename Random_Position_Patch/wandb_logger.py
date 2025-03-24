import wandb
from helper import get_class_name

class WandbLogger:
    def __init__(self, project_name, target_class, config):
        self.run = wandb.init(
            project=project_name, 
            entity='takonoselidze-charles-university', 
            config=config
        )
        self.target_class_name = get_class_name(target_class)
        self.create_tables()
        

    def create_tables(self):
        """Tables for structured logging"""
        # Table for tracking generator evaluation results
        self.generator_eval_table = wandb.Table(
            columns=["iteration", "target class name", "random noise #", "epoch", "ASR"]
        )
        
        # Table for tracking test results on target models
        self.target_model_results_table = wandb.Table(
            columns=["iteration", "target class name", "target model(s)", "train model(s)",
                     "random noise #" , "misclassified", "total", "ASR"]
        )
        
        # Table to track best generators per iteration
        self.best_generators_table = wandb.Table(
            columns=["iteration", "target class_name", "best epoch", "best ASR"]
        )
    

    def log_generator_evaluation(self, iteration, noise_i, epoch, asr):
        """Log individual generator evaluation result"""
        self.generator_eval_table.add_data(iteration, self.target_class_name, noise_i, epoch, f'{asr * 100:.2f}%')
        # Also log as metrics for easier time-series visualization
        wandb.log({
            f"eval/iter_{iteration}/noise_{noise_i}/epoch_ASR": asr * 100,
        })
        
    
    def log_patch_image(self, iteration, noise_i, epoch, patch):

        wandb.log({
            f"patches/iter_{iteration}/noise_{noise_i}/epoch_{epoch}": 
                wandb.Image(patch.cpu(), caption=f'Patch for epoch {epoch}')
        })
    

    def log_modified_image(self, iteration, noise_i, image_idx, modified_image, is_misclassified):
        """Log modified image with adversarial patch"""
        wandb.log({
            f"image_{image_idx}": 
                wandb.Image(modified_image.cpu(), caption=f'Iteration {iteration}, Noise {noise_i} {"misclassified" if is_misclassified else "not misclassified"}')
        })
    

    def log_batch_metrics(self, iteration, epoch, loss, batch_asr):
        """Log batch-level metrics"""
        wandb.log({
            f"training/iter_{iteration}/epoch_{epoch}/loss": loss,
            f"training/iter_{iteration}/epoch_{epoch}/ASR": batch_asr * 100,
        })

    
    def log_epoch_metrics(self, iteration, loss, avg_asr):
        """Log epoch-level metrics"""
        wandb.log({
            f"training/iter_{iteration}/avg_loss": loss,
            f"training/iter_{iteration}/avg_ASR": avg_asr * 100,
        })

    
    def log_target_model_results(self, iteration, noise_i, target_models, train_model_names, misclassified, total, asr):
        """Log test results on target models"""
        train_models_str = ", ".join(train_model_names)
        
        self.target_model_results_table.add_data(
            iteration, self.target_class_name, target_models, train_models_str, noise_i, misclassified, total, f'{asr * 100:.2f}%'
        )
 
    
    def log_best_generator(self, iteration, epoch, asr):
        """Log best generator info for an iteration"""
        self.best_generators_table.add_data(iteration, self.target_class_name, epoch, asr)
        
        wandb.log({
            f"summary/iter_{iteration}/best_epoch": epoch,
            f"summary/iter_{iteration}/best_ASR": asr,
        })


    def log_best_patch(self, iteration, noise_i, epoch, patch):
        wandb.log({
            f"patches/iter_{iteration}/noise_{noise_i}/epoch_{epoch}": 
                wandb.Image(patch.cpu(), caption=f'Patch for epoch {epoch}')
        })
    

    def finalize(self):
        """Log final tables and finish the run"""
        wandb.log({"generator_evaluation_results": self.generator_eval_table})
        wandb.log({"target_model_test_results": self.target_model_results_table})
        wandb.log({"best_generators_summary": self.best_generators_table})
        wandb.finish()