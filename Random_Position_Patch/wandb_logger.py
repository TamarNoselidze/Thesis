import wandb
from helper import get_class_name

class WandbLogger:
    def __init__(self, project_name, config):
        self.run = wandb.init(
            project=project_name, 
            entity='takonoselidze-charles-university', 
            config=config
        )
        
        # Create Tables for structured logging
        self.create_tables()
        
    def create_tables(self):
        """Create tables for structured logging"""
        # Table for tracking generator evaluation results
        self.generator_eval_table = wandb.Table(
            columns=["iteration", "target_class", "target_class_name", "epoch", "asr", "is_best"]
        )
        
        # Table for tracking test results on target models
        self.target_model_results_table = wandb.Table(
            columns=["iteration", "target_class", "target_class_name", 
                    "target_model", "train_models", "misclassified", "total", "asr"]
        )
        
        # Table to track best generators per iteration
        self.best_generators_table = wandb.Table(
            columns=["iteration", "target_class", "target_class_name", "best_epoch", "best_asr"]
        )
    
    def log_generator_evaluation(self, iteration, target_class, epoch, asr, is_best=False):
        """Log individual generator evaluation result"""
        class_name = get_class_name(target_class)
        self.generator_eval_table.add_data(iteration, target_class, class_name, epoch, asr, is_best)
        
        # Also log as metrics for easier time-series visualization
        wandb.log({
            f"eval/iter_{iteration}/target_{target_class}/epoch_{epoch}_asr": asr,
            "iteration": iteration
        })
        
        if is_best:
            wandb.log({
                f"eval/iter_{iteration}/target_{target_class}/best_asr": asr,
                f"eval/iter_{iteration}/target_{target_class}/best_epoch": epoch,
                "iteration": iteration
            })
    
    def log_patch_image(self, iteration, target_class, epoch, patch, is_best=False):
        """Log patch image"""
        class_name = get_class_name(target_class)
        caption_prefix = "Best " if is_best else ""
        wandb.log({
            f"patches/iter_{iteration}/target_{target_class}/{caption_prefix}epoch_{epoch}": 
                wandb.Image(patch.cpu(), caption=f'{caption_prefix}Patch for epoch {epoch}, target class "{class_name}" ({target_class})')
        })
    
    def log_modified_image(self, iteration, image_idx, modified_image, target_class):
        """Log modified image with adversarial patch"""
        class_name = get_class_name(target_class)
        wandb.log({
            f"images/iter_{iteration}/image_{image_idx}": 
                wandb.Image(modified_image.cpu(), caption=f'Target class "{class_name}" ({target_class})')
        })
    
    def log_batch_metrics(self, iteration, epoch, batch, loss, batch_asr):
        """Log batch-level metrics"""
        wandb.log({
            f"training/iter_{iteration}/epoch_{epoch}/batch_{batch}/loss": loss,
            f"training/iter_{iteration}/epoch_{epoch}/batch_{batch}/asr": batch_asr,
            "iteration": iteration,
            "epoch": epoch,
            "batch": batch
        })
    
    def log_epoch_metrics(self, iteration, epoch, loss, avg_asr):
        """Log epoch-level metrics"""
        wandb.log({
            f"training/iter_{iteration}/epoch_{epoch}/loss": loss,
            f"training/iter_{iteration}/epoch_{epoch}/avg_asr": avg_asr,
            "iteration": iteration,
            "epoch": epoch
        })
    
    def log_target_model_results(self, iteration, target_class, target_model, train_model_names, misclassified, total, asr):
        """Log test results on target models"""
        class_name = get_class_name(target_class)
        train_models_str = ", ".join(train_model_names)
        
        self.target_model_results_table.add_data(
            iteration, target_class, class_name, target_model, train_models_str, misclassified, total, asr
        )
        
        wandb.log({
            f"test/iter_{iteration}/target_{target_class}/model_{target_model}/asr": asr,
            "iteration": iteration
        })
    
    def log_best_generator(self, iteration, target_class, best_epoch, best_asr):
        """Log best generator info for an iteration"""
        class_name = get_class_name(target_class)
        self.best_generators_table.add_data(iteration, target_class, class_name, best_epoch, best_asr)
        
        wandb.log({
            f"summary/iter_{iteration}/target_{target_class}/best_epoch": best_epoch,
            f"summary/iter_{iteration}/target_{target_class}/best_asr": best_asr,
            "iteration": iteration
        })
    
    def finalize(self):
        """Log final tables and finish the run"""
        wandb.log({"generator_evaluation_results": self.generator_eval_table})
        wandb.log({"target_model_test_results": self.target_model_results_table})
        wandb.log({"best_generators_summary": self.best_generators_table})
        wandb.finish()