import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    """ 
    Adversarial loss for targeted attacks.
    Maximizes the log-probability of the target class (i.e., encourages misclassification into that class).
    """

    def __init__(self, target_class):
        super(AdversarialLoss, self).__init__()
        self.target_class = target_class

    def forward(self, predictions):
        log_probs = torch.log_softmax(predictions, dim=1)
        batch_indices = torch.arange(predictions.size(0), device=predictions.device)
        target_log_probs = log_probs[batch_indices, self.target_class]
        
        loss = -target_log_probs.mean() # Negative to make it a loss (maximize target class probability)
        
        return loss
