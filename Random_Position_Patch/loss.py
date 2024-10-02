import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self, target_class):
        super(AdversarialLoss, self).__init__()
        self.target_class = target_class
        #self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, target_class=None):
        # log softmax
        # log_probs = torch.log_softmax(predictions, dim=1)
        
        # # we want to maximize the probability of the target class y'

        # loss = -log_probs[:, self.target_class].mean()

        log_probs = torch.log_softmax(predictions, dim=1)
        batch_indices = torch.arange(predictions.size(0), device=predictions.device)
        target_log_probs = log_probs[batch_indices, self.target_class]
        loss = -target_log_probs.mean()
        
        return loss
