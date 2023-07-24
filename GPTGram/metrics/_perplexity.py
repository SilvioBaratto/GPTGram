from ._base import BaseMetric
import torch.nn.functional as F

class Perplexity(BaseMetric):
    def score(self, logits, labels):
        """
        Calculates the perplexity for the given logits and labels.

        Args:
            logits (torch.Tensor): The logits output by the model.
            labels (torch.Tensor): The true labels.

        Returns:
            float: The calculated perplexity.
        """
        # Calculate the cross entropy loss between the logits and labels
        loss = F.cross_entropy(logits, labels, ignore_index=-1)
        
        # Perplexity is the exponential of the loss
        perplexity = loss.exp().item()
        
        return perplexity