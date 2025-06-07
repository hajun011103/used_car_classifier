# loss.py (Focal Loss)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: focusing parameter
        alpha: class weight tensor or scalar (e.g., [0.25, 0.75])
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        true_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        true_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - true_probs) ** self.gamma

        loss = -alpha_t * focal_term * true_log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_class_alpha():
    """
    Computes alpha (class weights) based on inverse frequency
    """
    class_counts = [0] * config.NUM_CLASSES

    for _, label in config.dataset.samples:
        class_counts[label] += 1

    class_counts = np.array(class_counts)
    class_counts[class_counts == 0] = 1  # avoid divide-by-zero

    alpha = 1.0 / class_counts
    alpha = alpha / alpha.sum()  # normalize to sum to 1

    return torch.tensor(alpha, dtype=torch.float32)
