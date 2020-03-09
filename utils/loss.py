import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        self.loss = nn.BCELoss()
    
    def forward(self, preds, labels):
        loss = self.loss(preds, labels)
        return loss.mean()
