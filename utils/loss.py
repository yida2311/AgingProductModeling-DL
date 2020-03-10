import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCELoss()
    
    def forward(self, preds, labels):
        loss = self.loss(preds, labels)
        return loss.mean()
