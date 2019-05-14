import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        loss = nn.functional.cross_entropy(x, y)
        return loss
