import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
