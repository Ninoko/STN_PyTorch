import math
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 0.5 * x * (1 + self.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
