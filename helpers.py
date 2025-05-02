
import torch
from torch import nn
import numpy as np

class LearnableLogitScaling(nn.Module):
    def __init__(
        self, 
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100
        ):
        super().__init__()
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        self.max_logit_scale = max_logit_scale

        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)

        if self.learnable: self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:              self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        clipped = torch.clip(self.log_logit_scale.exp(), max = self.max_logit_scale)
        return clipped * x
    
    def extra_repr(self):
        st = f"logit_scale_init={self.logit_scale_init},learnable={self.learnable}," \
             f" max_logit_scale={self.max_logit_scale}"
        return st

class SelectElement(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    
    def forward(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return nn.functional.normalize(x, p = 2, dim = self.dim)
