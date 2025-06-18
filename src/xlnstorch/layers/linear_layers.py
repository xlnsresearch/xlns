import torch
from . import LNSModule
from .. import rand

class LNSIdentity(LNSModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return input

class LNSLinear(LNSModule):

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        sqrt_k = (1.0 / in_features) ** 0.5
        self.register_parameter("weight", (rand(out_features, in_features, device=device) * 2 - 1) * sqrt_k)

        if self.has_bias:
            self.register_parameter("bias", (rand(out_features, device=device) * 2 - 1) * sqrt_k)
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)