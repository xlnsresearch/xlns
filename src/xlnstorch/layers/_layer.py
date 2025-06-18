import torch

class LNSModule(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_parameter(self, name, param, requires_grad=True):
        setattr(self, name, param)
        super().register_parameter(
            name + "_lns", torch.nn.Parameter(param._lns, requires_grad=requires_grad)
        )