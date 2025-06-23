import torch

class LNSModule(torch.nn.Module):
    """
    An LNS module that serves as a base class for all LNS layers.

    This class extends `torch.nn.Module` and provides a mechanism to register
    LNSTensor parameters. For a given LNSTensor parameter, say `param`, it converts
    the `param._lns` attribute into a `torch.nn.Parameter` and registers it with
    the name `param_lns` in the module. This is necessary to ensure that PyTorch
    has visibility of the LNS parameters for optimization and state management.

    Examples
    --------
    >>> class CustomLayer(xlnstorch.layers.LNSModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.register_parameter("custom_param", xlnstorch.lnstensor(1.0, f=10))
    >>> layer = CustomLayer()
    >>> print(layer.custom_param) # LNSTensor(value=1.0, base=1.0006771306930664)
    >>> print(layer.custom_param_lns) # Parameter containing: tensor(0., dtype=torch.float64, requires_grad=True)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def register_parameter(self, name, param, requires_grad=True):
        setattr(self, name, param)
        param._lns = torch.nn.Parameter(param._lns, requires_grad=requires_grad)
        super().register_parameter(name + "_lns", param._lns)