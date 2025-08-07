from collections import OrderedDict
import torch
from xlnstorch import LNS_ZERO

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
    >>> class CustomLayer(xlnstorch.nn.LNSModule):
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
        """
        Registers a parameter in the module.

        This method overrides the default `register_parameter` method to handle
        LNSTensor parameters specifically. It converts the `_lns` attribute of the
        LNSTensor into a `torch.nn.Parameter` and registers it with the name
        `name + "_lns"`. The base value of the LNSTensor is registered as a buffer
        with the name `name + "_base"`.
        """
        setattr(self, name, param)
        param._lns = torch.nn.Parameter(param._lns, requires_grad=requires_grad)
        if requires_grad:
            param.register_grad_hook()
        super().register_parameter(name + "_lns", param._lns)
        super().register_buffer(name + "_base", param.base)

    def lns_parameters(self):
        """
        Returns a list of parameter groups for the module.
        Each group contains parameters and their corresponding base values.
        """
        for full_name, param in self.named_parameters():

            name_split = full_name.rsplit('.', 1)
            if len(name_split) == 1:
                submodule = self
                name = name_split[0]
            else:
                name = name_split[1]
                submodule = self.get_submodule(name_split[0])

            if name.endswith("_lns") and hasattr(submodule, name[:-4]):
                base_name = name[:-4] + "_base"
                yield {
                    "params": param,
                    "base": getattr(submodule, base_name)
                }

            else:
                yield {
                    "params": param,
                }

    def zero_grad(self, set_to_none: bool = True):
        """Clears the gradients of all parameters in the module."""
        for param in self.parameters():
            param._lns_grad._lns.fill_(LNS_ZERO)
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.fill_(LNS_ZERO)

class LNSSequential(torch.nn.Sequential, LNSModule):
    pass
    # This class doesn't need to implement any additional functionality
    # beyond what is provided by torch.nn.Sequential and LNSModule.

    # It inherits the behavior of both classes, allowing it to be used
    # as a sequential container for LNS layers.

    # The following commented-out code is an alternative implementation
    # that could be used if needed, but it is not necessary for the current
    # functionality of LNSSequential.

    # def __init__(self, *args):
    #     super().__init__()

    #     if len(args) == 1 and isinstance(args[0], OrderedDict):
    #         for key, module in args[0].items():
    #             self.add_module(key, module)

    #     else:
    #         for idx, module in enumerate(args):
    #             self.add_module(str(idx), module)

    # def forward(self, input):
    #     for module in self._modules.values():
    #         input = module(input)
    #     return input