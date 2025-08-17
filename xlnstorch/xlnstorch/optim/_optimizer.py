import torch
from xlnstorch import LNS_ZERO, lnstensor

class LNSOptimizer(torch.optim.Optimizer):

    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def zero_grad(self, set_to_none: bool = True):
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for param in group['params']:
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

    def make_lnstensor_params(self, *param_names):
        """Convert specified parameters in defaults to LNS tensors."""
        for group in self.param_groups:
            base = group["base"]
            for name in param_names:
                if name in group:
                    group[name] = lnstensor(group[name], b=base)._lns