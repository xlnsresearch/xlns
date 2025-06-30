import torch
from .. import LNS_ZERO

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
                param._incoming_grads = []
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        if param.grad.grad_fn is not None:
                            param.grad.detach_()
                        else:
                            param.grad.requires_grad_(False)
                        param.grad.fill_(LNS_ZERO)