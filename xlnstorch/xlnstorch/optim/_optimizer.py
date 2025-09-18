import torch
from xlnstorch import LNS_ZERO_FP, lnstensor
from xlnstorch.ops import LNSOps

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
                param._lns_grad._lns.fill_(LNS_ZERO_FP)
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        if param.grad.grad_fn is not None:
                            param.grad.detach_()
                        else:
                            param.grad.requires_grad_(False)
                        param.grad.fill_(LNS_ZERO_FP)

    def make_lnstensor_params(self, *param_names):
        """Convert specified parameters in defaults to LNS tensors."""
        for group in self.param_groups:
            base = group["base"]
            for name in param_names:
                if name in group:
                    group[name] = lnstensor(group[name], b=base).lns # .lns views to int64

    def validate_param(self, param_name, condition):
        """Validate a parameter in all parameter groups."""
        for group in self.param_groups:

            if param_name not in group:
                continue

            group_value = lnstensor(group[param_name], from_lns=True, b=group["base"])
            valid = condition(group_value)

            if not valid:
                str_val = group_value.item() if group_value.numel() == 1 else group_value
                raise ValueError(f"Invalid {param_name}: {str_val}")

    def lns_param_groups(self):
        for group in self.param_groups:
            yield group, LNSOps(group["base"])

    def lns_ops(self):
        for group in self.param_groups:
            yield LNSOps(group["base"])