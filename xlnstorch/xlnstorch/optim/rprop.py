import torch
from xlnstorch import LNS_ZERO, LNS_ONE, LNS_NEG_ONE
from . import LNSOptimizer

class LNSRprop(LNSOptimizer):
    """
    Implements the Rprop (resilient backpropagation) algorithm.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.Rprop`, but
    is designed to work with LNSTensor objects. See the PyTorch documentation
    for more details on the Rprop algorithm.

    Parameters
    -----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    etas : tuple of LNSTensor, tuple of float, optional
        Tuple of two factors (η₋, η₊) for decreasing and increasing the step size
        (default: (0.5, 1.2)). Must satisfy 0 < η₋ < 1 and η₊ > 1.
    step_sizes : tuple of LNSTensor, tuple of float, optional
        Tuple of two step sizes (Γ_min, Γ_max) for clamping the step size
        (default: (1e-6, 50.0)). Must satisfy 0 < Γ_min < Γ_max.
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-6, 50.0),
            *,
            maximize=False,
        ):
        defaults = dict(
            lr=lr,
            eta_minus=etas[0],
            eta_plus=etas[1],
            step_min=step_sizes[0],
            step_max=step_sizes[1],
            maximize=maximize,
        )
        super(LNSRprop, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "eta_minus", "eta_plus", "step_min", "step_max")

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("eta_minus", lambda eta_minus: 0.0 < eta_minus < 1.0)
        self.validate_param("eta_plus", lambda eta_plus: eta_plus > 1.0)
        self.validate_param("step_min", lambda step_min: step_min > 0.0)
        self.validate_param("step_max", lambda step_max: step_max > 0.0)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            eta_m = group["eta_minus"]
            eta_p = group["eta_plus"]
            step_min = group["step_min"]
            step_max = group["step_max"]
            maximize = group["maximize"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)

                if maximize:
                    grad = ops.neg(grad)

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    state["prev_grad"] = ops.zeros_like(data)
                    state["step_size"] = p.clone().fill_(lr)

                # Retrieve running stats
                prev_grad = state["prev_grad"]
                step_size = state["step_size"]

                # 1. Element-wise sign comparison of grads
                grad_prod = ops.mul(prev_grad, grad)
                grad_prod_sign = ops.sign(grad_prod)

                # positive mask and clamp to Γ_max: η ← η * η_+
                pos_mask = ops.eq(grad_prod_sign, LNS_ONE)
                step_size_pos = ops.mul(step_size, eta_p)
                step_size_pos = ops.minimum(step_size_pos, step_max)
                step_size = torch.where(pos_mask, step_size_pos, step_size)

                # negative mask and clamp to Γ_min: η ← η * η_-
                neg_mask = ops.eq(grad_prod_sign, LNS_NEG_ONE)
                step_size_neg = ops.mul(step_size, eta_m)
                step_size_neg = ops.maximum(step_size_neg, step_min)
                step_size = torch.where(neg_mask, step_size_neg, step_size)
                grad = torch.where(neg_mask, LNS_ZERO, grad) # when flipped signs, ignore grad

                # 2. Parameter update: θ ← θ - sign(g_t) * η_t
                grad_sign = ops.sign(grad)
                delta = ops.mul(step_size, grad_sign)
                p.data = ops.sub(data, delta).view(torch.float64)

                state["step_size"] = step_size
                state["prev_grad"] = grad.clone()

        return loss