import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like, full_like
from ..operators import (
    lns_sub,
    lns_mul,
    lns_neg,
    lns_sign,
    lns_eq,
    lns_minimum,
    lns_maximum,
)
from . import LNSOptimizer

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

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
        This should be obtained from a model's `parameter_groups()` method.
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

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if (etas[0] <= 0 or etas[0] >= 1 or etas[1] <= 1):
            raise ValueError(
                f"Invalid etas tuple: η₋ must be in (0,1), η₊ must be > 1, got {etas}"
            )

        if step_sizes[0] <= 0 or step_sizes[0] >= step_sizes[1]:
            raise ValueError(
                f"Invalid step_sizes tuple: must satisfy 0 < Γ_min < Γ_max, got {step_sizes}"
            )

        defaults = dict(
            lr=_as_lnstensor(lr),
            eta_minus=_as_lnstensor(etas[0]),
            eta_plus=_as_lnstensor(etas[1]),
            step_min=_as_lnstensor(step_sizes[0]),
            step_max=_as_lnstensor(step_sizes[1]),
            maximize=maximize,
        )
        super(LNSRprop, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eta_m = group["eta_minus"]
            eta_p = group["eta_plus"]
            step_min = group["step_min"]
            step_max = group["step_max"]
            maximize = group["maximize"]
            base = group["base"]

            lr, eta_m, eta_p, step_min, step_max = align_lnstensor_bases(
                lr, eta_m, eta_p, step_min, step_max, base=base
            )

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad # g_t

                if maximize:
                    grad = lns_neg(grad)

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    state["prev_grad"] = zeros_like(p, b=base)._lns
                    state["step_size"] = p.clone().fill_(lr._lns)

                # Retrieve running stats
                prev_grad = state["prev_grad"]
                step_size = state["step_size"]

                # 1. Element-wise sign comparison of grads
                grad_prod = lns_mul(prev_grad, grad)
                grad_prod_sign = lns_sign(grad_prod, base)

                # positive mask and clamp to Γ_max: η ← η * η_+
                pos_mask = lns_eq(grad_prod_sign, LNSTensor.get_internal_tensor(1.0, base))
                step_size_pos = lns_mul(step_size, eta_p._lns)
                step_size_pos = lns_minimum(step_size_pos, step_max._lns, base)
                step_size = torch.where(pos_mask, step_size_pos, step_size)

                # negative mask and clamp to Γ_min: η ← η * η_-
                neg_mask = lns_eq(grad_prod_sign, LNSTensor.get_internal_tensor(-1.0, base))
                step_size_neg = lns_mul(step_size, eta_m._lns)
                step_size_neg = lns_maximum(step_size_neg, step_min._lns, base)
                step_size = torch.where(neg_mask, step_size_neg, step_size)
                grad = torch.where(neg_mask, LNS_ZERO, grad) # when flipped signs, ignore grad

                # 2. Parameter update: θ ← θ - sign(g_t) * η_t
                grad_sign = lns_sign(grad, base)
                delta = lns_mul(step_size, grad_sign)
                p.data = lns_sub(p.data, delta, base)

                state["step_size"] = step_size
                state["prev_grad"] = grad.clone()

        return loss