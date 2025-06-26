import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like
from ..operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_neg,
    lns_sqrt,
    lns_div,
)

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSRMSprop(torch.optim.Optimizer):
    """
    Implements the RMSprop algorithm with support for weight decay,
    momentum, and centered variants.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.RMSprop`,
    but is designed to work with LNSTensor objects. See the PyTorch documentation
    for more details on the RMSprop algorithm.

    Parameters:
    -----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `parameter_groups()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    alpha : LNSTensor, float, optional
        Smoothing constant (default: 0.99). Must be a non-negative LNSTensor or float
        in the range [0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator to improve numerical stability (default: 1e-08).
        Must be a non-negative LNSTensor or float.
    weight_decay : LNSTensor, float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    momentum : LNSTensor, float, optional
        Momentum factor (default: 0.0). Must be a non-negative LNSTensor or float.
    centered : bool, optional
        If True, computes the centered RMSprop variant (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0.0,
            momentum=0.0,
            centered=False,
            *,
            maximize=False,
        ):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if not (0.0 <= alpha < 1.0):
            raise ValueError("Invalid alpha value: {}".format(alpha))

        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(
            lr=_as_lnstensor(lr),
            alpha=_as_lnstensor(alpha),
            eps=_as_lnstensor(eps),
            weight_decay=_as_lnstensor(weight_decay),
            momentum=_as_lnstensor(momentum),
            centered=centered,
            maximize=maximize
        )
        super(LNSRMSprop, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay= group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]
            maximize = group["maximize"]
            base = group["base"]

            lr, alpha, eps, weight_decay, momentum = align_lnstensor_bases(
                lr, alpha, eps, weight_decay, momentum, base=base
            )

            one = LNSTensor.get_internal_tensor(1.0, base)
            one_minus_alpha = lns_sub(one, alpha._lns, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad  = p.grad # g_t
                state = self.state[p]

                if maximize:
                    grad = lns_neg(grad)

                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay._lns), base)

                if len(state) == 0:
                    # First time we see this parameter
                    zeros = zeros_like(p.data, b=base)._lns
                    state["square_avg"] = zeros.clone()
                    state["grad_avg"] = zeros.clone()
                    state["momentum_buffer"] = zeros.clone()

                # Retrieve running stats
                square_avg = state["square_avg"]
                grad_avg = state["grad_avg"]
                buf = state["momentum_buffer"]

                # 1. square_avg: v_t ← α v_{t-1} + (1-α) * g_t²
                grad_sq = lns_mul(grad, grad)
                square_avg = lns_add(
                    lns_mul(square_avg, alpha._lns), # α v_{t-1}
                    lns_mul(grad_sq, one_minus_alpha), # (1-α) g_t²
                    base
                )

                # 2. centered:
                # g_avg ← α g_avg + (1-α) * g_t
                # v'_t = v_t - (g_avg) ^ 2
                if centered:
                    grad_avg = lns_add(
                        lns_mul(grad_avg, alpha._lns), # α g_avg
                        lns_mul(grad, one_minus_alpha), # (1-α) * g_t
                        base
                    )
                    avg_sq = lns_mul(grad_avg, grad_avg)
                    denom = lns_sub(square_avg, avg_sq, base)

                else:
                    denom = square_avg

                # denominator: sqrt((v'_t) + ε)
                denom = lns_add(lns_sqrt(denom, base), eps._lns, base)

                # 3. momentum buffering:
                if not lns_equal(momentum._lns, LNS_ZERO):
                    # b_t ← μ b_{t-1} + g_t / denom
                    buf_div = lns_div(grad, denom, base)
                    buf = lns_add(lns_mul(buf, momentum._lns), buf_div, base)

                    # θ ← θ - γ * b_t
                    delta = lns_mul(buf, lr._lns)

                else:
                    # θ ← θ - γ * g_t / denom
                    step_dir = lns_div(grad, denom, base)
                    delta = lns_mul(step_dir, lr._lns)

                # 4. parameter update: θ ← θ - γ * b_t
                p.data = lns_sub(p.data, delta, base)

                state["square_avg"] = square_avg
                state["grad_avg"] = grad_avg
                state["momentum_buffer"] = buf

        return loss