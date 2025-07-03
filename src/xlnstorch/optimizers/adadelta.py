import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like
from ..operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_neg,
    lns_div,
    lns_sqrt,
)
from . import LNSOptimizer

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSAdadelta(LNSOptimizer):
    """
    Implements the LNSAdadelta algorithm for LNSTensor parameters,
    supporting weight decay and a "maximize" mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.Adamax`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the Adamax algorithm.

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `parameter_groups()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.002). Must be a non-negative LNSTensor or float.
    rho : LNSTensor, float, optional
        Coefficient used for computing running averages of gradient (default: 0.9).
        Must be a non-negative LNSTensor or float in the range (0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-6).
        Must be a non-negative LNSTensor or float.
    weight_decay : LNSTensor or float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=1.0,
            rho=0.9,
            eps=1e-6,
            weight_decay=0.0,
            *,
            maximize=False
        ):

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not (0.0 < rho < 1.0):
            raise ValueError(f"Invalid rho value: {rho}")

        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=_as_lnstensor(lr),
            rho=_as_lnstensor(rho),
            eps=_as_lnstensor(eps),
            weight_decay=_as_lnstensor(weight_decay),
            maximize=maximize,
        )
        super(LNSAdadelta, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            base = group["base"]

            lr, rho, eps, weight_decay = align_lnstensor_bases(
                lr, rho, eps, weight_decay, base=base
            )

            one = LNSTensor.get_internal_tensor(1.0, base)
            one_minus_rho = lns_sub(one, rho._lns, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad # g_t
                state = self.state[p]

                if maximize:
                    grad = lns_neg(grad)

                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay._lns), base)

                if len(state) == 0:
                    # First time we see this parameter
                    zeros = zeros_like(p.data, b=base)._lns
                    state["square_avg"] = zeros.clone()
                    state["acc_delta"] = zeros.clone()

                # Retrieve running stats
                square_avg = state["square_avg"] # E[g^2]
                acc_delta = state["acc_delta"] # E[Δ^2]

                # 1. square average: v_t ← ρ v_{t-1} + (1-ρ) g_t^2
                grad_sq = lns_mul(grad, grad)
                square_avg = lns_add(
                    lns_mul(square_avg, rho._lns),
                    lns_mul(grad_sq, one_minus_rho),
                    base
                )

                # 2. Compute update: Δx_t ← sqrt((acc_delta + ε) / (square_avg + ε)) * g_t
                numer = lns_add(acc_delta, eps._lns, base)
                denom = lns_add(square_avg, eps._lns, base)
                rms_ratio = lns_sqrt(lns_div(numer, denom, base), base)
                delta = lns_mul(rms_ratio, grad)

                # 3. accumulate delta: u_t ← ρ u_{t-1} + (1-ρ) Δx_t^2
                delta_sq = lns_mul(delta, delta)
                acc_delta = lns_add(
                    lns_mul(acc_delta, rho._lns),
                    lns_mul(delta_sq, one_minus_rho),
                    base
                )

                # 4. Parameter update: θ ← θ - η * Δx_t
                step = lns_mul(delta, lr._lns)
                p.data = lns_sub(p.data, step, base)

                state["square_avg"] = square_avg
                state["acc_delta"] = acc_delta

        return loss