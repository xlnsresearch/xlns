import torch
from xlnstorch import lnstensor, LNS_ZERO, LNS_ONE
from xlnstorch.operators import (
    lns_sub,
    lns_equal,
    lns_neg,
    lns_mul,
    lns_add,
    lns_div,
    lns_pow,
    lns_maximum,
    lns_gt,
)
from . import LNSOptimizer

class LNSASGD(LNSOptimizer):
    """
    Implements the ASGD algorithm for LNSTensor parameters,
    including optional weight decay and a "maximize" mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.ASGD`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the ASGD algorithm.

    Note that this optimizer doesn't seem to implement the ASGD algorithm
    correctly, but it is made to match the PyTorch implementation as
    closely as possible.

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.001). Must be a non-negative LNSTensor or float.
    lambd : LNSTensor, float, optional
        Coefficient for the learning rate decay (default: 0.0001). Must be a non-negative
        LNSTensor or float.
    alpha : LNSTensor, float, optional
        Exponent for the learning rate decay (default: 0.75). Must be a non-negative
        LNSTensor or float in the range (0.0, 1.0].
    t0 : LNSTensor, float, optional
        The point at which the learning rate decay starts (default: 1000000.0).
        Must be a non-negative LNSTensor or float.
    weight_decay : LNSTensor or float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            lambd=0.0001,
            alpha=0.75,
            t0=1000000.0,
            weight_decay=0.0,
            *,
            maximize=False
        ):

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if lambd < 0.0:
            raise ValueError(f"Invalid lambda value: {lambd}")

        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Invalid alpha value: {alpha}")

        if t0 < 0.0:
            raise ValueError(f"Invalid t0 value: {t0}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(LNSASGD, self).__init__(params, defaults)
        self.make_lnstensor_params("lr", "lambd", "alpha", "t0", "weight_decay")

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lambd = group["lambd"]
            alpha = group["alpha"]
            t0 = group["t0"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            base = group["base"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if maximize:
                    grad = lns_neg(grad)

                if not lns_equal(weight_decay, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay, base), base)

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = LNS_ZERO.clone()
                    state["averaging_coef"] = LNS_ONE.clone()
                    state["averaged_param"] = p.data.clone()

                # Retrieve running stats
                step = lns_add(state["step"], LNS_ONE, base)
                averaging_coef = state["averaging_coef"]
                averaged_param = state["averaged_param"]

                # 1. learning-rate schedule
                denom = lns_add(LNS_ONE, lns_mul(lambd, lns_mul(lr, step, base), base), base)
                denom = lns_pow(denom, lnstensor(alpha, from_lns=True, b=base).value, base)
                current_lr = lns_div(lr, denom, base)

                # 2. update averaged parameter
                decay = lns_sub(LNS_ONE, lns_mul(lambd, current_lr, base), base)
                p.data = lns_mul(p.data, decay, base)
                p.data = lns_sub(p.data, lns_mul(grad, current_lr, base), base)

                # 3. update averaged parameter
                if lns_gt(step, t0):
                    denom = lns_maximum(LNS_ONE, lns_sub(step, t0, base), base)
                    averaging_coef = lns_div(LNS_ONE, denom, base)

                diff = lns_sub(p.data, averaged_param, base)
                averaged_param = lns_add(averaged_param, lns_mul(diff, averaging_coef, base), base)

                state["step"] = step
                state["averaging_coef"] = averaging_coef
                state["averaged_param"] = averaged_param