import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases
from ..operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_div,
    lns_sqrt,
    lns_neg,
)

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSAdagrad(torch.optim.Optimizer):
    """
    Implements the Adagrad algorithm with support for learning rate decay,
    weight decay, and an initial accumulator value.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.Adagrad`,
    but is designed to work with LNSTensor objects. See the PyTorch documentation
    for more details on the Adagrad algorithm.

    Parameters
    -----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `parameter_groups()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    lr_decay : LNSTensor, float, optional
        Learning rate decay factor (default: 0.0). Must be a non-negative LNSTensor
        or float.
    weight_decay : LNSTensor, float, optional
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor
        or float.
    initial_accumulator_value : LNSTensor, float, optional
        Initial value for the accumulator (default: 0). Must be a non-negative
        LNSTensor or float.
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-10).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization
        (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            lr_decay=0.0,
            weight_decay=0.0,
            initial_accumulator_value=0,
            eps=1e-10,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if lr_decay < 0.0:
            raise ValueError(f"Invalid learning rate decay: {lr_decay}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial accumulator value: {initial_accumulator_value}")

        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=_as_lnstensor(lr),
            lr_decay=_as_lnstensor(lr_decay),
            weight_decay=_as_lnstensor(weight_decay),
            initial_accumulator_value=_as_lnstensor(initial_accumulator_value),
            eps=_as_lnstensor(eps),
            maximize=maximize
        )
        super(LNSAdagrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            init_acc_val = group["initial_accumulator_value"]
            eps = group["eps"]
            maximize = group["maximize"]
            base = group["base"]

            lr, lr_decay, weight_decay, init_acc_val, eps = align_lnstensor_bases(
                lr, lr_decay, weight_decay, init_acc_val, eps, base=base
            )

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad # g_t
                state = self.state[p]

                if maximize:
                    grad = lns_neg(grad)

                # 1. State initialisation (run the first time we see this parameter)
                if len(state) == 0:
                    state["step"] = LNS_ZERO.clone()
                    state["sum"] = torch.full_like(p, init_acc_val._lns)

                state["step"] = lns_add(state["step"], LNSTensor.get_internal_tensor(1.0, base), base)
                step = state["step"] # t

                # 2. step lr: γ' ← γ / (1 + (t − 1) * η)
                if not lns_equal(lr_decay._lns, LNS_ZERO):
                    denom = lns_add(
                        LNSTensor.get_internal_tensor(1.0, base),
                        lns_mul(lr_decay._lns, lns_sub(
                            step, LNSTensor.get_internal_tensor(1.0, base), base)), base)
                    lr_t = lns_div(lr._lns, denom, base)

                else:
                    lr_t = lr._lns

                # 3. weight decay: g_t ← g_t + λ*θ_{t-1}
                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay._lns), base)

                # 4. Accumulator update: s_t ← s_{t-1} + g_t^2
                s_prev = state["sum"]
                s_t = lns_add(s_prev, lns_mul(grad, grad), base)
                state["sum"] = s_t

                # 5. Parameter update: θ_t ← θ_{t-1} ± γ' * g_t / (sqrt(s_t) + ε)
                sqrt_s_t = lns_sqrt(s_t, base)
                denom = lns_add(sqrt_s_t, eps._lns, base)
                delta = lns_div(lns_mul(grad, lr_t), denom, base)
                p.data = lns_sub(p.data, delta, base)

        return loss