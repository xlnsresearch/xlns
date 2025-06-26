import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like
from ..operators import (
    lns_sub,
    lns_equal,
    lns_mul,
    lns_add,
    lns_div,
    lns_pow,
    lns_maximum,
    lns_abs,
)

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSAdamax(torch.optim.Optimizer):
    """
    Implements the Adamax optimization algorithm for LNSTensor parameters,
    including optional weight–decay regularisation, and a “maximize” mode.

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
    betas : Tuple[float, float] or Tuple[LNSTensor, LNSTensor], optional
        Coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999)). Must be two non-negative LNSTensor or float values
        in the range [0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-8).
    weight_decay : LNSTensor or float
        Weight decay (L2 penalty) (default: 0.0). Must be a non-negative LNSTensor or float.
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """
    def __init__(
            self,
            params,
            lr=0.002,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid β1 value: {betas[0]}")

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid β2 value: {betas[1]}")

        if eps <= 0.0:
            raise ValueError(f"Invalid ε value: {eps}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=_as_lnstensor(lr),
            beta1=_as_lnstensor(betas[0]),
            beta2=_as_lnstensor(betas[1]),
            eps=_as_lnstensor(eps),
            weight_decay=_as_lnstensor(weight_decay),
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
    
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, beta1, beta2, eps, weight_decay = align_lnstensor_bases(
                lr, beta1, beta2, eps, weight_decay, base=base)

            one = LNSTensor.get_internal_tensor(1.0, base)
            one_minus_beta1 = lns_sub(one, beta1._lns, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = lns_sub(LNS_ZERO, grad, base) # −∇f

                # 2. weight decay: g ← g + λθ
                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    grad = lns_add(grad, lns_mul(p.data, weight_decay._lns), base)

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = 0
                    state["exp_avg"] = zeros_like(p.data, b=base)._lns # m_0
                    state["inf_norm"] = zeros_like(p.data, b=base)._lns # u_0

                # Retrieve running stats
                exp_avg = state["exp_avg"] # m_{t-1}
                inf_norm = state["inf_norm"] # u_{t-1}
                state["step"] += 1
                t = state["step"]

                # 3. m_t ← β_1*m_{t-1} + (1 − β_1)*g
                exp_avg = lns_add(
                    lns_mul(exp_avg, beta1._lns),
                    lns_mul(grad, one_minus_beta1),
                    base
                )

                # 4. u_t ← max(β_2*u_{t-1}, |g_t| + ε)
                inf_norm = lns_maximum(
                    lns_mul(inf_norm, beta2._lns),
                    lns_add(lns_abs(grad), eps._lns, base),
                    base
                )

                # 5. θ ← θ − γ*m / (sqrt(1 - b_1^t) * u)
                t_tensor = torch.tensor(t, dtype=torch.int64)
                one_minus_beta1_t = lns_sub(one, lns_pow(beta1._lns, t_tensor, base), base)
                denom = lns_mul(one_minus_beta1_t, inf_norm)
                step_size = lns_mul(lr._lns, lns_div(exp_avg, denom, base))
                p.data = lns_sub(p.data, step_size, base)

                state["exp_avg"] = exp_avg
                state["inf_norm"] = inf_norm

        return loss