import torch
from xlnstorch import LNS_ZERO, LNS_ONE, zeros_like
from xlnstorch.operators import (
    lns_sub,
    lns_neg,
    lns_equal,
    lns_mul,
    lns_add,
    lns_div,
    lns_sqrt,
    lns_pow,
    lns_maximum,
)
from . import LNSOptimizer

class LNSAdamW(LNSOptimizer):
    """
    Implements the AdamW optimization algorithm for LNSTensor parameters,
    including optional weight–decay regularisation, the AMSGrad variant,
    and a “maximize” mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.AdamW`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the AdamW algorithm.

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.001). Must be a non-negative LNSTensor or float.
    betas : Tuple[float, float] or Tuple[LNSTensor, LNSTensor], optional
        Coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999)). Must be two non-negative LNSTensor or float values
        in the range [0.0, 1.0).
    eps : LNSTensor, float, optional
        Term added to the denominator for numerical stability (default: 1e-8).
    weight_decay : LNSTensor or float
        Weight decay (L2 penalty) (default: 0.01). Must be a non-negative LNSTensor or float.
    amsgrad : bool, optional
        Uses the AMSGrad variant that maintains the maximum of past squared gradients
        (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
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
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self.make_lnstensor_params("lr", "beta1", "beta2", "eps", "weight_decay")

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
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]
            base = group["base"]

            one_minus_beta1 = lns_sub(LNS_ONE, beta1, base)
            one_minus_beta2 = lns_sub(LNS_ONE, beta2, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = lns_neg(grad) # −∇f

                # 2. weight decay: θ ← θ - γλθ
                if not lns_equal(weight_decay, LNS_ZERO):
                    p.data = lns_sub(p.data, lns_mul(lns_mul(lr, weight_decay, base), p.data, base), base)

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = 0
                    state["exp_avg"] = zeros_like(p.data, b=base)._lns # m_0
                    state["exp_avg_sq"] = zeros_like(p.data, b=base)._lns # v_0
                    if amsgrad:
                        state["max_exp_avg_sq"] = zeros_like(p.data, b=base)._lns # v_max_0

                # Retrieve running stats
                exp_avg = state["exp_avg"] # m_{t-1}
                exp_avg_sq = state["exp_avg_sq"] # v_{t-1}
                state["step"] += 1
                t = state["step"]

                # 3. m_t ← β_1*m_{t-1} + (1 − β_1)*g
                exp_avg = lns_add(
                    lns_mul(exp_avg, beta1, base),
                    lns_mul(grad, one_minus_beta1, base),
                    base
                )

                # 4. v_t ← β_2*v_{t-1} + (1 − β_2)*g^2
                grad_squared = lns_mul(grad, grad, base)
                exp_avg_sq = lns_add(
                    lns_mul(exp_avg_sq, beta2, base),
                    lns_mul(grad_squared, one_minus_beta2, base),
                    base
                )

                # 5. bias correction:
                # m'_t = m_t / (1 − β_1^t)
                # v'_t = v_t / (1 − β_2^t)
                t_tensor = torch.tensor(t, dtype=torch.int64)
                beta1_t = lns_pow(beta1, t_tensor, base)
                beta2_t = lns_pow(beta2, t_tensor, base)
                bias_corr1 = lns_sub(LNS_ONE, beta1_t, base)
                bias_corr2 = lns_sub(LNS_ONE, beta2_t, base)

                exp_avg_hat = lns_div(exp_avg, bias_corr1, base)
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    max_exp_avg_sq = lns_maximum(max_exp_avg_sq, exp_avg_sq, base)
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    denom_sq = lns_div(max_exp_avg_sq, bias_corr2, base)
                else:
                    denom_sq = lns_div(exp_avg_sq, bias_corr2, base)

                # 6. θ ← θ − γ*m' / (sqrt(v') + ε)
                denom = lns_add(lns_sqrt(denom_sq, base), eps, base)
                step_size = lns_mul(lr, lns_div(exp_avg_hat, denom, base), base)
                p.data = lns_sub(p.data, step_size, base)

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss