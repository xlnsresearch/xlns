import torch
from xlnstorch import LNSTensor, LNS_ZERO, LNS_ONE, zeros_like
from xlnstorch.operators import (
    lns_equal,
    lns_sub,
    lns_mul,
    lns_add,
    lns_pow,
    lns_div,
    lns_sqrt,
    lns_neg,
    lns_gt,
)
from . import LNSOptimizer

class LNSRAdam(LNSOptimizer):
    """
    Implements the rectified Adam optimization algorithm for LNSTensor
    parameters, including decoupled weight decay, and a "maximize" mode.

    This optimizer is analogous to PyTorch's :py:class:`torch.optim.RAdam`,
    but is designed to work with LNSTensor objects. See the PyTorch
    documentation for more details on the RAdam algorithm.

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
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
    decoupled_weight_decay : bool, optional
        If True, applies decoupled weight decay (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            decoupled_weight_decay=False,
            *,
            maximize=False
        ):

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1 value: {betas[0]}")

        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
        )
        super(LNSRAdam, self).__init__(params, defaults)
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
            decoupled_weight_decay = group["decoupled_weight_decay"]
            maximize = group["maximize"]
            base = group["base"]

            two = LNSTensor.get_internal_tensor(2.0, base)
            four = LNSTensor.get_internal_tensor(4.0, base)
            five = LNSTensor.get_internal_tensor(5.0, base)

            one_minus_beta1 = lns_sub(LNS_ONE, beta1, base)
            one_minus_beta2 = lns_sub(LNS_ONE, beta2, base)
            rho_inf = lns_sub(lns_div(two, lns_sub(LNS_ONE, beta2, base), base), LNS_ONE, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = lns_neg(grad)

                # 2. weight decay:
                if not lns_equal(weight_decay, LNS_ZERO):
                    if decoupled_weight_decay:
                        # θ ← θ − γ λ θ
                        wd_step = lns_mul(lr, weight_decay, base)
                        p.data  = lns_sub(p.data, lns_mul(p.data, wd_step, base), base)
                    else:
                        # g ← g + λ θ
                        grad = lns_add(grad, lns_mul(weight_decay, p.data, base), base)

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = zeros_like(p, b=base)._lns # m_0
                    state["exp_avg_sq"] = zeros_like(p, b=base)._lns # v_0

                # Retrieve running stats
                t = state["step"] + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 3. first and second moments:
                # m_t ← β_1*m_{t-1} + (1-β_1)*g_t
                # v_t ← β_2*v_{t-1} + (1-β_2)*g_t^2
                exp_avg = lns_add(
                    lns_mul(exp_avg, beta1, base),
                    lns_mul(grad, one_minus_beta1, base),
                    base
                )
                grad_sq = lns_mul(grad, grad, base)
                exp_avg_sq = lns_add(
                    lns_mul(exp_avg_sq, beta2, base),
                    lns_mul(grad_sq, one_minus_beta2, base),
                    base
                )

                # 5. bias-corrected first moment: m'_t ← m_t / (1 - β_1^t)
                beta1_pow = lns_pow(beta1, t, base)
                one_minus_beta1_pow = lns_sub(LNS_ONE, beta1_pow, base)
                exp_avg_hat = lns_div(exp_avg, one_minus_beta1_pow, base)

                # 6. ρ_t ← ρ_∞ - 2t*β_2^t / (1 - β_2^t)
                t_lns = LNSTensor.get_internal_tensor(t, base)
                beta2_pow = lns_pow(beta2, t, base)
                one_minus_beta2_pow = lns_sub(LNS_ONE, beta2_pow, base)
                corr_term = lns_div(
                    lns_mul(two, lns_mul(t_lns, beta2_pow, base), base),
                    one_minus_beta2_pow,
                    base
                )
                rho_t = lns_sub(rho_inf, corr_term, base)

                # 7. update rule based on ρ_t:
                if lns_gt(rho_t, five):
                    # l_t ← sqrt(1-β_2^t) / (sqrt(v_t) + ε)
                    l_t = lns_div(
                        lns_sqrt(one_minus_beta2_pow, base),
                        lns_add(lns_sqrt(exp_avg_sq, base), eps, base),
                        base
                    )
                    # r_t ← sqrt((ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t))
                    r_t_num = lns_mul(rho_inf, lns_mul(lns_sub(rho_t, four, base), lns_sub(rho_t, two, base), base), base)
                    r_t_den = lns_mul(rho_t, lns_mul(lns_sub(rho_inf, four, base), lns_sub(rho_inf, two, base), base), base)
                    r_t = lns_sqrt(lns_div(r_t_num, r_t_den, base), base)
                    # step ← γ * m'_t * l_t * r_t
                    step = lns_mul(lr, lns_mul(exp_avg_hat, lns_mul(l_t, r_t, base), base), base)
                else:
                    # step ← γ * m'_t
                    step = lns_mul(lr, exp_avg_hat, base)

                # 8. Update parameters: θ ← θ − step
                p.data = lns_sub(p.data, step, base)

                state["step"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss