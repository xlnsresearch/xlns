import torch
from .. import LNSTensor, lnstensor, LNS_ZERO, align_lnstensor_bases, zeros_like
from ..operators import (
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

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSRAdam(LNSOptimizer):
    """
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
            lr=_as_lnstensor(lr),
            beta1=_as_lnstensor(betas[0]),
            beta2=_as_lnstensor(betas[1]),
            eps=_as_lnstensor(eps),
            weight_decay=_as_lnstensor(weight_decay),
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
        )
        super(LNSRAdam, self).__init__(params, defaults)

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

            lr, beta1, beta2, eps, weight_decay = align_lnstensor_bases(
                lr, beta1, beta2, eps, weight_decay, base=base
            )

            one = LNSTensor.get_internal_tensor(1.0, base)
            two = LNSTensor.get_internal_tensor(2.0, base)
            four = LNSTensor.get_internal_tensor(4.0, base)
            five = LNSTensor.get_internal_tensor(5.0, base)

            one_minus_beta1 = lns_sub(one, beta1._lns, base)
            one_minus_beta2 = lns_sub(one, beta2._lns, base)
            rho_inf = lns_sub(lns_div(two, lns_sub(one, beta2._lns, base), base), one, base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 1. flip sign if we want to maximise
                if maximize:
                    grad = lns_neg(grad)

                # 2. weight decay:
                if not lns_equal(weight_decay._lns, LNS_ZERO):
                    if decoupled_weight_decay:
                        # θ ← θ − γ λ θ
                        wd_step = lns_mul(lr._lns, weight_decay._lns)
                        p.data  = lns_sub(p.data, lns_mul(p.data, wd_step), base)
                    else:
                        # g ← g + λ θ
                        grad = lns_add(grad, lns_mul(weight_decay._lns, p.data), base)

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
                    lns_mul(exp_avg, beta1._lns),
                    lns_mul(grad, one_minus_beta1),
                    base
                )
                grad_sq = lns_mul(grad, grad)
                exp_avg_sq = lns_add(
                    lns_mul(exp_avg_sq, beta2._lns),
                    lns_mul(grad_sq, one_minus_beta2),
                    base
                )

                # 5. bias-corrected first moment: m'_t ← m_t / (1 - β_1^t)
                beta1_pow = lns_pow(beta1._lns, t, base)
                one_minus_beta1_pow = lns_sub(one, beta1_pow, base)
                exp_avg_hat = lns_div(exp_avg, one_minus_beta1_pow, base)

                # 6. ρ_t ← ρ_∞ - 2t*β_2^t / (1 - β_2^t)
                t_lns = LNSTensor.get_internal_tensor(t, base)
                beta2_pow = lns_pow(beta2._lns, t, base)
                one_minus_beta2_pow = lns_sub(one, beta2_pow, base)
                corr_term = lns_div(
                    lns_mul(two, lns_mul(t_lns, beta2_pow)),
                    one_minus_beta2_pow,
                    base
                )
                rho_t = lns_sub(rho_inf, corr_term, base)

                # 7. update rule based on ρ_t:
                if lns_gt(rho_t, five):
                    # l_t ← sqrt(1-β_2^t) / (sqrt(v_t) + ε)
                    l_t = lns_div(
                        lns_sqrt(one_minus_beta2_pow, base),
                        lns_add(lns_sqrt(exp_avg_sq, base), eps._lns, base),
                        base
                    )
                    # r_t ← sqrt((ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t))
                    r_t_num = lns_mul(rho_inf, lns_mul(lns_sub(rho_t, four, base), lns_sub(rho_t, two, base)))
                    r_t_den = lns_mul(rho_t, lns_mul(lns_sub(rho_inf, four, base), lns_sub(rho_inf, two, base)))
                    r_t = lns_sqrt(lns_div(r_t_num, r_t_den, base), base)
                    # step ← γ * m'_t * l_t * r_t
                    step = lns_mul(lr._lns, lns_mul(exp_avg_hat, lns_mul(l_t, r_t)))
                else:
                    # step ← γ * m'_t
                    step = lns_mul(lr._lns, exp_avg_hat)

                # 8. Update parameters: θ ← θ − step
                p.data = lns_sub(p.data, step, base)

                state["step"] = t
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss