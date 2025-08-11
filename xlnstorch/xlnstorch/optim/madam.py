import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ZERO, LNS_ONE, align_lnstensor_bases, zeros_like
from . import LNSOptimizer
from xlnstorch.operators import (
    lns_mul,
    lns_sum,
    lns_div,
    lns_sqrt,
    lns_sub,
    lns_pow,
    lns_add,
    lns_neg,
    lns_clamp,
    lns_sign,
    lns_exp,
)

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSMadam(LNSOptimizer):
    """
    """

    def __init__(
            self,
            params,
            lr=0.01,
            beta=0.999,
            eps=1e-8,
            p_scale=3.0,
            g_bound=10.0,
            use_pow=False,
            *,
            maximize=False
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if not (0.0 < beta <= 1.0):
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(
            lr=_as_lnstensor(lr),
            beta=_as_lnstensor(beta),
            eps=_as_lnstensor(eps),
            p_scale=_as_lnstensor(p_scale),
            g_bound=_as_lnstensor(g_bound),
            use_pow=use_pow,
            maximize=maximize
        )
        super(LNSMadam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            p_scale = group["p_scale"]
            g_bound = group["g_bound"]
            use_pow = group["use_pow"]
            maximize = group["maximize"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, beta, eps, p_scale, g_bound = align_lnstensor_bases(lr, beta, eps, p_scale, g_bound, base=base)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    # First time we see this parameter
                    rms = lns_sqrt(lns_div(lns_sum(lns_mul(p, p), base),
                                           LNSTensor.get_internal_tensor(p.numel(), base),
                                           base), base)
                    state['max'] = lns_mul(p_scale._lns, rms)
                    state['step'] = 0
                    state['exp_avg_sq'] = zeros_like(p.data, b=base)._lns

                # retrieve running stats
                max = state['max']
                step = state['step'] + 1
                exp_avg_sq = state['exp_avg_sq']

                bias_correction = lns_sub(LNS_ONE, lns_pow(beta._lns, torch.tensor(step), base), base)
                exp_avg_sq = lns_add(
                    lns_mul(beta._lns, exp_avg_sq),
                    lns_mul(lns_sub(LNS_ONE, beta._lns, base), lns_mul(grad, grad)),
                    base
                )
                corrected_exp_avg_sq = lns_add(lns_div(exp_avg_sq, bias_correction, base), eps._lns, base)

                g_normed = lns_div(grad, lns_sqrt(corrected_exp_avg_sq, base), base)
                g_normed = lns_clamp(g_normed, lns_neg(g_bound._lns), g_bound._lns)

                if use_pow:
                    if maximize:
                        exponent = lns_mul(lr._lns, lns_mul(g_normed, lns_sign(p, base)))
                    else:
                        exponent = lns_mul(lns_neg(lr._lns), lns_mul(g_normed, lns_sign(p, base)))
                    p.data = lns_mul(p.data, lns_exp(exponent, base))
                    p.data = lns_clamp(p.data, lns_neg(max), max)

                else:
                    if maximize:
                        mul_term = lns_add(LNS_ONE, lns_mul(lr._lns, lns_mul(g_normed, lns_sign(p, base))), base)
                    else:
                        mul_term = lns_sub(LNS_ONE, lns_mul(lr._lns, lns_mul(g_normed, lns_sign(p, base))), base)
                    p.data = lns_mul(p.data, mul_term)

                # update running stats
                state['step'] = step
                state['exp_avg_sq'] = exp_avg_sq

        return loss