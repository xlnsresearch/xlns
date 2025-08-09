import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ZERO, LNS_ONE, align_lnstensor_bases
from xlnstorch.operators import (
    lns_mul,
    lns_sign,
    lns_eq,
)
from . import LNSOptimizer

def _as_lnstensor(x):
    if isinstance(x, LNSTensor):
        return x
    else:
        return lnstensor(x)

class LNSSignMul(LNSOptimizer):
    r"""
    Implements a simple sign multiplication algorithm for LNSTensor
    parameters.

    .. math::
        \begin{aligned}
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{input} : \gamma \text{ (lr)},\;
                              \theta_{0} \text{ (params)},\;
                              f(\theta) \text{ (objective)},\;
                              \textit{use_pow},\;                               \\
            &\hspace{17mm}    \textit{maximize}                                 \\
            &\textbf{initialize} :                                              \\
            &\hspace{5mm}\alpha \; = \;
                \begin{cases}
                    2^{\gamma}, & \text{if } \textit{use_pow}                   \\
                    1 + \gamma, & \text{otherwise}
                \end{cases}
                \quad\text{(primary multiplier)}                                \\
            &\hspace{5mm}  \alpha^{-1} \; = \; 1 / \alpha
                \quad\text{(inverse multiplier)}                                \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{for } t = 1 \textbf{ to } \ldots \textbf{ do}              \\
            &\hspace{5mm}\textbf{if } \textit{maximize}:                        \\
            &\hspace{10mm} g_t \leftarrow
                           -\nabla_{\theta} f_t \left(\theta_{t-1}\right)       \\
            &\hspace{5mm}\textbf{else}:                                         \\
            &\hspace{10mm} g_t \leftarrow
                           \nabla_{\theta} f_t \left(\theta_{t-1}\right)        \\
            &\hspace{5mm} S_t \leftarrow
                          \operatorname{sign} \bigl(\theta_{t-1}\bigr)
                          \cdot
                          \operatorname{sign} \left(g_t\right)                  \\
            &\hspace{5mm} u_t \leftarrow
                \begin{cases}
                    \alpha^{-1}, & \text{if } S_t                               \\
                    \alpha,      & \text{otherwise}
                \end{cases}                                                     \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} \cdot u_t            \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\[-1.ex]
            &\textbf{return } \theta_t                                          \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\
        \end{aligned}

    Parameters
    ----------
    params : iterable
        An iterable of parameters to optimize or dicts defining parameter groups.
        This should be obtained from a model's `lns_parameters()` method.
    lr : LNSTensor, float, optional
        Learning rate (default: 0.01). Must be a non-negative LNSTensor or float.
    use_pow : bool, optional
        If True, uses a power-based multiplier (default: False).
    maximize : bool, optional
        If True, optimizes the parameters for maximization instead of minimization (default: False).
    """

    def __init__(
            self,
            params,
            lr=0.01,
            use_pow=False,
            *,
            maximize=False
        ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=_as_lnstensor(lr),
            use_pow=use_pow,
            maximize=maximize
        )
        super(LNSSignMul, self).__init__(params, defaults)

        # precompute 1 + lr and 1 / (1 + lr)
        for group in self.param_groups:
            lr_ = group["lr"]
            use_pow_ = group["use_pow"]

            if use_pow_:
                group["mul_term"] = 2.0 ** lr_
            else:
                group["mul_term"] = 1.0 + lr_

            group["inv_mul_term"] = 1.0 / group["mul_term"]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mul_term = group["mul_term"]
            inv_mul_term = group["inv_mul_term"]
            maximize = group["maximize"]
            base = group["base"]

            # Align the parameters to the base of the group.
            lr, mul_term, inv_mul_term = align_lnstensor_bases(lr, mul_term, inv_mul_term, base=base)

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                same_sign = lns_eq(lns_sign(grad, base), lns_sign(p, base))
                mul_update = torch.where(same_sign ^ maximize, inv_mul_term._lns, mul_term._lns)

                p.data = lns_mul(p.data, mul_update)