import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ONE
from xlnstorch.operators import (
    lns_mul,
    lns_sign,
    lns_ne,
    lns_neg,
    lns_abs,
    lns_pow,
    lns_add,
    lns_reciprocal,
)
from . import LNSOptimizer

class LNSMul(LNSOptimizer):
    r"""
    Implements a simple multiplication algorithm for LNSTensor
    parameters.

    .. math::
        \begin{aligned}
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{input} : \gamma \text{ (lr)},\;
                              \theta_{0} \text{ (params)},\;
                              f(\theta) \text{ (objective)},\;
                              \textit{use_pow},\;                               \\
            &\hspace{17mm}    \textit{maximize}                                 \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{for } t = 1 \textbf{ to } \ldots \textbf{ do}              \\
            &\hspace{5mm} g_t \leftarrow
                          \nabla_{\theta} f_t \left(\theta_{t-1}\right)         \\
            &\hspace{5mm} \textbf{if } \textit{use_pow}:                        \\
            &\hspace{10mm} u_t \leftarrow
                           2 ^ {-\gamma g_t
                                \operatorname{sign} \bigl(\theta_{t-1}\bigr)}   \\
            &\hspace{10mm} \textbf{if } \textit{maximize}:                      \\
            &\hspace{15mm} u_t \leftarrow 1 / u_t                               \\
            &\hspace{5mm} \textbf{else}:                                        \\
            &\hspace{10mm} u_t \leftarrow
                           1 + \gamma \lvert g_t \rvert                         \\
            &\hspace{10mm} \textbf{if }
                           \left(\operatorname{sign} \bigl(\theta_{t-1}\bigr)
                           \neq \operatorname{sign} \left(g_t\right)\right)
                           \oplus \textit{maximize}:                            \\
            &\hspace{15mm} u_t \leftarrow 1 / u_t                               \\
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
            maximize=False,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            use_pow=use_pow,
            maximize=maximize
        )
        super(LNSMul, self).__init__(params, defaults)
        self.make_lnstensor_params("lr")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            use_pow = group["use_pow"]
            maximize = group["maximize"]
            base = group["base"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad

                if use_pow:
                    mul_term = lns_mul(lns_neg(lr), lns_mul(grad, lns_sign(p, base), base), base)
                    if maximize:
                        mul_term = lns_neg(mul_term)
                    mul_term = lns_pow(
                        LNSTensor.get_internal_tensor(2.0, base),
                        lnstensor(mul_term, from_lns=True, b=base).value,
                        base
                    )

                else:
                    mul_term = lns_add(LNS_ONE, lns_mul(lr, lns_abs(grad), base), base)
                    diff_sign = lns_ne(lns_sign(grad, base), lns_sign(p, base))
                    mul_term = torch.where(diff_sign ^ maximize, mul_term,
                                           lns_reciprocal(mul_term, base))

            p.data = lns_mul(p.data, mul_term, base)

        return loss