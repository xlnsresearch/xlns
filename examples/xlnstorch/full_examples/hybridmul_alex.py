import torch
from xlnstorch import LNS_ONE
from xlnstorch.optim import LNSOptimizer

class LNSHybridMulAlex(LNSOptimizer):
    r"""
    Implements a hybrid multiplication algorithm for LNSTensor
    parameters. This optimizer uses a heuristic to decide between
    using a standard multiplicative update, a sign-based update
    or a gradient descent-like update.

    .. math::
        \begin{aligned}
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{input} : \gamma \text{ (lr)},\;
                              \theta_{0} \text{ (params)},\;
                              f(\theta) \text{ (objective)}                     \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                \\
            &\textbf{for } t = 1 \textbf{ to } \ldots \textbf{ do}              \\
            &\hspace{5mm} g_t \leftarrow
                          \nabla_{\theta} f_t \left(\theta_{t-1}\right)         \\
            &\hspace{5mm} \textbf{if }
                           \operatorname{sign} \bigl(\theta_{t-1}\bigr)
                           = \operatorname{sign} \left(g_t\right)
                           \lor \lvert g_t \rvert < \gamma
                           \lor \lvert \theta_{t-1} \rvert < \gamma:            \\
            &\hspace{10mm} u_t \leftarrow
                           1 / \left(1 + \gamma \lvert g_t \rvert \right)       \\
            &\hspace{5mm} \textbf{else}:                                        \\
            &\hspace{10mm} u_t \leftarrow
                           \max \left( 2 ^ {\gamma},
                           1 +  \gamma \cdot \lvert g_t /
                           \theta_{t-1} \rvert \right)                          \\
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
    """

    def __init__(
            self,
            params,
            lr=0.01,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
        )
        super(LNSHybridMulAlex, self).__init__(params, defaults)
        self.make_lnstensor_params("lr")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)

                same_sign = ops.eq(ops.sign(grad), ops.sign(data))
                small_values = ops.lt(ops.abs(grad), lr) | ops.lt(ops.abs(data), lr)
                mul_mask = same_sign | small_values

                lr_mul_grad = ops.mul(lr, ops.abs(grad))
                mul_update = ops.add(LNS_ONE, lr_mul_grad)
                gd_update = ops.add(LNS_ONE, ops.div(lr_mul_grad, ops.abs(data)))
                mul_term = torch.where(mul_mask, ops.reciprocal(mul_update), gd_update)

                p.data = ops.mul(data, mul_term).view(torch.float64)

        return loss
