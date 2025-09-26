import torch
from xlnstorch import LNS_ZERO, LNS_ONE
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

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("lambd", lambda lambd: lambd >= 0.0)
        self.validate_param("alpha", lambda alpha: 0.0 < alpha <= 1.0)
        self.validate_param("t0", lambda t0: t0 >= 0.0)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= 0.0)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group, ops in self.lns_param_groups():
            lr = group["lr"]
            lambd = group["lambd"]
            alpha = group["alpha"]
            t0 = group["t0"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.view(torch.int64) # g_t
                data = p.data.view(torch.int64)
                state = self.state[p]

                if maximize:
                    grad = ops.neg(grad)

                if not ops.equal(weight_decay, LNS_ZERO):
                    grad = ops.add(grad, ops.mul(data, weight_decay))

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = LNS_ZERO.clone()
                    state["averaging_coef"] = LNS_ONE.clone()
                    state["averaged_param"] = data.clone()

                # Retrieve running stats
                step = ops.add(state["step"], LNS_ONE)
                averaging_coef = state["averaging_coef"]
                averaged_param = state["averaged_param"]

                # 1. learning-rate schedule
                denom = ops.add(LNS_ONE, ops.mul(lambd, ops.mul(lr, step)))
                denom = ops.pow(denom, ops.from_lns(alpha))
                current_lr = ops.div(lr, denom)

                # 2. update averaged parameter
                decay = ops.sub(LNS_ONE, ops.mul(lambd, current_lr))
                data = ops.mul(data, decay)
                data = ops.sub(data, ops.mul(grad, current_lr))
                p.data = data.view(torch.float64)

                # 3. update averaged parameter
                if ops.gt(step, t0):
                    denom = ops.maximum(LNS_ONE, ops.sub(step, t0))
                    averaging_coef = ops.div(LNS_ONE, denom)

                diff = ops.sub(data, averaged_param)
                averaged_param = ops.add(averaged_param, ops.mul(diff, averaging_coef))

                state["step"] = step
                state["averaging_coef"] = averaging_coef
                state["averaged_param"] = averaged_param