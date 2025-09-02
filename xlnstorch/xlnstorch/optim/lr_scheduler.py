from typing import Callable, List, Union
from typing_extensions import override
from bisect import bisect_right
import torch
from xlnstorch import LNSTensor, lnstensor, LNS_ONE
from . import LNSOptimizer

def _lns(value: Union[float, LNSTensor], base) -> LNSTensor:
    if isinstance(value, LNSTensor):
        return lnstensor(value, b=base)._lns.view(torch.int64)
    return LNSTensor.get_internal_tensor(value, base)

def get_lr_bases(optimizer: LNSOptimizer) -> List[torch.Tensor]:
    """
    Returns a list of base values for the learning rates of each
    parameter group in the optimizer.

    Parameters
    ----------
    optimizer : LNSOptimizer
        The optimizer from which to extract the base learning rates.

    Returns
    -------
    List[torch.Tensor]
        A list of base learning rates for each parameter group.

    Raises
    ------
    TypeError
        If the provided optimizer is not an instance of `LNSOptimizer`.
    """
    if not isinstance(optimizer, LNSOptimizer):
        raise TypeError(f"{type(optimizer).__name__} is not an LNSOptimizer")

    return [group["base"] for group in optimizer.param_groups]

class LNSLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to the initial learning rate multipled by a given function of the epoch.

    See also: :class:`torch.optim.lr_scheduler.LambdaLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    lr_lambda : Callable[[int], float | LNSTensor] or List[Callable[[int], float | LNSTensor]]
        A function or a list of functions which computes a multiplicative factor given an integer parameter
        `epoch`, which is the index of the current epoch.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            lr_lambda: Union[Callable[[int], Union[float, LNSTensor]], List[Callable[[int], Union[float, LNSTensor]]]],
            last_epoch: int = -1,
        ):
        super().__init__(optimizer, lr_lambda, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        new_lrs = []
        for ops, lmbda, base_lr in zip(self.optimizer.lns_ops(), self.lr_lambdas, self.base_lrs):
            lns_lmbda = _lns(lmbda(self.last_epoch), ops.base)
            new_lrs.append(ops.mul(base_lr, lns_lmbda))

        return new_lrs

class LNSMultiplicativeLR(torch.optim.lr_scheduler.MultiplicativeLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to the previous learning rate multipled by a given multiplicative factor.

    See also: :class:`torch.optim.lr_scheduler.MultiplicativeLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    lr_lambda : Callable[[int], float | LNSTensor] or List[Callable[[int], float | LNSTensor]]
        A function or a list of functions which computes a multiplicative factor given an integer parameter
        `epoch`, which is the index of the current epoch.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            lr_lambda: Union[Callable[[int], Union[float, LNSTensor]], List[Callable[[int], Union[float, LNSTensor]]]],
            last_epoch: int = -1,
        ):
        super().__init__(optimizer, lr_lambda, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch > 0:
            new_lrs = []

            for (group, ops), lmbda in zip(self.optimizer.lns_param_groups(), self.lr_lambdas):
                lns_lmbda = _lns(lmbda(self.last_epoch), ops.base)
                new_lrs.append(ops.mul(group["lr"], lns_lmbda))

            return new_lrs

        return [group["lr"] for group in self.optimizer.param_groups]

class LNSStepLR(torch.optim.lr_scheduler.StepLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter
    group by a factor of `gamma` every `step_size` epochs.

    See also: :class:`torch.optim.lr_scheduler.StepLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    step_size : int
        Period of learning rate decay.
    gamma : float | LNSTensor
        Multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            step_size: int,
            gamma: Union[float, LNSTensor] = 0.1,
            last_epoch: int = -1,
        ):
        super().__init__(optimizer, step_size, gamma, last_epoch)
        self.gammas = [_lns(gamma, ops.base) for ops in self.optimizer.lns_ops()]

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []
        for (group, ops), gamma in zip(self.optimizer.lns_param_groups(), self.gammas):
            new_lrs.append(ops.mul(group["lr"], gamma))

        return new_lrs

    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        cf_lrs = []
        for ops, base_lr, gamma in zip(self.optimizer.lns_ops(), self.base_lrs, self.gammas):
            pow = ops.pow(gamma, torch.tensor(self.last_epoch // self.step_size, dtype=torch.int64))
            cf_lrs.append(ops.mul(base_lr, pow))

        return cf_lrs

class LNSMultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter
    group by a factor of `gamma` at specified epochs.

    See also: :class:`torch.optim.lr_scheduler.MultiStepLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    milestones : List[int]
        List of epoch indices where the learning rate should be decayed.
    gamma : float | LNSTensor
        Multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            milestones: List[int],
            gamma: Union[float, LNSTensor] = 0.1,
            last_epoch: int = -1,
        ):
        self.gammas = [_lns(gamma, ops.base) for ops in optimizer.lns_ops()]
        super().__init__(optimizer, milestones, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []
        for (group, ops), gamma in zip(self.optimizer.lns_param_groups(), self.gammas):
            pow = ops.pow(gamma, torch.tensor(self.milestones[self.last_epoch], dtype=torch.int64))
            new_lrs.append(ops.mul(group["lr"], pow))

        return new_lrs

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        cf_lrs = []

        for ops, base_lr, gamma in zip(self.optimizer.lns_ops(), self.base_lrs, self.gammas):
            pow = ops.pow(gamma, torch.tensor(bisect_right(milestones, self.last_epoch), dtype=torch.int64))
            cf_lrs.append(ops.mul(base_lr, pow))

        return cf_lrs

class LNSConstantLR(torch.optim.lr_scheduler.ConstantLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to a constant value until a pre-determined number of epochs is reached.

    See also: :class:`torch.optim.lr_scheduler.ConstantLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    factor : float | LNSTensor
        Multiplicative factor of the learning rate.
    total_iters : int, optional
        The number of iterations for which the learning rate will be constant.
        Default: 0.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            factor: Union[float, LNSTensor] = 1.0 / 3,
            total_iters: int = 0,
            last_epoch: int = -1,
        ):
        self.factors = [_lns(factor, ops.base) for ops in optimizer.lns_ops()]
        super().__init__(optimizer, factor, total_iters, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            new_lrs = []

            for ops, base_lr, factor in zip(self.optimizer.lns_ops(), self.base_lrs, self.factors):
                new_lrs.append(ops.mul(base_lr, factor))

            return new_lrs

        if self.last_epoch != self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []
        for (group, ops), factor in zip(self.optimizer.lns_param_groups(), self.factors):
            new_lrs.append(ops.div(group["lr"], factor))

        return new_lrs

class LNSLinearLR(torch.optim.lr_scheduler.LinearLR):
    """
    An LNS learning rate scheduler that sets the learning rate of each parameter group
    to a linearly decaying value.

    See also: :class:`torch.optim.lr_scheduler.LinearLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    start_factor : float | LNSTensor
        The initial factor for the learning rate.
    end_factor : float | LNSTensor
        The final factor for the learning rate.
    total_iters : int, optional
        The number of iterations over which the learning rate will decay.
        Default: 5.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            start_factor: Union[float, LNSTensor] = 1.0 / 3,
            end_factor: Union[float, LNSTensor] = 1.0,
            total_iters: int = 5,
            last_epoch: int = -1,
        ):
        self.start_factor_lns = [_lns(start_factor, ops.base) for ops in optimizer.lns_ops()]
        self.end_factor_lns = [_lns(end_factor, ops.base) for ops in optimizer.lns_ops()]
        self.total_iters_lns = [_lns(total_iters, ops.base) for ops in optimizer.lns_ops()]
        super().__init__(optimizer, start_factor, end_factor, total_iters, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            new_lrs = []

            for (group, ops), start_factor in zip(self.optimizer.lns_param_groups(), self.start_factor_lns):
                new_lrs.append(ops.mul(group["lr"], start_factor))

            return new_lrs

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []

        for (group, ops), start_factor, end_factor, total_iters in zip(
            self.optimizer.lns_param_groups(), self.start_factor_lns,
            self.end_factor_lns, self.total_iters_lns
        ):

            delta_factor = ops.sub(end_factor, start_factor)
            epoch_minus_one = ops.sub(ops.to_lns(self.last_epoch), LNS_ONE)
            denom = ops.add(
                ops.mul(total_iters, start_factor),
                ops.mul(epoch_minus_one, delta_factor)
            )

            division_result = ops.div(delta_factor, denom)
            add_one_result = ops.add(LNS_ONE, division_result)

            new_lrs.append(ops.mul(group["lr"], add_one_result))

        return new_lrs

    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        cf_lrs = []

        for ops, base_lr, start_factor, end_factor, total_iters in zip(
            self.optimizer.lns_ops(), self.base_lrs, self.start_factor_lns,
            self.end_factor_lns, self.total_iters_lns
        ):

            factor_diff = ops.sub(end_factor, start_factor)
            clamped_epoch = ops.minimum(total_iters, ops.to_lns(self.last_epoch))
            numerator = ops.mul(factor_diff, clamped_epoch)

            interpolation = ops.div(numerator, total_iters)
            lr_factor = ops.add(start_factor, interpolation)

            cf_lrs.append(ops.mul(base_lr, lr_factor))

        return cf_lrs

class LNSExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter group
    by gamma each epoch.

    See also: :class:`torch.optim.lr_scheduler.ExponentialLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    gamma : float | LNSTensor
        Multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            gamma: Union[float, LNSTensor],
            last_epoch: int = -1
        ):
        self.gammas = [_lns(gamma, ops.base) for ops in optimizer.lns_ops()]
        super().__init__(optimizer, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []
        for (group, ops), gamma in zip(self.optimizer.lns_param_groups(), self.gammas):
            new_lrs.append(ops.mul(group["lr"], gamma))

        return new_lrs

    @override
    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        cf_lrs = []

        for ops, base_lr, gamma in zip(self.optimizer.lns_ops(), self.base_lrs, self.gammas):
            pow = ops.pow(gamma, torch.tensor(self.last_epoch, dtype=torch.int64))
            cf_lrs.append(ops.mul(base_lr, pow))

        return cf_lrs

class LNSPolynomialLR(torch.optim.lr_scheduler.PolynomialLR):
    """
    An LNS learning rate scheduler that decays the learning rate of each parameter group
    by a polynomial factor in the given total_iters.

    See also: :class:`torch.optim.lr_scheduler.PolynomialLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    total_iters : int
        The number of iterations over which the learning rate will decay.
    power : float | LNSTensor
        The power of the polynomial decay.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            total_iters: int = 5,
            power: Union[float, LNSTensor] = 0.9,
            last_epoch: int = -1,
        ):
        self.total_iters_lns = [_lns(total_iters, ops.base) for ops in optimizer.lns_ops()]
        power = power.value.item() if isinstance(power, LNSTensor) else power
        super().__init__(optimizer, total_iters, power, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        new_lrs = []

        for (group, ops), total_iters in zip(self.optimizer.lns_param_groups(), self.total_iters_lns):

            last_epoch_lns = ops.to_lns(self.last_epoch)
            last_epoch_sub_one = ops.sub(last_epoch_lns, LNS_ONE)
            last_epoch_div_total_iters = ops.div(last_epoch_lns, total_iters)
            last_epoch_sub_one_div_total_iters = ops.div(last_epoch_sub_one, total_iters)

            numerator = ops.sub(LNS_ONE, last_epoch_div_total_iters)
            denominator = ops.sub(LNS_ONE, last_epoch_sub_one_div_total_iters)
            fraction = ops.div(numerator, denominator)

            power_result = ops.pow(fraction, torch.tensor(self.power))
            new_lrs.append(ops.mul(group["lr"], power_result))

        return new_lrs

    @override
    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        cf_lrs = []

        for ops, base_lr, total_iters in zip(self.optimizer.lns_ops(), self.base_lrs, self.total_iters_lns):

            clamped_epoch = ops.minimum(total_iters, ops.to_lns(self.last_epoch))
            epoch_fraction = ops.div(clamped_epoch, total_iters)
            one_minus_fraction = ops.sub(LNS_ONE, epoch_fraction)

            pow_result = ops.pow(one_minus_fraction, torch.tensor(self.power))
            cf_lrs.append(ops.mul(base_lr, pow_result))

        return cf_lrs

class LNSReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    An LNS learning rate scheduler that reduces the learning rate of each parameter group
    when a metric has stopped improving.

    See also: :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    mode : str, optional
        One of `min`, `max`. In `min` mode, the learning rate will be reduced when the quantity
        monitored has stopped decreasing; in `max` mode, it will be reduced when the quantity
        monitored has stopped increasing. Default: `min`.
    factor : float | LNSTensor, optional
        Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
    patience : int, optional
        Number of epochs with no improvement after which learning rate will be reduced.
        Default: 10.
    threshold : float | LNSTensor, optional
        Threshold for measuring the new optimum, to only focus on significant changes.
        Default: 1e-4.
    threshold_mode : str, optional
        One of `rel`, `abs`. In `rel` mode, the threshold is a relative change;
        in `abs` mode, it is an absolute change. Default: `rel`.
    cooldown : int, optional
        Number of epochs to wait before resuming normal operation after lr has been reduced.
        Default: 0.
    min_lr : float | LNSTensor | List[float] | List[LNSTensor], optional
        A scalar or a list of scalars defining the lower bound on the learning rate
        of each parameter group. Default: 0.0.
    eps : float | LNSTensor, optional
        Minimal decay applied to lr. If the difference between new and old lr is smaller than eps,
        the update is ignored. Default: 1e-8.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            mode: str = "min",
            factor: Union[float, LNSTensor] = 0.1,
            patience: int = 10,
            threshold: Union[float, LNSTensor] = 1e-4,
            threshold_mode: str = "rel",
            cooldown: int = 0,
            min_lr: Union[float, LNSTensor, List[float], List[LNSTensor]] = 0.0,
            eps: Union[float, LNSTensor] = 1e-8,
        ):
        self.factor_lns = [_lns(factor, ops.base) for ops in optimizer.lns_ops()]
        self.eps_lns = [_lns(eps, ops.base) for ops in optimizer.lns_ops()]
        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps)

        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = [_lns(lr, ops.base) for lr, ops in zip(min_lr, optimizer.lns_ops())]
        else:
            self.min_lrs = [_lns(min_lr, ops.base) for ops in optimizer.lns_ops()]

    def _reduce_lr(self, epoch):
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError("The number of param groups in the optimizer must match the number of min_lrs.")
            else:
                self.min_lrs = [_lns(self.default_min_lr, ops.base) for ops in self.optimizer.lns_ops()]

        for (group, ops), factor, eps, min_lr in zip(
            self.optimizer.lns_param_groups(), self.factor_lns, self.eps_lns, self.min_lrs
        ):

            old_lr = group["lr"]
            new_lr = ops.maximum(ops.mul(old_lr, factor), min_lr)

            if ops.gt(ops.sub(old_lr, new_lr), eps):
                group["lr"] = new_lr

class LNSChainedScheduler(torch.optim.lr_scheduler.ChainedScheduler):
    """
    A scheduler that chains multiple schedulers together.

    Note that this scheduler is a subclass of torch's `ChainedScheduler`,
    and is implemented for completeness. You can use the torch version
    directly with LNS optimizers.

    See also: :class:`torch.optim.lr_scheduler.ChainedScheduler`

    Parameters
    ----------
    schedulers : List[torch.optim.lr_scheduler.LRScheduler]
        List of schedulers to chain.
    """

    def __init__(
            self,
            schedulers: List[torch.optim.lr_scheduler.LRScheduler],
            optimizer: LNSOptimizer,
        ):
        super().__init__(schedulers, optimizer)

class LNSSequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """
    A scheduler that applies a sequence of schedulers in order.

    Note that this scheduler is a subclass of torch's `SequentialLR`,
    and is implemented for completeness. You can use the torch version
    directly with LNS optimizers.

    See also: :class:`torch.optim.lr_scheduler.SequentialLR`

    Parameters
    ----------
    optimizer : LNSOptimizer
        Wrapped optimizer.
    schedulers : List[torch.optim.lr_scheduler.LRScheduler]
        List of schedulers to apply sequentially.
    milestones : List[int]
        List of epochs at which to switch to the next scheduler.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            optimizer: LNSOptimizer,
            schedulers: List[torch.optim.lr_scheduler.LRScheduler],
            milestones: List[int],
            last_epoch: int = -1,
        ):
        super().__init__(optimizer, schedulers, milestones, last_epoch)