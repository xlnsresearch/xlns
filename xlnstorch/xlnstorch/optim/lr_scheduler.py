from typing import Callable, List
from typing_extensions import override
from bisect import bisect_right
import torch
from xlnstorch import LNSTensor, lnstensor
from . import LNSOptimizer
from xlnstorch.operators import (
    lns_mul,
    lns_pow,
    lns_div,
)

def _lns(value: float | LNSTensor, base) -> LNSTensor:
    if isinstance(value, LNSTensor):
        return lnstensor(value, b=base)._lns
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
            lr_lambda: Callable[[int], float | LNSTensor] | List[Callable[[int], float | LNSTensor]],
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        super().__init__(optimizer, lr_lambda, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        return [
            lns_mul(base_lr, _lns(lmbda(self.last_epoch), base), base)
            for lmbda, base_lr, base in zip(self.lr_lambdas, self.base_lrs, self.lns_lr_bases)
        ]

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
            lr_lambda: Callable[[int], float | LNSTensor] | List[Callable[[int], float | LNSTensor]],
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        super().__init__(optimizer, lr_lambda, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch > 0:
            return [
                lns_mul(group["lr"], _lns(lmbda(self.last_epoch), base), base)
                for lmbda, group, base in zip(self.lr_lambdas, self.optimizer.param_groups, self.lns_lr_bases)
            ]

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
            gamma: float | LNSTensor = 0.1,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.gammas = [_lns(gamma, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, step_size, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(group["lr"], gamma, base)
            for group, gamma, base in zip(self.optimizer.param_groups, self.gammas, self.lns_lr_bases)
        ]

    def _get_closed_form_lr(self) -> List[torch.Tensor]:
        return [
            lns_mul(base_lr, lns_pow(gamma, torch.tensor(self.last_epoch // self.step_size), base), base)
            for base_lr, gamma, base in zip(self.base_lrs, self.gammas, self.lns_lr_bases)
        ]

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
            gamma: float | LNSTensor = 0.1,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.gammas = [_lns(gamma, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, milestones, gamma, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_mul(group["lr"], lns_pow(gamma, torch.tensor(self.milestones[self.last_epoch]), base), base)
            for group, gamma, base in zip(self.optimizer.param_groups, self.gammas, self.lns_lr_bases)
        ]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        return [
            lns_mul(base_lr, lns_pow(gamma, torch.tensor(bisect_right(milestones, self.last_epoch)), base), base)
            for base_lr, gamma, base in zip(self.base_lrs, self.gammas, self.lns_lr_bases)
        ]

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
            factor: float | LNSTensor = 1.0 / 3,
            total_iters: int = 0,
            last_epoch: int = -1,
        ):
        self.lns_lr_bases = get_lr_bases(optimizer)
        self.factors = [_lns(factor, base) for base in self.lns_lr_bases]
        super().__init__(optimizer, factor, total_iters, last_epoch)

    @override
    def get_lr(self) -> List[torch.Tensor]:
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                lns_mul(base_lr, factor, base)
                for base_lr, factor, base in zip(self.base_lrs, self.factors, self.lns_lr_bases)
            ]

        if self.last_epoch != self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            lns_div(group["lr"], factor, base)
            for group, factor, base in zip(self.optimizer.param_groups, self.factors, self.lns_lr_bases)
        ]