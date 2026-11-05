import math
from warnings import warn

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


class _TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        # Save _last_lr before it gets filtered out
        last_lr = state_dict.get("_last_lr", None)

        if "after_scheduler_dict" not in state_dict:
            warn(
                "after_scheduler_dict is missing; the nested scheduler state "
                "cannot be restored"
            )
        else:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)

        # Restore optimizer's lr from _last_lr to ensure consistency
        # This is critical because PyTorch's CosineAnnealingLR.get_lr() uses
        # group["lr"] to compute the next lr, but load_state_dict doesn't
        # update the optimizer's lr automatically.
        if last_lr is not None:
            for param_group, lr in zip(self.optimizer.param_groups, last_lr):
                param_group["lr"] = lr


class _WarmupScheduler(_TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmupLR(_WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class _FlatLR(_LRScheduler):
    """Keep every parameter group at its configured base learning rate."""

    def get_lr(self):
        return self.base_lrs


class ConstantWarmupLR(_WarmupScheduler):
    """Linear warmup followed by a constant learning rate."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                "warmup_steps must be in [0, total_steps), got "
                f"{warmup_steps} for total_steps={total_steps}"
            )
        base_scheduler = _FlatLR(optimizer, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class WarmupStableDecayLR(_LRScheduler):
    """Fixed-horizon WSD, indexed by the update about to use the current LR.

    Construction selects update zero; the optimizer advances once after each
    update. Warmup includes both floor and peak and therefore needs zero or at
    least two updates. Stable updates use peak. Decay starts one increment below
    peak and includes the floor on the last used update. Beyond the horizon the
    LR stays at floor. Each parameter group's base LR is its own peak.
    """

    _CONTRACT = (
        "total_steps",
        "warmup_steps",
        "stable_steps",
        "decay_steps",
        "min_lr_ratio",
        "lr_decay_style",
        "base_lrs",
    )

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        decay_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        lr_decay_style: str = "linear",
        last_epoch: int = -1,
    ):
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("WSD total_steps must be a positive integer")
        if decay_steps is None:
            decay_steps = int(total_steps * 0.2)
        if type(warmup_steps) is not int or warmup_steps < 0 or warmup_steps == 1:
            raise ValueError("WSD warmup_steps must be zero or at least two")
        if (
            type(decay_steps) is not int
            or not 1 <= decay_steps <= total_steps - warmup_steps
        ):
            raise ValueError("WSD decay_steps must be positive and fit after warmup")
        if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
            raise ValueError("WSD min_lr_ratio must be finite and in [0, 1]")
        if lr_decay_style not in ("linear", "cosine"):
            raise ValueError("WSD lr_decay_style must be linear or cosine")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.stable_steps = total_steps - warmup_steps - decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.lr_decay_style = lr_decay_style
        super().__init__(optimizer, last_epoch=last_epoch)

    def _factor(self, index):
        if index < self.warmup_steps:
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * index / (
                self.warmup_steps - 1
            )
        if index < self.total_steps - self.decay_steps:
            return 1.0
        progress = min(
            1.0, (index - (self.total_steps - self.decay_steps) + 1) / self.decay_steps
        )
        if progress == 1.0:
            return self.min_lr_ratio
        remaining = 1 - progress
        if self.lr_decay_style == "cosine":
            remaining = (1 + math.cos(math.pi * progress)) / 2
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * remaining

    def get_lr(self):
        return [peak * self._factor(self.last_epoch) for peak in self.base_lrs]

    def validate_state_dict(self, state_dict):
        """Reject schedule changes before mutating optimizer or scheduler state."""
        for name in self._CONTRACT:
            if name not in state_dict or state_dict[name] != getattr(self, name):
                raise ValueError(f"WSD resume schedule mismatch: {name}")
        index = state_dict.get("last_epoch")
        if type(index) is not int or index < 0:
            raise ValueError("WSD resume last_epoch must be nonnegative")
        expected = [peak * self._factor(index) for peak in self.base_lrs]
        if state_dict.get("_last_lr") != expected:
            raise ValueError("WSD resume LR disagrees with its schedule position")

    def load_state_dict(self, state_dict):
        self.validate_state_dict(state_dict)
        super().load_state_dict(state_dict)
        for group, lr in zip(
            self.optimizer.param_groups, self.get_last_lr(), strict=True
        ):
            group["lr"] = lr


__all__ = ["ConstantWarmupLR", "CosineAnnealingWarmupLR", "WarmupStableDecayLR"]
