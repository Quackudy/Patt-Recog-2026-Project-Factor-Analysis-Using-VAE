"""Config-driven optimizer and LR scheduler construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer


_OPTIMIZER_CLASSES: dict[str, type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def build_optimizer(model: torch.nn.Module, training: Mapping[str, Any]) -> Optimizer:
    cfg = dict(training.get("optimizer") or {})
    name = str(cfg.pop("name", "adam")).lower()
    if name not in _OPTIMIZER_CLASSES:
        raise ValueError(f"Unknown optimizer {name!r}; choose from {sorted(_OPTIMIZER_CLASSES)}")
    cls = _OPTIMIZER_CLASSES[name]
    cfg.setdefault("lr", training["lr"])
    return cls(model.parameters(), **cfg)


def current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


@dataclass
class LRScheduleHooks:
    """Encapsulates when/how to call ``scheduler.step`` (batch vs epoch vs metric)."""

    _batch_step: Callable[[], None] | None = None
    _epoch_step: Callable[[float, float | None, bool], None] | None = None

    def after_optimizer_step(self) -> None:
        if self._batch_step is not None:
            self._batch_step()

    def after_epoch(
        self,
        *,
        train_loss: float,
        val_loss: float | None,
        did_validate: bool,
    ) -> None:
        if self._epoch_step is not None:
            self._epoch_step(train_loss, val_loss, did_validate)


def build_lr_schedule(
    optimizer: Optimizer,
    training: Mapping[str, Any],
    *,
    num_epochs: int,
    steps_per_epoch: int,
) -> LRScheduleHooks:
    cfg = dict(training.get("scheduler") or {})
    name = str(cfg.pop("name", "cosine_step")).lower()

    if name == "none":
        return LRScheduleHooks()

    if name == "cosine_step":
        t_max = int(cfg.pop("T_max", num_epochs * steps_per_epoch))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, **cfg)

        def batch_step() -> None:
            scheduler.step()

        return LRScheduleHooks(_batch_step=batch_step)

    if name == "cosine_epoch":
        t_max = int(cfg.pop("T_max", num_epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, **cfg)

        def epoch_step(_train: float, _val: float | None, _did_val: bool) -> None:
            scheduler.step()

        return LRScheduleHooks(_epoch_step=epoch_step)

    if name == "reduce_on_plateau":
        monitor = str(cfg.pop("monitor", "val_loss"))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg)

        def epoch_step(train_loss: float, val_loss: float | None, did_validate: bool) -> None:
            if monitor == "val_loss":
                if not did_validate or val_loss is None:
                    return
                scheduler.step(val_loss)
            elif monitor == "train_loss":
                scheduler.step(train_loss)
            else:
                raise ValueError(f"Unknown scheduler.monitor {monitor!r}; use 'val_loss' or 'train_loss'")

        return LRScheduleHooks(_epoch_step=epoch_step)

    raise ValueError(
        f"Unknown scheduler {name!r}; choose from "
        f"none, cosine_step, cosine_epoch, reduce_on_plateau"
    )
