"""DEBUG-level summaries of ``nn.Module`` instances (params, submodules)."""

from __future__ import annotations

import logging

import torch.nn as nn


def log_network_profile(
    model: nn.Module,
    logger: logging.Logger,
    *,
    model_cfg: dict | None = None,
) -> None:
    """Log parameter counts and top-level submodule breakdown at DEBUG only."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    lines: list[str] = []
    if model_cfg is not None:
        lines.append(f"model_cfg: {model_cfg!r}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"parameters: total={total:,} trainable={trainable:,}")

    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        mod_type = type(mod).__name__
        lines.append(f"  {name}: {mod_type} params={n:,}")

    logger.debug("Network profile:\n%s", "\n".join(lines))
