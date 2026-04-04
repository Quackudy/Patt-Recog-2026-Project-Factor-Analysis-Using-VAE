"""Colored console logging for CLIs and pipelines."""

from __future__ import annotations

import logging
import os

from colorlog import ColoredFormatter


def _resolve_level() -> int:
    for key in ("FACTOR_VAE_LOG_LEVEL", "LOGLEVEL"):
        raw = os.environ.get(key)
        if raw:
            return getattr(logging, raw.upper(), logging.INFO)
    return logging.INFO


def configure_colored_logging(
    name: str = "factor_vae",
    level: int | None = None,
) -> logging.Logger:
    """
    Attach a single colored stream handler to the named logger.
    Level defaults to env ``FACTOR_VAE_LOG_LEVEL`` or ``LOGLEVEL`` (e.g. DEBUG), else INFO.
    """
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(level if level is not None else _resolve_level())
    log.propagate = False

    handler = logging.StreamHandler()
    handler.setLevel(log.level)
    handler.setFormatter(
        ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(white)s%(name)s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    log.addHandler(handler)
    return log
