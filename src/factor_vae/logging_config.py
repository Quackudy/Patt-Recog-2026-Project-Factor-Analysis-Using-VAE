"""Colored console logging for CLIs and pipelines."""

from __future__ import annotations

import logging
import yaml
from pathlib import Path
from colorlog import ColoredFormatter


def _resolve_level() -> int:

    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path.absolute()}"
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    level_name = config.get("log_level", "INFO")

    if isinstance(level_name, str):
        return getattr(logging, level_name.upper(), logging.INFO)
    
    if isinstance(level_name, int):
        return level_name

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
