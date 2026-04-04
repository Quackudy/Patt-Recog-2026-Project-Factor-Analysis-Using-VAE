"""CLI entry: load ``config.yaml`` and run training."""

from pathlib import Path

import yaml

from factor_vae.logging_config import configure_colored_logging
from factor_vae.training.pipeline import run_training


def main(config=None, *, logger=None):
    if config is None:
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_path.absolute()}"
            )
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    log = logger or configure_colored_logging()
    run_training(config, logger=log)


if __name__ == "__main__":
    log = configure_colored_logging()
    try:
        main(logger=log)
    except Exception:
        log.exception("Unhandled exception in training CLI")
        raise
