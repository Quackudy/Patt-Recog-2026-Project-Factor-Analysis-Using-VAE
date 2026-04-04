"""CLI entry: load ``config.yaml`` and run test-window evaluation."""

from pathlib import Path

import yaml

from factor_vae.evaluation.pipeline import run_evaluation
from factor_vae.logging_config import configure_colored_logging


def main():
    log = configure_colored_logging()
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path.absolute()}"
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log.info("Generating predictions...")
    results = run_evaluation(config, logger=log)

    log.info("--- FactorVAE Evaluation Results ---")
    log.info("Mean Rank IC:  %.5f", results["Rank IC"])
    log.info("Rank ICIR:     %.5f", results["Rank ICIR"])
    log.info("Precision@10%%: %.5f", results["Precision@10%"])


if __name__ == "__main__":
    main()
