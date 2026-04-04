# Pattern Recognition 2026 : Final Project - Implementation of FactorVAE

Minimal setup to train FactorVAE with config-based params.

## Setup (I used UV, which I think is the easiest way)
1. cd to project root (where `pyproject.toml` is visible)
2. `uv sync` (installs dependencies from `pyproject.toml`)
3. Run these two command to scrape raw US and China data using Qlib
   python -m qlib.cli.data qlib_data --target_dir ./data/raw/cn_data --region cn
   python -m qlib.cli.data qlib_data --target_dir ./data/raw/us_data --region us
3. Ensure `config.yaml` exists at root, e.g.:
   - `data.dataset` must be either `US` or `CN` 
   - `training.save_dir` is `./best_models` 
   - `mlflow.tracking_uri` is `sqlite:///mlflow.db`

## Run

- `uv run factor-vae-makedata` <- Run this the first time using project *OR you change training period*
- `uv run factor-vae-train`
- `uv run factor-vae-eval`

Run these from the repo root (where `config.yaml` lives). In code, import the library as `factor_vae` (for example `from factor_vae.training.pipeline import run_training`).

## Notes

- If path errors appear, run from root.
- For `config.yaml` not found, ensure current working directory is repo root.
