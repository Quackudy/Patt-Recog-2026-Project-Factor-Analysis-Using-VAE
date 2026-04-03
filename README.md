# Pattern Recognition 2026 : Final Project - Implementation of FactorVAE

Minimal setup to train FactorVAE with config-based params.

## Setup (I used UV, which I think is the easiest way)
1. cd to project root (where `pyproject.toml` is visible)
2. `uv sync` (installs dependencies from `pyproject.toml`)
3. Ensure `config.yaml` exists at root, e.g.:
   - `data.dataset` points to `./data/sp500_data.pkl`
   - `training.save_dir` is `./best_models`
   - `mlflow.tracking_uri` is `sqlite:///mlflow.db`

## Run

- `uv run factorVAE`
- Or explicit:
  - `uv run python scripts/main.py`

## Notes

- If path errors appear, run from root.
- For `config.yaml` not found, ensure current working directory is repo root.
