"""Offline evaluation from config (test window, saved weights)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
import os

from factor_vae.data.dataset import init_data_loader
from factor_vae.evaluation.metrics import compute_metrics
from factor_vae.inference.prediction import generate_prediction_scores
from factor_vae.models.network_profile import log_network_profile
from factor_vae.models.vae.factory import build_factor_vae

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOG = logging.getLogger(__name__)


def run_evaluation(
    config: Mapping,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Load checkpoint from ``config['test']``, score the test period from ``config['data']``,
    merge labels, and return ``compute_metrics`` output.
    """
    log = logger or _LOG
    model_cfg = config["model"]
    data_cfg = config["data"]
    test_cfg = config["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluation device: %s", device)
    model_path = Path(test_cfg["model_dir"]) / test_cfg["model_name"]
    model = build_factor_vae(model_cfg)
    log_network_profile(model, log, model_cfg=dict(model_cfg))
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

        
    dataset_type = data_cfg.get("dataset")

    if dataset_type == "US":
        path = "data/processed/sp500_data.pkl"
    elif dataset_type == "CN":
        path = "data/processed/csi_data.pkl"
    else:
         raise ValueError(f"Invalid dataset type: {dataset_type}. Expected 'US' or 'CN'.")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the pickle file at: {os.path.abspath(path)}. "
            "Make sure you ran the Qlib data generation script first!"
        )

    dataset = pd.read_pickle(path).iloc[:, :159]

    dataset.rename(columns={dataset.columns[-1]: "LABEL0"}, inplace=True)

    seq_len = data_cfg["seq_len"]
    test_dataloader = init_data_loader(
        dataset,
        shuffle=False,
        step_len=seq_len,
        start=data_cfg["test_start_time"],
        end=data_cfg["test_end_time"],
        select_feature=data_cfg.get("select_feature"),
    )

    test_ds = test_dataloader.dataset
    try:
        scores = generate_prediction_scores(
            model,
            test_dataloader,
            test_ds,
            seq_len,
            logger=log,
        )
        merged = pd.merge(scores, dataset["LABEL0"], right_index=True, left_index=True)
        return compute_metrics(merged)
    finally:
        del test_dataloader
        test_ds.cleanup()
