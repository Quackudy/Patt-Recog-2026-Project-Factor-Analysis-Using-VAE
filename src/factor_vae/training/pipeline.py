"""End-to-end training orchestration (MLflow, checkpoints)."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
import torch
from tqdm.auto import tqdm

from factor_vae.data.dataset import init_data_loader
from factor_vae.models.network_profile import log_network_profile
from factor_vae.models.vae.factory import build_factor_vae
from factor_vae.seed import set_seed
from factor_vae.training.loop import train, validate
from factor_vae.training.optim_factory import (
    build_lr_schedule,
    build_optimizer,
    current_lr,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOG = logging.getLogger(__name__)


def run_training(config: Mapping, *, logger: logging.Logger | None = None) -> None:
    """Train from a loaded ``config`` dict (same shape as ``config.yaml``)."""
    log = logger or _LOG

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        log.info("Set spawn method")
    except RuntimeError:
        pass

    training = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    mlflow_cfg = config["mlflow"]

    set_seed(training["seed"])
    if not os.path.exists(training["save_dir"]):
        os.makedirs(training["save_dir"])

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment"])

    model = build_factor_vae(model_cfg)
    log_network_profile(model, log, model_cfg=dict(model_cfg))

        
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

    train_dataloader = init_data_loader(
        dataset,
        shuffle=True,
        step_len=data_cfg["seq_len"],
        start=data_cfg["fit_start_time"],
        end=data_cfg["fit_end_time"],
        num_workers=data_cfg["num_workers"],
        select_feature=data_cfg.get("select_feature"),
    )

    valid_dataloader = init_data_loader(
        dataset,
        shuffle=False,
        step_len=data_cfg["seq_len"],
        start=data_cfg["val_start_time"],
        end=data_cfg["val_end_time"],
        num_workers=0,
        select_feature=data_cfg.get("select_feature"),
    )

    train_ds = train_dataloader.dataset
    valid_ds = valid_dataloader.dataset
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("*************** Using %s ***************", device)
        training["device"] = str(device)

        model.to(device)
        best_val_loss = 10000.0
        optimizer = build_optimizer(model, training)
        steps_per_epoch = len(train_dataloader)
        lr_hooks = build_lr_schedule(
            optimizer,
            training,
            num_epochs=training["num_epochs"],
            steps_per_epoch=steps_per_epoch,
        )

        log.info("Measuring data loading speed...")
        bench_start = time.time()
        num_batches = 0
        for i, (batch, _) in enumerate(train_dataloader):
            num_batches += 1
            if i == 49:
                break
        avg_time = (time.time() - bench_start) / num_batches
        est_epoch_min = (avg_time * len(train_dataloader)) / 60
        log.info("Average time to load one batch: %.4f seconds", avg_time)
        log.info(
            "Estimated time for one full epoch (%d batches): %.2f minutes",
            len(train_dataloader),
            est_epoch_min,
        )

        with mlflow.start_run(run_name=training["run_name"]):
            mlflow.log_params(
                {
                    "num_epochs": training["num_epochs"],
                    "lr": training["lr"],
                    "optimizer": (training.get("optimizer") or {}).get("name", "adam"),
                    "scheduler": (training.get("scheduler") or {}).get("name", "cosine_step"),
                    "num_latent": model_cfg["num_latent"],
                    "num_portfolio": model_cfg["num_portfolio"],
                    "seq_len": data_cfg["seq_len"],
                    "num_factor": model_cfg["num_factor"],
                    "hidden_size": model_cfg["hidden_size"],
                    "num_workers": data_cfg["num_workers"],
                    "seed": training["seed"],
                    "fit_start_time": data_cfg["fit_start_time"],
                    "fit_end_time": data_cfg["fit_end_time"],
                    "val_start_time": data_cfg["val_start_time"],
                    "val_end_time": data_cfg["val_end_time"],
                    "dataset": data_cfg["dataset"],
                    "device": str(device),
                    "est_epoch_min": round(est_epoch_min, 2),
                }
            )

            for epoch in tqdm(range(training["num_epochs"])):
                train_loss = train(
                    model,
                    train_dataloader,
                    optimizer,
                    device,
                    after_optimizer_step=lr_hooks.after_optimizer_step,
                )
                mlflow.log_metric("train_loss", train_loss, step=epoch + 1)

                do_validate = (
                    (epoch + 1) % training["val_interval"] == 0
                    or epoch == training["num_epochs"] - 1
                )
                val_loss: float | None = None
                if do_validate:
                    val_loss = validate(model, valid_dataloader, device)
                    mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
                    log.info(
                        "Epoch %d: Train Loss: %.4f, Validation Loss: %.4f",
                        epoch + 1,
                        train_loss,
                        val_loss,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_root = os.path.join(
                            training["save_dir"],
                            f'{training["run_name"]}_factor_{model_cfg["num_factor"]}_hdn_{model_cfg["hidden_size"]}_port_{model_cfg["num_portfolio"]}_seed_{training["seed"]}.pt',
                        )
                        torch.save(model.state_dict(), save_root)
                        mlflow.log_artifact(save_root)
                        log.info("Model saved at %s", save_root)
                else:
                    log.debug(
                        "Epoch %d: Train Loss: %.4f (validation skipped)",
                        epoch + 1,
                        train_loss,
                    )

                lr_hooks.after_epoch(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    did_validate=do_validate,
                )
                mlflow.log_metric("lr", current_lr(optimizer), step=epoch + 1)

            mlflow.log_metric("best_val_loss", best_val_loss)
    finally:
        del train_dataloader
        del valid_dataloader
        train_ds.cleanup()
        valid_ds.cleanup()
