import logging
import torch
import pandas as pd
import os
from tqdm.auto import tqdm
import yaml
import time
import mlflow
from colorlog import ColoredFormatter
from factorVAE.model import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from factorVAE.dataset import init_data_loader
from factorVAE.train_model import train, validate
from factorVAE.utils import set_seed, DataArgument

# Logging setup
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(white)s%(message)s",
    datefmt="%H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))
logger = logging.getLogger('factorVAE')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main(config=None):

    # If called via CLI (uv run factorVAE), config will be None
    if config is None:
        # Use Pathlib for more robust path handling
        from pathlib import Path
        
        config_path = Path('config.yaml')

        if not config_path.exists():
            # This will now show the path relative to where you typed 'python...'
            raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set spawn method")
    except RuntimeError:
        pass # Already set


    training = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    mlflow_cfg = config['mlflow']
    
    set_seed(training['seed'])
    if not os.path.exists(training['save_dir']):
        os.makedirs(training['save_dir'])
    
    # ── MLflow setup (local SQLite) ───────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg['experiment'])

    # create model
    feature_extractor = FeatureExtractor(num_latent=model_cfg['num_latent'], hidden_size=model_cfg['hidden_size'])
    factor_encoder = FactorEncoder(num_factors=model_cfg['num_factor'], num_portfolio=model_cfg['num_portfolio'], hidden_size=model_cfg['hidden_size'])
    alpha_layer = AlphaLayer(model_cfg['hidden_size'])
    beta_layer = BetaLayer(model_cfg['hidden_size'], model_cfg['num_factor'])
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(model_cfg['hidden_size'], model_cfg['num_factor'])
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    
    data_args = DataArgument(
        fit_start_time=data_cfg['fit_start_time'],
        fit_end_time=data_cfg['fit_end_time'],
        val_start_time=data_cfg['val_start_time'],
        val_end_time=data_cfg['val_end_time'],
        seq_len=data_cfg['seq_len'],
        num_workers=data_cfg['num_workers'],
    )
    
    # create dataloaders
    dataset = pd.read_pickle(data_cfg['dataset']).iloc[:, :159]
    dataset.rename(columns={dataset.columns[-1]: 'LABEL0'}, inplace=True)

    train_dataloader = init_data_loader(dataset,
                                        shuffle=True,
                                        step_len=data_args.seq_len, 
                                        start=data_args.fit_start_time,
                                        end=data_args.fit_end_time, 
                                        num_workers=data_args.num_workers,
                                        select_feature=data_args.select_feature)

    valid_dataloader = init_data_loader(dataset,
                                        shuffle=False, 
                                        step_len=data_args.seq_len, 
                                        start=data_args.val_start_time, 
                                        end=data_args.val_end_time, 
                                        num_workers=0,
                                        select_feature=data_args.select_feature)
    
    T_max = len(train_dataloader) * training['num_epochs']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"*************** Using {device} ***************")
    training['device'] = str(device)
        
    factorVAE.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=training['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # ── Measure data loading speed ────────────────────────────────────────────
    logger.info("Measuring data loading speed...")
    start_time = time.time()
    num_batches = 0
    for i, (batch, _) in enumerate(train_dataloader):
        num_batches += 1
        if i == 49:
            break
    avg_time = (time.time() - start_time) / num_batches
    est_epoch_min = (avg_time * len(train_dataloader)) / 60
    logger.info(f"Average time to load one batch: {avg_time:.4f} seconds")
    logger.info(f"Estimated time for one full epoch ({len(train_dataloader)} batches): {est_epoch_min:.2f} minutes")

    # ── Training loop with MLflow ─────────────────────────────────────────────
    with mlflow.start_run(run_name=training['run_name']):

        # Log all hyperparameters
        mlflow.log_params({
            "num_epochs":    training['num_epochs'],
            "lr":            training['lr'],
            "num_latent":    model_cfg['num_latent'],
            "num_portfolio": model_cfg['num_portfolio'],
            "seq_len":       data_args.seq_len,
            "num_factor":    model_cfg['num_factor'],
            "hidden_size":   model_cfg['hidden_size'],
            "num_workers":   data_args.num_workers,
            "seed":          training['seed'],
            "fit_start_time": data_args.fit_start_time,
            "fit_end_time":  data_args.fit_end_time,
            "val_start_time":data_args.val_start_time,
            "val_end_time":  data_args.val_end_time,
            "dataset":       data_cfg['dataset'],
            "device":        str(device),
            "est_epoch_min": round(est_epoch_min, 2),
        })

        for epoch in tqdm(range(training['num_epochs'])):
            train_loss = train(factorVAE, train_dataloader, optimizer, scheduler, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch + 1)

            do_validate = (epoch + 1) % training['val_interval'] == 0 or epoch == training['num_epochs'] - 1
            if do_validate:
                val_loss = validate(factorVAE, valid_dataloader, device)
                mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
                logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_root = os.path.join(
                        training['save_dir'],
                        f'{training["run_name"]}_factor_{model_cfg["num_factor"]}_hdn_{model_cfg["hidden_size"]}_port_{model_cfg["num_portfolio"]}_seed_{training["seed"]}.pt'
                    )
                    torch.save(factorVAE.state_dict(), save_root)
                    mlflow.log_artifact(save_root)
                    print(f"Model saved at {save_root}")
            else:
                logger.debug(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (validation skipped)")

        mlflow.log_metric("best_val_loss", best_val_loss)


if __name__ == '__main__':

    try:
        main()
    except Exception :
        logger.exception("Unhandled exception in FactorVAE main")
        raise
