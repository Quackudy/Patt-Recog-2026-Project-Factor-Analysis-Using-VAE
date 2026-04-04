import torch
import pandas as pd
from factorVAE.dataset import init_data_loader
from factorVAE.model import FactorVAE
from factorVAE.utils import test_args, generate_prediction_scores
from factorVAE.evaluation import compute_metrics
from pathlib import Path
import yaml

def main():
    # 1. Setup Arguments (matching your best model)

    config_path = Path('config.yaml')

    if not config_path.exists():
        # This will now show the path relative to where you typed 'python...'
        raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    model_cfg = config['model']
    data_cfg = config['data']
    test_cfg = config['test']
    data_cfg = config["data"]


    args = test_args(
        num_factor=model_cfg["num_factor"], 
        num_latent=model_cfg["num_latent"],
        num_portfolio=model_cfg["num_portfolio"],
        hidden_size=model_cfg["hidden_size"],
        seq_length=data_cfg["seq_len"]
    )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    # Update this path to your actual saved weights
    model_path = Path(test_cfg['model_dir']) / test_cfg['model_name']
    
    # Initialize components as per your main.py structure
    from factorVAE.model import FeatureExtractor, FactorEncoder, AlphaLayer, BetaLayer, FactorDecoder, FactorPredictor
    
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    
    model = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 3. Load Test Data
    # Ensure the path to your .pkl file is correct
    dataset_path = data_cfg["dataset"]
    dataset = pd.read_pickle(dataset_path).iloc[:, :159]
    dataset.rename(columns={dataset.columns[-1]: 'LABEL0'}, inplace=True) 
    
    test_dataloader = init_data_loader(
        dataset,
        shuffle=False, 
        step_len=args.seq_length, 
        start=data_cfg["test_start_time"], 
        end=data_cfg["test_end_time"]
    )

    # 4. Generate Predictions
    print("Generating predictions...")
    output = generate_prediction_scores(model, test_dataloader, test_dataloader.dataset, args.seq_length)
    
    # Merge with actual labels for calculation
    output = pd.merge(output, dataset['LABEL0'], right_index=True, left_index=True)

    # 5. Run Evaluation
    print("\n--- FactorVAE Evaluation Results ---")
    results = compute_metrics(output)
    print(f"Mean Rank IC:  {results['Rank IC']:.5f}")
    print(f"Rank ICIR:     {results['Rank ICIR']:.5f}")
    print(f"Precision@10%: {results['Precision@10%']:.5f}")

if __name__ == "__main__":
    main()
