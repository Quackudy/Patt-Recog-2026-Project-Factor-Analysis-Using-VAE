import torch
import numpy as np
import random
from dataclasses import dataclass, field
from factorVAE.model import FactorVAE, FeatureExtractor, FactorEncoder, FactorDecoder, FactorPredictor, AlphaLayer, BetaLayer
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@dataclass
class DataArgument:
    save_dir: str = field(
        default='./data',
        metadata={"help": 'directory to save model'}
    )
    fit_start_time: str = field(
        default="2010-12-01",
        metadata={"help": "fit_start_time"}
    )

    fit_end_time: str= field(
        default="2017-12-31", 
        metadata={"help": "fit_end_time"}
    )

    val_start_time : str = field(
        default='2018-01-01', 
        metadata={"help": "val_start_time"}
    )

    val_end_time: str =field(default='2018-12-31')

    seq_len : int = field(default=20)

    normalize: bool = field(
        default=True,
    )
    select_feature: str = field(
        default=None,
    )
    num_workers: int = field(default=2)



def load_model(args):
    feature_extractor = FeatureExtractor(num_latent = args.num_latent, hidden_size =args.hidden_size)

    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)

    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    return factorVAE


@torch.no_grad()
def generate_prediction_scores(model, test_dataloader, test_dataset, seq_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    model.eval()
    ls = []
    
    with tqdm(total=len(test_dataloader)) as pbar:
        for i, (char_with_label, _) in enumerate(test_dataloader):
            # The dataset includes the label in the final feature column.
            char = char_with_label[:, :, :-1].to(device)
            #returns = char_with_label[:, :, -1].to(device)
            if char.shape[1] != seq_len:
                print("unexpected seq length", char.shape)
                continue
            predictions = model.prediction(char.float())
            ls.append(predictions.detach().cpu())
            pbar.update(1)

    ls = torch.cat(ls, dim=0)
    multi_index = pd.MultiIndex.from_tuples(test_dataset.sampler.get_index(), names=["datetime","instrument"])
    ls = pd.DataFrame(ls.numpy(), index=multi_index, columns=['score'])
    return ls

@dataclass
class test_args:
    num_factor: int
    normalize: bool = True
    select_feature: bool = True
    
    batch_size: int = 300
    seq_length: int = 20

    hidden_size: int = 20
    num_latent: int = 20
    num_portfolio: int = 128

    save_dir='./best_model'
    use_qlib: bool = False
    

def RankIC(df, column1='LABEL0', column2='Pred'):
    ric_values_multiindex = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0_rank'] = daily_data[column1].rank()
        daily_data['pred_rank'] = daily_data[column2].rank()
        ric, _ = spearmanr(daily_data['LABEL0_rank'], daily_data['pred_rank'])
        ric_values_multiindex.append(ric)

    if not ric_values_multiindex:
        return np.nan, np.nan

    ric = np.mean(ric_values_multiindex)
    std = np.std(ric_values_multiindex)
    ir = ric / std if std != 0 else np.nan
    return pd.DataFrame({'RankIC': [ric], 'RankIC_IR': [ir]})
