"""Build the FactorVAE stack from a ``config['model']`` dict."""

from factor_vae.models.vae.network import (
    AlphaLayer,
    BetaLayer,
    FactorDecoder,
    FactorEncoder,
    FactorPredictor,
    FactorVAE,
    FeatureExtractor,
)


def build_factor_vae(model_cfg: dict) -> FactorVAE:
    dropout = model_cfg.get("dropout", 0.1)
    gru_layers = model_cfg.get("gru_layers", 1)
    
    feature_extractor = FeatureExtractor(
        num_latent=model_cfg["num_latent"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=gru_layers,
        dropout=dropout,
    )
    factor_encoder = FactorEncoder(
        num_factors=model_cfg["num_factor"],
        num_portfolio=model_cfg["num_portfolio"],
        hidden_size=model_cfg["hidden_size"],
    )
    alpha_layer = AlphaLayer(model_cfg["hidden_size"], dropout=dropout)
    beta_layer = BetaLayer(model_cfg["hidden_size"], model_cfg["num_factor"])
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(
        model_cfg["hidden_size"], model_cfg["num_factor"], dropout=dropout
    )
    return FactorVAE(
        feature_extractor,
        factor_encoder,
        factor_decoder,
        factor_predictor,
    )
