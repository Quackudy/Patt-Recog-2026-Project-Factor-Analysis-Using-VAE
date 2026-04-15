import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, num_latent, hidden_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x : (batch_size, seq_length, num_latent)

        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)

        stock_latent, _ = self.gru(out)
        return stock_latent[:,-1,:] # (batch_size, hidden_size)

class FactorEncoder(nn.Module):
    def __init__(self, num_factors, num_portfolio, hidden_size):
        super(FactorEncoder, self).__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(hidden_size, num_portfolio)
        self.softmax = nn.Softmax(dim=0) 
        
        self.linear_mu = nn.Linear(num_portfolio, num_factors)
        self.linear_sigma = nn.Linear(num_portfolio, num_factors)
        self.softplus = nn.Softplus()
        
    def mapping_layer(self, portfolio_return):
        # portfolio_return : (batch_size, 1)

        mean = self.linear_mu(portfolio_return.squeeze(1))
        sigma = self.softplus(self.linear_sigma(portfolio_return.squeeze(1)))
        return mean, sigma
    
    def forward(self, stock_latent, returns):
        # stock_latent: (batch_size, hidden_size)
        # returns: (batch_size, 1) (Returns for a single period)
        weights = self.linear(stock_latent)
        weights = self.softmax(weights) # (batch_size, num_portfolio)


        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        portfolio_return = torch.mm(weights.transpose(1,0), returns) 
        
        return self.mapping_layer(portfolio_return)

class AlphaLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AlphaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, stock_latent):
        stock_latent = self.linear1(stock_latent)
        stock_latent = self.leakyrelu(stock_latent)
        alpha_mu = self.mu_layer(stock_latent)
        alpha_sigma = self.sigma_layer(stock_latent)
        return alpha_mu, self.softplus(alpha_sigma)
        
class BetaLayer(nn.Module):
    """calcuate factor exposure beta(N*K)"""
    def __init__(self, hidden_size, num_factors):
        super(BetaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, num_factors)
    
    def forward(self, stock_latent):
        beta = self.linear1(stock_latent)
        return beta
        
class FactorDecoder(nn.Module):
    def __init__(self, alpha_layer, beta_layer):
        super(FactorDecoder, self).__init__()

        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, stock_latent, factor_mu, factor_sigma):
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)

        factor_sigma = factor_sigma.clone()
        factor_sigma[factor_sigma == 0] = 1e-6

        # Eq. (12): mu_y = mu_alpha + beta * mu_z
        mu = alpha_mu + torch.matmul(beta, factor_mu)
        # Eq. (12): sigma_y = sqrt(sigma_alpha^2 + beta^2 * sigma_z^2)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2) + 1e-6)

        return mu, sigma

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()

        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.capture_attention = False
        self._last_attention_weights: torch.Tensor | None = None

    def forward(self, stock_latent):

        self.key = self.key_layer(stock_latent)
        self.value = self.value_layer(stock_latent)

        # Eq. (14): normalize by ||q||_2 * ||k^(i)||_2  (cosine-style, as in paper)
        q_norm = torch.norm(self.query) + 1e-6
        k_norm = torch.norm(self.key, dim=1, keepdim=False) + 1e-6  # (N,)
        attention_weights = torch.matmul(self.query, self.key.transpose(1, 0)) / (q_norm * k_norm)  # (N,)

        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights)  # max(0, x) as in paper

        weight_sum = attention_weights.sum() + 1e-6
        attention_weights = attention_weights / weight_sum  # normalize to sum=1

        if self.capture_attention:
            self._last_attention_weights = attention_weights.detach().clone()

        context_vector = torch.matmul(attention_weights, self.value)  # (H,)
        return context_vector

class FactorPredictor(nn.Module):
    def __init__(self, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_factor = num_factor
        self.attention_layers = nn.ModuleList([AttentionLayer(self.hidden_size) for _ in range(num_factor)])
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def set_attention_capture(self, enabled: bool) -> None:
        """Enable storing attention weights on each ``AttentionLayer`` (for visualization)."""
        for layer in self.attention_layers:
            layer.capture_attention = enabled
            if not enabled:
                layer._last_attention_weights = None

    def forward(self, stock_latent):
        # stock_latent : (N, H)
        
        for i in range(self.num_factor):
            attention_layer = self.attention_layers[i](stock_latent)
            if i == 0:
                h_multi = attention_layer
            else:
                h_multi = torch.cat((h_multi, attention_layer), dim=0)
        h_multi = h_multi.view(self.num_factor, -1)

        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        pred_mu = self.mu_layer(h_multi)
        pred_sigma = self.sigma_layer(h_multi)
        pred_sigma = self.softplus(pred_sigma)
        pred_mu = pred_mu.view(-1)
        pred_sigma = pred_sigma.view(-1)
        return pred_mu, pred_sigma

class FactorVAE_old(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        # mu1, mu2: (batch_size, 1)
        # sigma1, sigma2: (batch_size, 1)
        # output: (batch_size, 1)
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div

    def forward(self, x, returns):
        # x: (batch_size, seq_length, num_latent)
        # returns: (batch_size, 1)
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        rec_mu, rec_sigma = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        reconstruction_loss = FactorVAE.gaussian_nll_loss(rec_mu, rec_sigma, returns)
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        return vae_loss, rec_mu, factor_mu, factor_sigma, pred_mu, pred_sigma

    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        rec_mu, rec_sigma = self.factor_decoder(stock_latent, pred_mu, pred_sigma)
        return rec_mu
    
class FactorVAE(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        # mu1, mu2: (batch_size, 1)
        # sigma1, sigma2: (batch_size, 1)
        # output: (batch_size, 1)
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div

    @staticmethod
    def gaussian_nll_loss(mu, sigma, target):
        """Negative log-likelihood of a Gaussian distribution.
        Implements the first term of Eq. (17): -log P(y_rec | x, z_post).
        = 0.5 * [log(2π) + log(σ²) + (y - µ)²/σ²]  summed over stocks.
        """
        nll = 0.5 * (torch.log(2 * torch.tensor(torch.pi)) +
                     torch.log(sigma**2 + 1e-6) +
                     (target - mu)**2 / (sigma**2 + 1e-6))
        return nll.mean()

    def forward(self, x, returns):
        # x: (batch_size, seq_length, num_latent)
        # returns: (batch_size, 1)

        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)

        # Decoder returns (mu, sigma) of the return distribution — Eq. (12)
        rec_mu, rec_sigma = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        # Loss term 1: negative log-likelihood for reconstruction — Eq. (17) first term
        reconstruction_loss = self.gaussian_nll_loss(rec_mu, rec_sigma, returns)

        # Loss term 2: KL divergence KL(posterior || prior) — Eq. (17) second term
        pred_sigma = pred_sigma.clone()
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        return vae_loss, rec_mu, factor_mu, factor_sigma, pred_mu, pred_sigma

    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        rec_mu, rec_sigma = self.factor_decoder(stock_latent, pred_mu, pred_sigma)
        return rec_mu
