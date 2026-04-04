from scipy.stats import spearmanr

def compute_metrics(df, pred_col='score', target_col='LABEL0'):
    """
    Computes Rank IC and ICIR based on the FactorVAE paper.
    Assumes df has a MultiIndex (datetime, instrument).
    """
    # 1. Rank IC: Daily Spearman Correlation
    def daily_rank_ic(group):
        return spearmanr(group[pred_col], group[target_col])[0]

    daily_ic = df.groupby(level='datetime').apply(daily_rank_ic)
    
    # 2. IC Metrics
    rank_ic_mean = daily_ic.mean()
    rank_ic_std = daily_ic.std()
    icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0
    
    # 3. Precision@N (Top-k Performance)
    # This measures how many of the top 10% predicted are actually in the top 10%
    def daily_precision(group, k=0.1):
        n = len(group)
        top_k = int(n * k)
        if top_k == 0: 
            return 0
        
        top_pred = group.nlargest(top_k, pred_col).index
        top_true = group.nlargest(top_k, target_col).index
        return len(set(top_pred) & set(top_true)) / top_k

    precision_at_10 = df.groupby(level='datetime').apply(daily_precision).mean()

    return {
        "Rank IC": rank_ic_mean,
        "Rank ICIR": icir,
        "Precision@10%": precision_at_10,
        "IC_Series": daily_ic
    }

if __name__ == "__main__":
    # Example usage with your output data
    # output = pd.read_pickle('your_prediction_output.pkl')
    # metrics = compute_metrics(output)
    # print(f"Rank IC: {metrics['Rank IC']:.4f}")
    # print(f"Rank ICIR: {metrics['Rank ICIR']:.4f}")
    pass
