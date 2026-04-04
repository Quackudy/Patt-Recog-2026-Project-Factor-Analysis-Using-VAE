"""
Generating data using Qlib - Alpha158 dataset
"""

import qlib
from pathlib import Path
import yaml
from qlib.constant import REG_CN, REG_US
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158


START_TIME = '2008-01-01'
END_TIME = '2026-04-03'

def main():
    # 1. Load Config
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path.absolute()}")
    
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    region_type = data_cfg.get("dataset") # Expects "US" or "CN"

    # 2. Configure Paths based on region
    if region_type == "CN":
        provider_uri = "data/raw/cn_data"
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        market = "csi300"
        output_name = 'csi_data.pkl'
    elif region_type == "US":
        provider_uri = "data/raw/us_data"
        qlib.init(provider_uri=provider_uri, region=REG_US)
        market = "sp500"
        output_name = 'sp500_data.pkl'
    else:
        raise ValueError(f"Invalid dataset region in config: {region_type}. Use 'US' or 'CN'.")

    # Ensure output directory exists
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    # 3. Data Handler Configuration
    data_handler_config = {
        "start_time": START_TIME,
        "end_time": END_TIME,
        "fit_start_time": data_cfg["fit_start_time"],
        "fit_end_time": data_cfg["fit_end_time"],
        "instruments": market,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
    }

    # 4. Fetch and Save
    print(f"Fetching {region_type} data from {provider_uri}...")
    dataset = Alpha158(**data_handler_config)
    
    # Fetch from DataHandlerLP
    df = dataset.fetch(col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    
    # Flatten MultiIndex: [feature/label, column_name] -> [column_name]
    df.columns = df.columns.droplevel(0)
    
    df.to_pickle(output_path)
    print(f"Successfully saved {df.shape} to {output_path}")

if __name__ == "__main__":
    main()
