import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from pathlib import Path
import pandas as pd

def main():
    qlib.init(provider_uri="data/raw/cn_data", region=REG_CN)
    
    # We want raw returns for the test period
    # Test period: 2019-01-01 to 2020-12-31
    data_handler_config = {
        "start_time": "2019-01-01",
        "end_time": "2020-12-31",
        "instruments": "csi300",
        "infer_processors": [], # No processing
        "learn_processors": [], # No processing
        "label": ["Ref($close, -2)/Ref($close, -1) - 1"],
    }
    
    print("Fetching raw returns for backtesting...")
    dataset = Alpha158(**data_handler_config)
    df = dataset.fetch(col_set="label")
    df.columns = ["RAW_RETURN"]
    
    output_path = Path("data/processed/raw_test_returns.pkl")
    df.to_pickle(output_path)
    print(f"Saved raw returns to {output_path}")

if __name__ == "__main__":
    main()
