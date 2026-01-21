import pandas as pd
import glob
import os
from qlib.data import D
import qlib

def check_ic():
    # Initialize Qlib (Minimal)
    qlib.init(provider_uri=r"C:\Users\wuwei\.qlib\qlib_data\cn_data", region="cn") # Default, just need libs
    # Wait, user is using US data or CN?
    # Config says: provider_uri: "~/.qlib/qlib_data/us_data"
    # I should use the path from config.yaml if possible, or just standard load.
    
    # Load Preds
    base_path = r"D:\Work\Antigravity\qlib_strategy_test"
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Loading artifacts from: {latest_run}")
    
    pred_path = os.path.join(latest_run, "pred.pkl")
    label_path = os.path.join(latest_run, "label.pkl")
    
    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        print("Predictions or Labels not found.")
        return

    pred = pd.read_pickle(pred_path)
    label = pd.read_pickle(label_path)
    
    # Ensure index alignment
    # pred usually has (datetime, instrument) index
    # label same
    
    common_index = pred.index.intersection(label.index)
    pred = pred.loc[common_index]
    label = label.loc[common_index]
    
    # Combine
    df = pd.concat([pred, label], axis=1)
    df.columns = ['score', 'return']
    
    # Filter for Training Period (2010-2020)
    # index level 0 is datetime
    df_train = df.loc[pd.IndexSlice["2010-01-01":"2020-12-31", :], :]
    
    print(f"Training Period Data Points: {len(df_train)}")
    
    # Calculate IC (Rank Correlation)
    # Group by Date, then correlation
    def calc_ic(g):
        return g['score'].corr(g['return'], method='pearson') # Pearson or Spearman
        
    def calc_rank_ic(g):
        return g['score'].corr(g['return'], method='spearman')
    
    daily_ic = df_train.groupby(level='datetime').apply(calc_ic)
    daily_rank_ic = df_train.groupby(level='datetime').apply(calc_rank_ic)
    
    print(f"\n--- IC Analysis (2010-2020) ---")
    print(f"Mean Pearson IC: {daily_ic.mean():.4f}")
    print(f"Mean Rank IC (Information Coefficient): {daily_rank_ic.mean():.4f}")
    print(f"ICStd: {daily_ic.std():.4f}")
    print(f"ICIR (IC / Std): {daily_ic.mean() / daily_ic.std():.4f}")
    
    print("\n--- Interpretation ---")
    if daily_rank_ic.mean() > 0:
        print("POSITIVE IC: The model significantly verified it could predict returns.")
        print("The fact it lost money means 'High Rank' stocks performed worse than 'Low Rank' stocks relative to cost/risk?")
        print("Wait. If Rank IC is positive, High Score = High Return.")
        print("So, if we bought High Score stocks, we SHOULD make money (before costs).")
    else:
        print("NEGATIVE/ZERO IC: The model failed to predict returns.")

if __name__ == "__main__":
    check_ic()
