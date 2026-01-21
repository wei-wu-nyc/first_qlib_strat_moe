import pandas as pd
import pickle
import os
import glob

def inspect_data():
    base_path = r"D:\Work\Antigravity\qlib_strategy_test"
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Inspecting artifacts from: {latest_run}")
    
    report_path = os.path.join(latest_run, "portfolio_analysis", "report_normal_1day.pkl")
    with open(report_path, "rb") as f:
        df = pickle.load(f)
    
    # Check 2022 performance specifically
    if isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index
    else:
        df = df.reset_index()
    
    df['date'] = pd.to_datetime(df['date'])
    df_2022 = df[(df['date'] >= "2022-01-01") & (df['date'] <= "2022-12-31")]
    
    print("\n2022 Stats:")
    print(df_2022['return'].describe())
    print(f"2022 Cumulative Return: {(1 + df_2022['return']).prod() - 1:.4f}")
    
    # Check 2010-2020 (Train)
    df_train = df[(df['date'] >= "2010-01-01") & (df['date'] <= "2020-12-31")]
    print(f"\nTrain Cumulative Return: {(1 + df_train['return']).prod() - 1:.4f}")

if __name__ == "__main__":
    inspect_data()
