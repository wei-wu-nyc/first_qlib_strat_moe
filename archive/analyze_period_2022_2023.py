
import pandas as pd
import pickle
import os
import glob
import numpy as np

def load_latest_report():
    base_path = r"D:\Work\Antigravity\qlib_strategy_test"
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    if not runs:
        raise FileNotFoundError("No artifacts found")
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Loading from: {latest_run}")
    
    report_path = os.path.join(latest_run, "portfolio_analysis", "report_normal_1day.pkl")
    with open(report_path, "rb") as f:
        df = pickle.load(f)
    return df

def analyze_period():
    df = load_latest_report()
    df = df.reset_index()
    if 'datetime' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'datetime'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)

    # Filter for 2022-2023
    period_df = df['2022-01-01':'2023-12-31'].copy()
    
    print("\n--- Performance Analysis (2022-01-01 to 2023-12-31) ---")
    
    # 1. Total Return
    total_ret = (1 + period_df['return']).prod() - 1
    bench_ret = (1 + period_df['bench']).prod() - 1
    print(f"Strategy Total Return: {total_ret:.2%}")
    print(f"Benchmark (SPY) Return: {bench_ret:.2%}")
    
    # 2. Max Drawdown
    cum_ret = (1 + period_df['return']).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    print(f"Max Drawdown: {max_dd:.2%}")
    
    # 3. Correlation
    corr = period_df['return'].corr(period_df['bench'])
    print(f"Correlation with Benchmark: {corr:.4f}")
    
    # 4. Monthly Returns (Check for consistent losses)
    # Explicitly select numeric columns to avoid TypeError with datetime/string columns
    numeric_cols = period_df[['return', 'bench']]
    monthly = numeric_cols.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    print("\nMonthly Returns (Strategy vs Benchmark):")
    for date, row in monthly.iterrows():
        print(f"{date.strftime('%Y-%m')}: Strat {row['return']:6.2%} | Bench {row['bench']:6.2%} | Diff {row['return']-row['bench']:6.2%}")
        
    # 5. Volatility
    strat_vol = period_df['return'].std() * np.sqrt(252)
    bench_vol = period_df['bench'].std() * np.sqrt(252)
    print(f"\nStrategy Volatility (Ann.): {strat_vol:.2%}")
    print(f"Benchmark Volatility (Ann.): {bench_vol:.2%}")

if __name__ == "__main__":
    analyze_period()
