import pandas as pd
import numpy as np
import json
import os
import glob

def diagnose():
    print("Loading data...")
    # Load Dashboard Data for Equity Curve
    with open(r"D:\Work\Antigravity\qlib_strategy_test\dashboard\data.json", "r") as f:
        data = json.load(f)
    
    ec = data['equity_curve']
    df = pd.DataFrame({
        'date': pd.to_datetime(ec['dates']),
        'segment': ec['segments'],
        'strategy': ec['strategy'], # This is Geometric or Log, doesn't matter for diffs
    })
    
    # We need daily returns to analyze properly, not just cumulative
    # Getting them from the latest pickle is better
    base_path = r"D:\Work\Antigravity\qlib_strategy_test"
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Analyzing artifacts from: {latest_run}")
    
    report_path = os.path.join(latest_run, "portfolio_analysis", "report_normal_1day.pkl")
    with open(report_path, "rb") as f:
        import pickle
        report = pickle.load(f)
    
    # Adjust report index
    if not isinstance(report.index, pd.DatetimeIndex):
        report.index = pd.to_datetime(report.index)
        
    print("\n--- REGIME ANALYSIS ---")
    # We don't have the raw 'is_bullish' signal saved in the report.
    # But we can infer it? No, strategy logic is hidden in execution.
    # However, we can approximate it by looking at the Benchmark.
    # Benchmark: typically SPY or similar.
    
    bench_ret = report['bench']
    # Reconstruct MA60 Regime
    # We need price history.
    # Approximate price from returns
    bench_price = (1 + bench_ret).cumprod()
    bench_ma60 = bench_price.rolling(window=60).mean()
    
    # Inferred Regime
    is_bullish = (bench_price > bench_ma60).astype(int)
    
    # Count Regimes per Year
    report['year'] = report.index.year
    report['is_bullish'] = is_bullish
    
    regime_stats = report.groupby('year')['is_bullish'].value_counts(normalize=True).unstack().fillna(0)
    print("Percentage of Time in BULLISH Regime (Approx):")
    print(regime_stats[1] * 100) # Show % Bullish
    
    print("\n--- RETURNS BY YEAR ---")
    annual_ret = report.groupby('year')['return'].apply(lambda x: (1 + x).prod() - 1)
    print(annual_ret)
    
    print("\n--- CRITICAL CHECK: 2010-2020 ---")
    train_df = report['2010':'2020']
    train_ret = (1 + train_df['return']).prod() - 1
    print(f"Total Training Period Return: {train_ret:.2%}")
    
    print("\n--- CORRELATION CHECK ---")
    # If we had the raw signals (pred.pkl), we could check correlation.
    # Let's try to load pred.pkl
    try:
        pred_path = os.path.join(latest_run, "pred.pkl")
        pred = pd.read_pickle(pred_path)
        print("Predictions loaded successfully.")
        print("Score stats:")
        print(pred.describe())
    except Exception as e:
        print(f"Could not load predictions: {e}")

if __name__ == "__main__":
    diagnose()
