
import pandas as pd
import pickle
import os
import json
import glob

def load_latest_artifacts(base_path):
    # Find the latest experiment run
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    if not runs:
        raise FileNotFoundError("No artifacts found in mlruns")
    
    # Sort by modification time to get the latest
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Loading artifacts from: {latest_run}")
    return latest_run

def export_data():
    base_path = r"D:\Work\Antigravity\qlib_strategy_test"
    artifact_path = load_latest_artifacts(base_path)
    
    # Load Portfolio Analysis
    port_analysis_path = os.path.join(artifact_path, "portfolio_analysis", "port_analysis_1day.pkl")
    report_normal_path = os.path.join(artifact_path, "portfolio_analysis", "report_normal_1day.pkl")
    
    data = {}

    # 1. Equity Curve & Segments
    try:
        with open(report_normal_path, "rb") as f:
            report_df = pickle.load(f)
        
        # Reset index to get Date as a column
        report_df = report_df.reset_index()
        if 'datetime' not in report_df.columns and 'index' in report_df.columns:
             report_df.rename(columns={'index': 'datetime'}, inplace=True)
        
        report_df['date'] = pd.to_datetime(report_df['datetime'])
        report_df['date_str'] = report_df['date'].dt.strftime('%Y-%m-%d')
        
        # Define Segments
        train_end = pd.Timestamp("2020-12-31")
        valid_end = pd.Timestamp("2021-12-31")
        
        def get_segment(d):
            if d <= train_end:
                return "Train"
            elif d <= valid_end:
                return "Valid"
            else:
                return "Test"
        
        report_df['segment'] = report_df['date'].apply(get_segment)
        
        report_df['segment'] = report_df['date'].apply(get_segment)
        
        # Calculate Equity Curve (Geometric Compounding)
        # User wants "Equity" (Account Balance).
        # We MUST clip returns to > -1.0 to ensure the Equity never hits exactly 0 (which breaks re-basing).
        # clipping at -0.9 means max daily loss is 90%.
        
        report_df['return'] = report_df['return'].fillna(0).clip(lower=-0.9)
        report_df['bench'] = report_df['bench'].fillna(0).clip(lower=-0.9)
        
        # Cumulative Product (Start at 1.0)
        report_df['strategy_cum'] = (1 + report_df['return']).cumprod()
        report_df['benchmark_cum'] = (1 + report_df['bench']).cumprod()
        
        # Calculate Indicators on Benchmark Cumulative (Proxy for Price)
        # This ensures they are on the same scale as the equity curve
        report_df['bench_ma20'] = report_df['benchmark_cum'].rolling(window=20).mean()
        report_df['bench_ma60'] = report_df['benchmark_cum'].rolling(window=60).mean()
        
        # Determine Regime based on these MAs (Replicating Strategy Logic)
        # Uptrend (1): Price > MA60 AND MA20 > MA60
        # Downtrend (-1): Price < MA60 AND MA20 < MA60
        # Choppy (0): Else
        def get_regime(row):
            price = row['benchmark_cum']
            ma20 = row['bench_ma20']
            ma60 = row['bench_ma60']
            
            if pd.isna(ma60): return 0
            
            if price > ma60 and ma20 > ma60:
                return 1 # Uptrend
            elif price < ma60 and ma20 < ma60:
                return -1 # Downtrend
            else:
                return 0 # Choppy
                
        report_df['regime'] = report_df.apply(get_regime, axis=1)
        
        equity_curve = {
            "dates": report_df['date_str'].tolist(),
            "strategy": report_df['strategy_cum'].tolist(),
            "benchmark": report_df['benchmark_cum'].tolist(),
            "bench_ma20": report_df['bench_ma20'].fillna(0).tolist(),
            "bench_ma60": report_df['bench_ma60'].fillna(0).tolist(),
            "regimes": report_df['regime'].tolist(),
            "segments": report_df['segment'].tolist()
        }
        data["equity_curve"] = equity_curve

        # 2. Metrics (Calculated ONLY on Test Segment 2022-2025)
        test_df = report_df[report_df['segment'] == 'Test'].copy()
        
        if not test_df.empty:
            # Annualized Return
            # Total Return = (Final / Initial) - 1
            # Annualized = (1 + Total)^(252 / Days) - 1
            total_ret = (1 + test_df['return']).prod() - 1
            days = len(test_df)
            ann_ret = (1 + total_ret) ** (252 / days) - 1
            
            # Max Drawdown
            cum = (1 + test_df['return']).cumprod()
            max_cum = cum.cummax()
            drawdown = (cum - max_cum) / max_cum
            max_dd = drawdown.min()
            
            # Information Ratio
            # IR = Mean(Active Return) / Std(Active Return) * sqrt(252)
            active_ret = test_df['return'] - test_df['bench']
            ir = active_ret.mean() / active_ret.std() * (252 ** 0.5)
            
            # Sharpe Ratio
            # Sharpe = Mean(Return) / Std(Return) * sqrt(252) (assuming 0 risk free for simplicity)
            sharpe = test_df['return'].mean() / test_df['return'].std() * (252 ** 0.5)
            
            metrics = {
                "annualized_return": ann_ret,
                "information_ratio": ir,
                "max_drawdown": max_dd,
                "sharpe_ratio": sharpe
            }
        else:
             metrics = {
                "annualized_return": 0.0,
                "information_ratio": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            }
        data["metrics"] = metrics

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        data["metrics"] = {}
        data["equity_curve"] = {"dates": [], "strategy": [], "benchmark": [], "segments": []}

    # 3. Save to JSON
    output_path = os.path.join(base_path, "dashboard", "data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Data exported to {output_path}")

if __name__ == "__main__":
    export_data()
