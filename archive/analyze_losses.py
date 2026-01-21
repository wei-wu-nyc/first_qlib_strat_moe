import pandas as pd
import numpy as np
import os
import glob
from qlib.data import D
import qlib

def analyze_losses():
    print("Initializing Qlib...")
    # Use config path
    qlib.init(provider_uri=r"d:\Work\TradeStationData\Auto_Export_Data\20251231\qlib_antigravity_daily\us_data", region="us")
    
    # Locate latest artifacts
    base_path = r"D:\Work\Antigravity\qlib_strategy_test"
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    latest_run = max(runs, key=os.path.getmtime)
    print(f"Analyzing artifacts from: {latest_run}")
    
    # Load Report (Returns)
    report_path = os.path.join(latest_run, "portfolio_analysis", "report_normal_1day.pkl")
    report = pd.read_pickle(report_path)
    
    # Load Positions (Holdings)
    pos_path = os.path.join(latest_run, "portfolio_analysis", "positions_normal_1day.pkl")
    # Positions structure: Dictionary or Key-Value?
    # Usually a list of Position objects or a Dict of Date -> {Stock: Check, Cash: X}
    positions = pd.read_pickle(pos_path)
    
    # Filter for Training Period (2010-2020) where the damage happened
    train_report = report.loc["2010-01-01":"2020-12-31"]
    
    # Identify Worst Days
    # return column
    worst_days = train_report.sort_values('return', ascending=True).head(5)
    print("\n--- TOP 5 WORST DAYS (2010-2020) ---")
    print(worst_days[['return', 'turnover', 'cost']])
    
    print("\n--- DEEP DIVE INTO CULPRITS ---")
    for date, row in worst_days.iterrows():
        date_ts = pd.Timestamp(date)
        print(f"\nDate: {date_ts.date()} | Portfolio Return: {row['return']:.2%}")
        
        # Get Position for this day
        # Positions dict keys are Timestamps?
        # Qlib positions structure is a bit complex. It might be a dict of AccountPosition.
        # Let's inspect the object type first safely.
        
        if isinstance(positions, dict):
            # Check if date exists
            # Keys might be datetime.date or datetime.datetime
            # Let's try converting keys to normalized Timestamps
            # But first print one key to see format
            if 'first_key_printed' not in globals():
                 print(f"Sample Position Key: {list(positions.keys())[0]} (Type: {type(list(positions.keys())[0])})")
                 globals()['first_key_printed'] = True
            
            # Qlib Report Return on Day T is from holding positions from Day T-1.
            # So we need to look up the position at T-1 Close.
            # We can find T-1 by looking at the sorted index limits.
            # But simpler: just iterate positions keys and find the one immediately preceding.
            
            sorted_keys = sorted(positions.keys())
            # Find index of date_ts
            try:
                # Exact match for T? No, we want T-1.
                # If we held it T-1 to T.
                # Actually, Qlib records position at end of T.
                # The return on T is (Value_T - Value_T-1) / Value_T-1.
                # Value_T is based on Position_T-1 prices at T?
                # Usually: Start with Pos_T-1. Prices move. End with Pos_T (after trading).
                # So the "Stocks Held" that caused the return are Pos_T-1.
                
                # Find the key <= date_ts - 1 day?
                # Let's just find the closest key before date_ts.
                
                prev_date = max([k for k in sorted_keys if k < date_ts])
                print(f"  Fetching positions from previous close: {prev_date}")
                daily_pos = positions.get(prev_date)
            except ValueError:
                print("  No previous position found.")
                daily_pos = None

            if not daily_pos:
                pass
        else:
            print("Unknown position format.")
            continue
            
        if not daily_pos:
            print("No position data found for this date.")
            continue
            
        # Extract stock holdings
        # daily_pos class usually behaves like a dict {stock_id: amount/weight}
        # In Qlib 'Position' object, .current_position is the dict.
        
        stock_list = []
        try:
            # Check attribute
            if hasattr(daily_pos, 'position'):
                 holdings = daily_pos.position # format {stock: amount}
            elif isinstance(daily_pos, dict):
                 holdings = daily_pos
            else:
                 holdings = {}
        except:
            holdings = {}

        stocks = [s for s in holdings.keys() if s != 'cash']
        if not stocks:
            print("No stocks held.")
            continue
            
        # Get Returns for these stocks on this day
        # Using Qlib D.features
        fields = ["$close", "Ref($close, -1)/$close - 1"] # Return (Close to Close)
        names = ["price", "ret"]
        
        # D.features requires list of strings
        stock_data = D.features(stocks, fields, start_time=date_ts, end_time=date_ts)
        stock_data.columns = names
        
        # Calculate contribution?
        # We need weights.
        # If calculating weights is hard, just sorting by worst return is enough.
        
        worst_stocks = stock_data.sort_values('ret', ascending=True).head(5)
        print("  Worst Performers Held:")
        for idx, s_row in worst_stocks.iterrows():
            # idx is (datetime, instrument)
            inst = idx[1]
            print(f"    {inst}: {s_row['ret']:.2%} (Price: {s_row['price']})")

    # Analyze Turnover Costs
    avg_turnover = train_report['turnover'].mean()
    avg_cost = train_report['cost'].mean()
    print(f"\n--- COST ANALYSIS ---")
    print(f"Average Daily Turnover: {avg_turnover:.2%}")
    print(f"Average Daily Cost: {avg_cost:.2%}")
    print(f"Annualized Cost Drag: {(1-(1-avg_cost)**252 - 1)*-1:.2%}") # Approx

if __name__ == "__main__":
    analyze_losses()
