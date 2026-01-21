import json
import pandas as pd

with open(r"D:\Work\Antigravity\qlib_strategy_test\dashboard\data.json", "r") as f:
    data = json.load(f)

ec = data['equity_curve']
dates = ec['dates']
strategy = ec['strategy']
benchmark = ec['benchmark']
segments = ec['segments']

df = pd.DataFrame({'date': dates, 'strategy': strategy, 'benchmark': benchmark, 'segment': segments})

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

print("\nLast 5 rows of Training (if any):")
print(df[df['segment'] == 'Train'].tail())

print("\nLast 5 rows of Valid (if any):")
print(df[df['segment'] == 'Valid'].tail())

print("\nFirst 5 rows of Test:")
print(df[df['segment'] == 'Test'].head())
print("\nLast 5 rows of Test:")
print(df[df['segment'] == 'Test'].tail())

# Check for Step-Wise behavior (zeros?)
print("\nZero diffs check (Strategy):")
print(df['strategy'].diff().value_counts().head())
