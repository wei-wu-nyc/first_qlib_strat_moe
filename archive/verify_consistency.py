import json
import numpy as np

with open(r"D:\Work\Antigravity\qlib_strategy_test\dashboard\data.json", "r") as f:
    data = json.load(f)

ec = data['equity_curve']
strategy = ec['strategy']
segments = ec['segments']

# Filter Test Period
test_indices = [i for i, s in enumerate(segments) if s == 'Test']
if not test_indices:
    print("No Test segment found!")
    exit()

test_vals = [strategy[i] for i in test_indices]
bench_vals = [ec['benchmark'][i] for i in test_indices]

start_val = test_vals[0]
end_val = test_vals[-1]
diff = end_val - start_val

bench_start = bench_vals[0]
bench_end = bench_vals[-1]
bench_diff = bench_end - bench_start

print(f"Test Segment Length: {len(test_vals)} days")
print(f"Strategy: Start={start_val}, End={end_val}, Diff={diff} ({'UP' if diff > 0 else 'DOWN'})")
print(f"Benchmark: Start={bench_start}, End={bench_end}, Diff={bench_diff} ({'UP' if bench_diff > 0 else 'DOWN'})")

metrics = data.get('metrics', {})
ann_ret = metrics.get('annualized_return', 'N/A')
print(f"Metrics Annualized Return: {ann_ret}")

if (diff > 0 and ann_ret > 0) or (diff < 0 and ann_ret < 0):
    print("CONSISTENT")
else:
    print("INCONSISTENT")
