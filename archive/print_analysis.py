
import pandas as pd
import pickle
import os

# Path to the specific experiment run artifacts
artifact_path = r"D:\Work\Antigravity\qlib_strategy_test\mlruns\643615296333955938\f94530274c3b4869963781a2dc27eb92\artifacts"
port_analysis_path = os.path.join(artifact_path, "portfolio_analysis", "port_analysis_1day.pkl")
indicator_path = os.path.join(artifact_path, "portfolio_analysis", "indicator_analysis_1day.pkl")

print("--- Portfolio Analysis ---")
try:
    with open(port_analysis_path, "rb") as f:
        port_analysis = pickle.load(f)
    print(port_analysis)
except Exception as e:
    print(f"Error loading port analysis: {e}")

print("\n--- Indicator Analysis ---")
try:
    with open(indicator_path, "rb") as f:
        indicator_analysis = pickle.load(f)
    print(indicator_analysis)
except Exception as e:
    print(f"Error loading indicator analysis: {e}")
