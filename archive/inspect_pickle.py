
import pandas as pd
import pickle
import os
import glob

def load_latest_artifacts(base_path):
    runs = glob.glob(os.path.join(base_path, "mlruns", "*", "*", "artifacts"))
    if not runs:
        raise FileNotFoundError("No artifacts found")
    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

base_path = r"D:\Work\Antigravity\qlib_strategy_test"
artifact_path = load_latest_artifacts(base_path)
port_analysis_path = os.path.join(artifact_path, "portfolio_analysis", "port_analysis_1day.pkl")

with open(port_analysis_path, "rb") as f:
    port_analysis = pickle.load(f)
    print("Keys:", port_analysis.keys())
    if 'risk' in port_analysis:
        print("Risk Type:", type(port_analysis['risk']))
        print("Risk Index:", port_analysis['risk'].index)
        print("Risk Columns:", port_analysis['risk'].columns)
        print("Risk Head:\n", port_analysis['risk'].head())
