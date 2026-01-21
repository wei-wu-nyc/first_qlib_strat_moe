import yaml
import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
import lightgbm as lgb
import itertools

# --- COPY OF CUSTOM HANDLER logic ---
class CustomHandler(Alpha158):
    def get_feature_config(self):
        # Alpha158 returns a tuple: (fields, names)
        base_conf = super().get_feature_config()
        fields = list(base_conf[0])
        names = list(base_conf[1])
        
        # Add Distance from MA features
        new_fields = [
            "($close - Mean($close, 10)) / Mean($close, 10)",
            "($close - Mean($close, 20)) / Mean($close, 20)",
            "($close - Mean($close, 60)) / Mean($close, 60)",
        ]
        new_names = ["DIST_MA10", "DIST_MA20", "DIST_MA60"]
        
        fields.extend(new_fields)
        names.extend(new_names)
        
        return (fields, names)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_market_regime(benchmark, start_time, end_time):
    # Same regime logic as adaptive_strategy.py
    fields = ["$close", "Mean($close, 60)", "Mean($close, 20)"]
    names = ["close", "ma60", "ma20"]
    df = D.features([benchmark], fields, start_time=start_time, end_time=end_time)
    df.columns = names
    
    df['regime'] = 0 # Default Choppy
    uptrend_mask = (df['close'] > df['ma60']) & (df['ma20'] > df['ma60'])
    downtrend_mask = (df['close'] < df['ma60']) & (df['ma20'] < df['ma60'])
    
    df.loc[uptrend_mask, 'regime'] = 1
    df.loc[downtrend_mask, 'regime'] = -1
    
    df_reset = df.reset_index()
    if 'datetime' in df_reset.columns:
        df_reset = df_reset.set_index('datetime')
    
    regime = df_reset['regime']
    if isinstance(regime.index, pd.DatetimeIndex) or 'datetime' in regime.index.names:
         regime = regime[~regime.index.duplicated(keep='last')]
    return regime

def run_tuning():
    config = load_config()
    qlib.init(provider_uri=config['qlib_init']['provider_uri'], region=REG_US)
    benchmark = config['benchmark']
    data_handler_config = config['data_handler_config']
    
    # Choppy Dataset
    dataset_config_choppy = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "CustomHandler",
                "module_path": "adaptive_strategy", # We will monkeypatch or use this file as main
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": [data_handler_config['fit_start_time'], data_handler_config['fit_end_time']],
                "valid": ["2021-01-01", "2021-12-31"],
                "test": [config['port_analysis_config']['backtest']['start_time'], config['port_analysis_config']['backtest']['end_time']],
            },
        },
    }
    
    # HACK: Qlib requires module path for the class. 
    # Since we are running this script directly, we can pass the class object if we bypass config loader
    # OR we can just register it.
    # Easiest: Manually init.
    from qlib.data.dataset import DatasetH
    handler_choppy = CustomHandler(**data_handler_config)
    dataset_choppy = DatasetH(handler=handler_choppy, segments=dataset_config_choppy['kwargs']['segments'])
    
    print("Preparing Data...")
    train_df = dataset_choppy.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid_df = dataset_choppy.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    
    # Filter for Choppy Regime ONLY
    # Train Regime
    train_regime = get_market_regime(benchmark, data_handler_config['fit_start_time'], data_handler_config['fit_end_time'])
    train_choppy_dates = train_regime[train_regime == 0].index
    
    # Valid Regime
    valid_regime = get_market_regime(benchmark, "2021-01-01", "2021-12-31")
    valid_choppy_dates = valid_regime[valid_regime == 0].index
    
    # Subset Data
    x_train = train_df.loc[train_df.index.get_level_values('datetime').isin(train_choppy_dates), 'feature']
    y_train = train_df.loc[train_df.index.get_level_values('datetime').isin(train_choppy_dates), 'label']
    
    x_valid = valid_df.loc[valid_df.index.get_level_values('datetime').isin(valid_choppy_dates), 'feature']
    y_valid = valid_df.loc[valid_df.index.get_level_values('datetime').isin(valid_choppy_dates), 'label']
    
    print(f"Train Samples (Choppy): {len(x_train)}")
    print(f"Valid Samples (Choppy): {len(x_valid)}")
    
    # Grid Search
    # Reduce search space for speed
    param_grid = {
        'max_depth': [3, 4, 5, 8],
        'num_leaves': [8, 16, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'lambda_l1': [0.1, 1.0, 10.0, 200.0],
        'min_data_in_leaf': [20, 100, 500] 
    }
    
    # Simplified Grid (Top Candidates to save time)
    # Focus on depth (shallow) vs regularization
    grid = [
        {'max_depth': 3, 'num_leaves': 7},
        {'max_depth': 4, 'num_leaves': 15},
        {'max_depth': 5, 'num_leaves': 31},
        {'max_depth': 6, 'num_leaves': 63},
        {'max_depth': 8, 'num_leaves': 210}, # Baselineish
    ]
    
    # We will iterate over these base configs and vary regularization slightly
    keys = ['max_depth', 'num_leaves']
    
    best_score = -999
    best_params = {}
    
    print("\nStarting Grid Search...")
    print(f"{'Depth':<6} {'Leaves':<8} {'L1':<8} {'MSE':<10} {'IC':<10}")
    
    base_lgb_params = {
        "objective": "regression",
        "metric": "mse",
        "colsample_bytree": 0.8879,
        "subsample": 0.8789,
        "lambda_l2": 580.9768, # Keep L2 high from existing
        "n_jobs": 4, # Parallelize
        "verbosity": -1,
        "n_estimators": 500, # Shorter for tuning
        "seed": 42,
        "deterministic": True
    }
    
    for g in grid:
        for l1 in [10.0, 100.0, 205.7]: # Try varying L1
             params = base_lgb_params.copy()
             params.update(g)
             params['lambda_l1'] = l1
             params['learning_rate'] = 0.05
             
             model = lgb.LGBMRegressor(**params)
             
             model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="mse", 
                       callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
             
             preds = model.predict(x_valid)
             
             # Calculate IC (Correlation between Pred and TRUTH)
             # Valid Label is 'Ref($close, -1)/$close - 1'
             # preds is predicted return.
             # IC = Corr(Preds, Truth)
             
             df_res = pd.DataFrame({'pred': preds, 'label': y_valid.iloc[:,0] if isinstance(y_valid, pd.DataFrame) else y_valid})
             ic = df_res.corr().iloc[0, 1]
             mse = ((df_res['pred'] - df_res['label'])**2).mean()
             
             print(f"{params['max_depth']:<6} {params['num_leaves']:<8} {l1:<8} {mse:.6f}   {ic:.6f}")
             
             # Metric: Maximize IC
             if ic > best_score:
                 best_score = ic
                 best_params = params
                 
    print("\nBest Parameters found:")
    print(best_params)
    print(f"Best IC: {best_score}")

if __name__ == "__main__":
    run_tuning()
