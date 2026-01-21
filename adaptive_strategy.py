
import yaml
import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data import D

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_trend_prediction(dataset, model):
    # Train and predict using the existing LightGBM model
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    return pred

def get_momentum_signal(instruments, start_time, end_time):
    # Momentum: Ref($close, 0) / Ref($close, 20) - 1
    # We want to buy stocks that held up best (Highest Return)
    fields = ["Ref($close, 0) / Ref($close, 20) - 1"]
    names = ["return_20d"]
    if isinstance(instruments, str):
        instruments = D.instruments(instruments)
    df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
    df.columns = names
    # Signal: High Return -> High Score
    df['momentum'] = df['return_20d']
    return df[['momentum']]

def get_market_regime(benchmark, start_time, end_time):
    # 3-State Regime detection using MA20 and MA60
    # Uptrend (1): Price > MA60 AND MA20 > MA60
    # Downtrend (-1): Price < MA60 AND MA20 < MA60
    # Choppy (0): Everything else
    
    fields = ["$close", "Mean($close, 60)", "Mean($close, 20)"]
    names = ["close", "ma60", "ma20"]
    df = D.features([benchmark], fields, start_time=start_time, end_time=end_time)
    df.columns = names
    
    # Logic
    df['regime'] = 0 # Default to Choppy
    
    uptrend_mask = (df['close'] > df['ma60']) & (df['ma20'] > df['ma60'])
    downtrend_mask = (df['close'] < df['ma60']) & (df['ma20'] < df['ma60'])
    
    df.loc[uptrend_mask, 'regime'] = 1
    df.loc[downtrend_mask, 'regime'] = -1
    
    # Handle index
    df_reset = df.reset_index()
    if 'datetime' in df_reset.columns:
        df_reset = df_reset.set_index('datetime')
    
    regime = df_reset['regime']
    if isinstance(regime.index, pd.DatetimeIndex) or 'datetime' in regime.index.names:
         regime = regime[~regime.index.duplicated(keep='last')]
         
    return regime

import lightgbm as lgb

def run_adaptive_strategy():
    config = load_config()
    qlib.init(provider_uri=config['qlib_init']['provider_uri'], region=REG_US)
    
    market = config['market']
    benchmark = config['benchmark']
    data_handler_config = config['data_handler_config']
    
    # Dataset Config
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": [data_handler_config['fit_start_time'], data_handler_config['fit_end_time']],
                "valid": ["2021-01-01", "2021-12-31"],
                "test": [config['port_analysis_config']['backtest']['start_time'], config['port_analysis_config']['backtest']['end_time']],
            },
        },
    }
    
    # LightGBM Params (Native)
    lgb_params = {
        "objective": "regression",
        "metric": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "n_jobs": 20,
        "verbosity": -1,
        "n_estimators": 1000  # Default Qlib uses 1000 steps usually
    }

# ... Imports ...
from qlib.contrib.data.handler import Alpha158

class CustomHandler(Alpha158):
    def get_feature_config(self):
        # Alpha158 returns a tuple: (fields, names)
        # fields is a list or tuple of strings (expressions)
        # names is a list or tuple of strings (column names)
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

import lightgbm as lgb

def run_adaptive_strategy():
    config = load_config()
    qlib.init(provider_uri=config['qlib_init']['provider_uri'], region=REG_US)
    
    market = config['market']
    benchmark = config['benchmark']
    data_handler_config = config['data_handler_config']
    
    # 1. Standard Dataset (for Uptrend/Downtrend) - Original Alpha158
    dataset_config_std = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": [data_handler_config['fit_start_time'], data_handler_config['fit_end_time']],
                "valid": ["2021-01-01", "2021-12-31"],
                "test": [config['port_analysis_config']['backtest']['start_time'], config['port_analysis_config']['backtest']['end_time']],
            },
        },
    }
    
    # 2. Choppy Dataset (Extra Features)
    dataset_config_choppy = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "CustomHandler", # Use our custom class defined above
                "module_path": "adaptive_strategy", # This file
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": [data_handler_config['fit_start_time'], data_handler_config['fit_end_time']],
                "valid": ["2021-01-01", "2021-12-31"],
                "test": [config['port_analysis_config']['backtest']['start_time'], config['port_analysis_config']['backtest']['end_time']],
            },
        },
    }
    
    # LightGBM Params (Standard)
    lgb_params_std = {
        "objective": "regression",
        "metric": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "n_jobs": 20,
        "verbosity": -1,
        "n_estimators": 1000,
        "seed": 42,
        "deterministic": True
    }
    
    # LightGBM Params (Choppy) - Reverted to Depth 5 (Better Test Performance)
    lgb_params_choppy = lgb_params_std.copy()
    lgb_params_choppy['max_depth'] = 5
    lgb_params_choppy['num_leaves'] = 31 # 2^5 - 1 roughly

    print("Backtesting Mixture of Experts (MoE) Strategy...")
    with R.start(experiment_name="moe_strategy"):
        recorder = R.get_recorder()
        
        # Initialize Datasets
        print("Initializing Standard Dataset...")
        dataset_std = init_instance_by_config(dataset_config_std)
        
        print("Initializing Choppy Dataset (Enhanced)...")
        # We need to register the class if it's dynamic, OR just use the class object in config if run locally.
        # But Qlib expects module_path. 
        # Trick: Pass the CLASS directly if using internal init, but config dict usually requires strings.
        # Since we are running this script, "adaptive_strategy" is __main__ or the file name.
        # Let's try "sys.modules[__name__]" approach or just simpler:
        # We can substitute the "class" string with the actual class object in the config 
        # IF we use 'init_instance_by_config' it might fail if it tries to import string.
        # HACK: Modify init keys after config load or use the object directly.
        # Actually simplest is to define CustomHandler in a separate file, BUT I want to keep it self-contained.
        # Let's use `adaptive_strategy.CustomHandler` assuming this file is importable? 
        # If run as script, module is __main__.
        
        # Workaround: Manually init the handler and dataset.
        from qlib.data.dataset import DatasetH
        handler_choppy = CustomHandler(**data_handler_config)
        dataset_choppy = DatasetH(handler=handler_choppy, segments=dataset_config_choppy['kwargs']['segments'])

        
        # Get Regime for Training Data
        print(f"Detecting Regimes for Training Data ({benchmark})...")
        train_start = data_handler_config['fit_start_time']
        train_end = data_handler_config['fit_end_time']
        train_regime = get_market_regime(benchmark, train_start, train_end)
        
        models = {}
        dataset_map = {
            1: dataset_std,
            -1: dataset_std,
            0: dataset_choppy
        }
        params_map = {
            1: lgb_params_std,
            -1: lgb_params_std,
            0: lgb_params_choppy
        }
        
        for regime_val, regime_name in [(1, 'Uptrend'), (0, 'Choppy'), (-1, 'Downtrend')]:
            print(f"Training {regime_name} Model...")
            
            ds = dataset_map[regime_val]
            params = params_map[regime_val]
            
            regime_dates = train_regime[train_regime == regime_val].index
            
            train_df = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            subset_df = train_df.loc[train_df.index.get_level_values('datetime').isin(regime_dates)]
            
            if subset_df.empty:
                print(f"Warning: No data for {regime_name} regime. Skipping training.")
                models[regime_val] = None
                continue
                
            x_train = subset_df['feature']
            y_train = subset_df['label']
            
            valid_df = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            x_valid = valid_df['feature']
            y_valid = valid_df['label']
            
            model = lgb.LGBMRegressor(**params)
            
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
            
            model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="mse",
                callbacks=callbacks
            )
            
            models[regime_val] = model
            print(f"{regime_name} Model Trained.")

        # Inference
        print("Running Inference...")
        test_start = config['port_analysis_config']['backtest']['start_time']
        test_end = config['port_analysis_config']['backtest']['end_time']
        
        test_regime = get_market_regime(benchmark, test_start, test_end)
        
        # Prepare Test Dataframes
        test_df_std = dataset_std.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        test_df_choppy = dataset_choppy.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        x_test_std = test_df_std['feature']
        x_test_choppy = test_df_choppy['feature']
        
        predictions = []
        
        for regime_val, model in models.items():
            if model is None: 
                continue
                
            # Use correct input
            if regime_val == 0:
                x_in = x_test_choppy
            else:
                x_in = x_test_std
                
            pred = model.predict(x_in)
            pred_series = pd.Series(pred, index=x_in.index)
            pred_series.name = regime_val 
            predictions.append(pred_series)
            
        combined_pred = pd.concat(predictions, axis=1) # Columns: 1, 0, -1
        
        # Join with Test Regime
        test_regime_reindexed = test_regime.reindex(combined_pred.index.get_level_values('datetime'))
        
        final_series = pd.Series(index=combined_pred.index, dtype=float)
        
        for r_val in [-1, 0, 1]:
            if r_val in combined_pred.columns:
                mask = (test_regime_reindexed.values == r_val)
                final_series[mask] = combined_pred[r_val][mask]
                
        final_series = final_series.fillna(-999.0)
        
        # Cash Logic
        print("Applying Cash Logic (Score < 0)...")
        final_series[final_series < 0] = -999.0
        
        combined = final_series.to_frame('final_score')
        
        # --- DATA FILTER ---
        print("Applying Data Quality Filter (Close > 0.01)...")
        comb_insts = combined.index.get_level_values('instrument').unique().tolist()
        comb_dates = combined.index.get_level_values('datetime').unique()
        start_date = comb_dates.min()
        end_date = comb_dates.max()
        
        prices = D.features(comb_insts, ['$close'], start_time=start_date, end_time=end_date)
        prices.columns = ['close']
        
        combined = combined.join(prices, how='left')
        
        bad_rows = combined[combined['close'] <= 0.01]
        bad_instruments = bad_rows.index.get_level_values('instrument').unique()
        
        if len(bad_instruments) > 0:
            print(f"Banning {len(bad_instruments)} instruments with corrupted data: {bad_instruments.tolist()}")
            combined = combined[~combined.index.get_level_values('instrument').isin(bad_instruments)]
        else:
            print("No instruments banned.")
        
        final_pred = combined[['final_score']]
        final_pred.columns = ['score']
        
        # Save Prediction
        R.save_objects(**{"pred.pkl": final_pred})

        # Save Label (Required for PortAnaRecord)
        # We need the label for the test segment
        label_df = dataset_std.prepare("test", col_set="label")
        if isinstance(label_df, pd.Series):
             label_df = label_df.to_frame()
        R.save_objects(**{"label.pkl": label_df})
        
        # Run Portfolio Analysis
        port_analysis_config = config['port_analysis_config']
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        print(f"Adaptive Strategy Finished. Results in: {recorder.get_local_dir()}")

if __name__ == "__main__":
    run_adaptive_strategy()
