import yaml
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import exists_qlib_data, flatten_dict
from qlib.tests.data import GetData

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_workflow():
    # 1. Load Configuration
    config = load_config()

    # 2. Initialize Qlib
    qlib.init(provider_uri=config['qlib_init']['provider_uri'], region=REG_US)
    print("Qlib initialized.")

    # 3. Initialize Data Handler & Dataset
    # We use a standard dataset config structure for Qlib's Workflow
    market = config['market']
    benchmark = config['benchmark']
    
    data_handler_config = config['data_handler_config']

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
                "valid": ["2017-01-01", "2017-12-31"],
                "test": [config['port_analysis_config']['backtest']['start_time'], config['port_analysis_config']['backtest']['end_time']],
            },
        },
    }

    # 4. Model Configuration (LightGBM)
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    }

    # 5. Start Workflow
    with R.start(experiment_name="trend_following_us"):
        print("Starting workflow...")
        
        # Train Model
        print("Training model...")
        model = init_instance_by_config(model_config)
        dataset = init_instance_by_config(dataset_config)
        model.fit(dataset)
        
        # Prediction
        print("Predicting...")
        recorder = R.get_recorder()
        pred_df = model.predict(dataset, segment="test")
        
        # Save Prediction
        R.save_objects(**{"pred.pkl": pred_df})
        
        # Backtest
        print("Backtesting...")
        port_analysis_config = config['port_analysis_config']
        
        # We need to manually set the signal for the strategy if it's not automatically picked up by some workflows.
        # But commonly we use PortAnaRecord for this.
        
        # Record Signals & Portfolio Analysis
        # The SignalRecord will save the prediction as a signal for the backtester
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # The PortAnaRecord runs the strategy backtest
        # We need to ensure the strategy config knows where to find the signal.
        # In Qlib, <PRED> usually refers to the prediction we just made.
        
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        print(f"Workflow finished. Report saved in: {recorder.get_local_dir()}")
        print("Check the 'analysis' folder in the recorder directory for performance metrics.")

if __name__ == "__main__":
    run_workflow()
