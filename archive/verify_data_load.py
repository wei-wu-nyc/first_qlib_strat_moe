import yaml
import qlib
from qlib.data import D
from qlib.config import REG_US

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize Qlib with the data provider
provider_uri = config['qlib_init']['provider_uri']
print(f"Initializing Qlib with data from: {provider_uri}")
qlib.init(provider_uri=provider_uri, region=REG_US)

# Try fetching calendar
print("Loading calendar...")
dates = D.calendar(start_time='2020-01-01', end_time='2020-01-10')
print(f"Calendar dates: {dates}")

# Try fetching features for a sample instrument
print("Loading features...")
instruments = D.instruments(market='all')
print(f"Instruments loaded.")
