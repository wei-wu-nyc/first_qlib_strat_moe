import qlib
from qlib.constant import REG_US
from qlib.contrib.data.handler import Alpha158

# Initialize Qlib (Minimal config)
qlib.init(provider_uri=r"D:\Work\TradeStationData\Auto_Export_Data\20251231\qlib_antigravity\day\us_stock", region=REG_US)

dh = Alpha158(instruments=['QQQ'], start_time='2020-01-01', end_time='2020-01-10')
print("Features in Alpha158:")
print(dh.get_feature_config())
