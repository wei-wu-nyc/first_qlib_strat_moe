
import qlib
from qlib.data import D
from qlib.constant import REG_US

# Initialize Qlib with NEW data source
provider_uri = "d:\\Work\\TradeStationData\\Auto_Export_Data\\20251231\\qlib_antigravity_daily\\us_data"
qlib.init(provider_uri=provider_uri, region=REG_US)

# Check calendar
try:
    calendar = D.calendar(start_time='2000-01-01')
    print(f"Data available from {calendar[0]} to {calendar[-1]}")
    print(f"Total days: {len(calendar)}")
    
    # Check if 2023 exists
    cal_2023 = D.calendar(start_time='2023-01-01', end_time='2023-12-31')
    print(f"Data points in 2023: {len(cal_2023)}")
except Exception as e:
    print(f"Error checking calendar: {e}")
