
import qlib
from qlib.data import D
from qlib.constant import REG_US
from qlib.data import D
from qlib.constant import REG_US

# Initialize Qlib
provider_uri = "D:\\work\\qlib\\qlib_data\\us_data"
qlib.init(provider_uri=provider_uri, region=REG_US)

# Check calendar
calendar = D.calendar(start_time='2020-01-01')
print(f"Data available from {calendar[0]} to {calendar[-1]}")
print(f"Total trading days since 2020: {len(calendar)}")
