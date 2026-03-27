"""Debug: output each sub-indicator to find which one diverges from TradingView."""

import pandas as pd
import numpy as np
from indicators import _rsi, _willr, _cmo, _stoch, _tsi, _dmi_adx, _ema, _rma

def debug_thermometer(df: pd.DataFrame, length: int = 14):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    value_rsi = _rsi(close, length)
    value_wr = _willr(high, low, close, length) + 100
    value_cmo = (_cmo(close, length) + 100) / 2
    value_kd = _stoch(close, high, low, length)

    short_len = round(length / 2)
    tsi_raw = _tsi(close, short_len, length)
    value_tsi = (tsi_raw + 1) / 2 * 100

    value_adx = _dmi_adx(high, low, close, di_length=length, adx_length=length + 4)

    rsi_of_adx = _rsi(value_adx, 14)
    sign = np.sign(close.values - open_.values)
    adxrsi = (rsi_of_adx * sign + 100) / 2

    index_raw = (
        value_rsi * 0.1
        + value_wr * 0.2
        + value_cmo * 0.1
        + value_kd * 0.3
        + value_tsi * 0.2
        + adxrsi * 0.1
    )
    index = _ema(index_raw, 3)

    return pd.DataFrame({
        "rsi": value_rsi,
        "wr": value_wr,
        "cmo": value_cmo,
        "kd": value_kd,
        "tsi": value_tsi,
        "adx": value_adx,
        "rsi_of_adx": rsi_of_adx,
        "adxrsi": adxrsi,
        "index_raw": index_raw,
        "index": index,
    })


# Load HOOD TV CSV and check the extra columns
df = pd.read_csv("/tmp/BATS_HOOD, 1D_f4797.csv")
df.columns = [c.strip().lower() for c in df.columns]

# The TV CSV has extra Plot columns - let's see what they contain
print("Columns:", list(df.columns))
print("\nFirst 5 rows of all columns:")
print(df.head().to_string())

# Check if the TV CSV has individual indicator values in the Plot columns
# The header shows: time,open,high,low,close,温度计,Plot,Shapes,Shapes,Plot,Plot,Plot,Plot,Plot,Plot,ADX
# Let's rename and examine
print("\n\nColumn values at row 50:")
for i, col in enumerate(df.columns):
    print(f"  col[{i}] '{col}': {df.iloc[50, i]}")

# Now compute our indicators
debug = debug_thermometer(df)
print("\n\nOur sub-indicators at row 50:")
for col in debug.columns:
    print(f"  {col}: {debug[col].iloc[50]:.4f}")

print(f"\n  TV 温度计: {df['温度计'].iloc[50]:.4f}")
print(f"  Our index: {debug['index'].iloc[50]:.4f}")

# Check if there's an ADX column in TV data to compare
if 'adx' in df.columns:
    print(f"\n  TV ADX: {df['adx'].iloc[50]:.4f}")
    print(f"  Our ADX: {debug['adx'].iloc[50]:.4f}")
