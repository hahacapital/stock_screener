"""
Validate thermometer indicator against TradingView exported CSV data.

Strategy: Download full stock history from yfinance (to match TradingView's
warm-up), compute our indicator, then compare against the TV-exported values.

Usage:
    python validate.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from indicators import compute_thermometer

TV_FILES = {
    "HOOD": "/tmp/BATS_HOOD, 1D_f4797.csv",
    "NVDA": "/tmp/BATS_NVDA, 1D_dc48d.csv",
}

WARMUP_SKIP = 50


def load_tv_csv(path: str) -> pd.DataFrame:
    """Load TradingView exported CSV."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date
    return df


def validate_symbol(symbol: str, tv_path: str):
    """Validate thermometer calculation for one symbol."""
    print(f"\n{'='*60}")
    print(f"Validating: {symbol}")
    print(f"{'='*60}")

    # Load TV data
    tv_df = load_tv_csv(tv_path)
    tv_dates = tv_df["date"].values
    tv_values = tv_df["温度计"].values

    # Download full history from yfinance
    yf_data = yf.download(symbol, start="2000-01-01", end="2026-03-26", progress=False)
    if yf_data.empty:
        print(f"  No yfinance data for {symbol}")
        return None

    # Flatten multi-level columns from yfinance
    if isinstance(yf_data.columns, pd.MultiIndex):
        yf_data.columns = [c[0].lower() for c in yf_data.columns]
    else:
        yf_data.columns = [c.lower() for c in yf_data.columns]

    print(f"  yfinance data: {len(yf_data)} rows, {yf_data.index[0].date()} to {yf_data.index[-1].date()}")

    # Compute thermometer on full history
    yf_data = yf_data.reset_index()
    yf_data["date_key"] = yf_data["date" if "date" in yf_data.columns else "Date"].dt.date
    computed = compute_thermometer(yf_data)

    # Build lookup: date -> computed value
    computed_map = {}
    for i, row in yf_data.iterrows():
        computed_map[row["date_key"]] = computed.iloc[i]

    # Compare against TV
    matched = []
    for i in range(len(tv_dates)):
        d = tv_dates[i]
        if isinstance(d, np.datetime64):
            d = pd.Timestamp(d).date()
        if d in computed_map and not np.isnan(computed_map[d]) and not np.isnan(tv_values[i]):
            matched.append((d, computed_map[d], tv_values[i]))

    if len(matched) <= WARMUP_SKIP:
        print(f"  Not enough matched rows ({len(matched)})")
        return None

    matched = matched[WARMUP_SKIP:]
    dates = [m[0] for m in matched]
    ours = np.array([m[1] for m in matched])
    tvs = np.array([m[2] for m in matched])

    diff = np.abs(ours - tvs)
    mae = np.mean(diff)
    max_err = np.max(diff)
    corr = np.corrcoef(ours, tvs)[0, 1]

    print(f"  Rows compared: {len(matched)}")
    print(f"  MAE (平均绝对误差): {mae:.4f}")
    print(f"  Max error (最大误差): {max_err:.4f}")
    print(f"  Correlation (相关系数): {corr:.6f}")

    worst_idx = np.argsort(diff)[-5:][::-1]
    print(f"\n  Worst 5 rows:")
    print(f"  {'Date':>12s}  {'Ours':>8s}  {'TV':>8s}  {'Diff':>8s}")
    for i in worst_idx:
        print(f"  {dates[i]}  {ours[i]:>8.2f}  {tvs[i]:>8.2f}  {diff[i]:>8.4f}")

    if mae < 0.5:
        print(f"\n  PASS — MAE {mae:.4f} < 0.5")
    else:
        print(f"\n  FAIL — MAE {mae:.4f} >= 0.5")

    return mae


if __name__ == "__main__":
    results = {}
    for symbol, path in TV_FILES.items():
        results[symbol] = validate_symbol(symbol, path)

    print(f"\n{'='*60}")
    print("Summary:")
    for symbol, mae in results.items():
        status = "PASS" if mae is not None and mae < 0.5 else "FAIL"
        mae_str = f"{mae:.4f}" if mae is not None else "N/A"
        print(f"  {symbol}: MAE={mae_str} [{status}]")
