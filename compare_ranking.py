"""
Compare different position ranking methods for the fund backtester.

Tests 4 ranking approaches on the 13 report stocks using Strategy 1:
  A. jojo值降序 (highest momentum first)
  B. 市值降序 (largest market cap first)
  C. ATR%降序 (highest volatility first)
  D. 历史盈亏比降序 (best historical profit factor first)

Usage:
    python compare_ranking.py
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_jojo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

START_DATE = "2015-01-01"
INITIAL_CAPITAL = 1_000_000
STOP_LOSS_PCT = 20.0
MIN_ATR_PCT = 2.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    strategy: int

@dataclass
class FundTrade:
    ticker: str
    entry_date: str
    exit_date: str
    pnl_pct: float
    holding_days: int
    exit_reason: str


def get_sp500_tickers():
    """Fetch current S&P 500 constituents from Wikipedia."""
    import requests
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        return sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as e:
        print(f"[WARN] Could not fetch S&P 500 list: {e}")
        return []


def download_data(tickers, start):
    """Download OHLC for all tickers."""
    print(f"Downloading {len(tickers)} tickers from {start}...")
    raw = yf.download(tickers, start=start, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in tickers:
            try:
                sub = raw[sym].dropna(how="all")
                sub.columns = [c.lower() for c in sub.columns]
                if len(sub) >= 60 and all(c in sub.columns for c in ["open", "high", "low", "close"]):
                    data[sym] = sub[["open", "high", "low", "close"]].copy()
            except Exception:
                pass
    print(f"  Got data for {len(data)} tickers.")
    return data


def precompute(data):
    """Pre-compute jojo values and ATR% for all tickers."""
    jojo_cache = {}
    atr_cache = {}
    for sym, df in data.items():
        jojo_cache[sym] = compute_jojo(df)
        # ATR%(14)
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = pd.Series(tr, index=df.index).rolling(14).mean()
        atr_pct = atr / df["close"].astype(float) * 100
        atr_cache[sym] = atr_pct
    return jojo_cache, atr_cache


def compute_historical_pf(data, jojo_cache):
    """Pre-compute historical profit factor for each ticker using Strategy 1."""
    pf_map = {}
    for sym, df in data.items():
        jvals = jojo_cache[sym].values
        closes = df["close"].astype(float).values
        trades_pnl = []
        in_pos = False
        entry_price = 0
        for i in range(1, len(jvals)):
            if np.isnan(jvals[i]) or np.isnan(jvals[i-1]):
                continue
            if not in_pos:
                if jvals[i] > 76 and jvals[i-1] <= 76:
                    in_pos = True
                    entry_price = closes[i]
            else:
                if jvals[i] < 68 and jvals[i-1] >= 68:
                    pnl = (closes[i] / entry_price - 1) * 100
                    trades_pnl.append(pnl)
                    in_pos = False
        gross_p = sum(p for p in trades_pnl if p > 0)
        gross_l = abs(sum(p for p in trades_pnl if p < 0))
        pf_map[sym] = gross_p / gross_l if gross_l > 0 else (10.0 if gross_p > 0 else 0)
    return pf_map


# Approximate market caps (billions, as of early 2026)
MARKET_CAPS = {
    "AAPL": 3500, "MSFT": 3200, "NVDA": 2800, "AMZN": 2200, "GOOG": 2100,
    "META": 1800, "TSM": 900, "TSLA": 800, "ANET": 130, "PLTR": 120,
    "MU": 100, "HOOD": 30, "RKLB": 10,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_fund(data, jojo_cache, atr_cache, rank_fn, label):
    """Run fund simulation with a given ranking function.

    rank_fn(candidates, date) -> sorted list of (ticker, jojo_val, atr_val)
    """
    # Build common date index
    all_dates = sorted(set().union(*(df.index for df in data.values())))

    cash = INITIAL_CAPITAL
    positions = {}  # ticker -> Position
    trades = []
    snapshots = []  # (date, equity)

    for di in range(1, len(all_dates)):
        date = all_dates[di]
        prev_date = all_dates[di - 1]

        # --- 1. Check stop losses ---
        to_close = []
        for sym, pos in positions.items():
            if sym not in data or date not in data[sym].index:
                continue
            price = float(data[sym].loc[date, "close"])
            pnl = (price / pos.entry_price - 1) * 100
            if pnl <= -STOP_LOSS_PCT:
                to_close.append((sym, price, f"止损({STOP_LOSS_PCT}%)"))

        for sym, price, reason in to_close:
            pos = positions.pop(sym)
            cash += pos.shares * price
            days = (date - pos.entry_date).days
            pnl = (price / pos.entry_price - 1) * 100
            trades.append(FundTrade(sym, str(pos.entry_date)[:10], str(date)[:10], round(pnl, 2), days, reason))

        # --- 2. Check sell signals ---
        to_sell = []
        for sym, pos in positions.items():
            if sym not in jojo_cache:
                continue
            jojo = jojo_cache[sym]
            if date not in jojo.index or prev_date not in jojo.index:
                continue
            j_today = jojo.loc[date]
            j_yest = jojo.loc[prev_date]
            if np.isnan(j_today) or np.isnan(j_yest):
                continue
            # S1 sell: cross below 68
            if j_today < 68 and j_yest >= 68:
                price = float(data[sym].loc[date, "close"])
                to_sell.append((sym, price, "下穿68"))

        for sym, price, reason in to_sell:
            pos = positions.pop(sym)
            cash += pos.shares * price
            days = (date - pos.entry_date).days
            pnl = (price / pos.entry_price - 1) * 100
            trades.append(FundTrade(sym, str(pos.entry_date)[:10], str(date)[:10], round(pnl, 2), days, reason))

        # --- 3. Check buy signals ---
        candidates = []
        for sym in data:
            if sym in positions:
                continue
            if sym not in jojo_cache or sym not in atr_cache:
                continue
            jojo = jojo_cache[sym]
            atr = atr_cache[sym]
            if date not in jojo.index or prev_date not in jojo.index:
                continue
            j_today = jojo.loc[date]
            j_yest = jojo.loc[prev_date]
            if np.isnan(j_today) or np.isnan(j_yest):
                continue
            # S1 buy: cross above 76 + ATR% filter
            if j_today > 76 and j_yest <= 76:
                atr_val = atr.loc[date] if date in atr.index else 0
                if np.isnan(atr_val) or atr_val < MIN_ATR_PCT:
                    continue
                candidates.append((sym, float(j_today), float(atr_val)))

        # --- 4. Rank and allocate ---
        if candidates and len(positions) < MAX_POSITIONS:
            ranked = rank_fn(candidates, date)
            available_slots = MAX_POSITIONS - len(positions)
            equity = cash + sum(
                pos.shares * float(data[s].loc[date, "close"])
                for s, pos in positions.items()
                if date in data[s].index
            )
            size_per_pos = equity / MAX_POSITIONS

            for sym, j_val, atr_val in ranked[:available_slots]:
                if cash < size_per_pos * 0.5:  # need at least half a position
                    break
                alloc = min(cash, size_per_pos)
                price = float(data[sym].loc[date, "close"])
                shares = int(alloc / price)
                if shares <= 0:
                    continue
                cost = shares * price
                cash -= cost
                positions[sym] = Position(sym, date, price, shares, 1)

        # --- 5. Daily snapshot ---
        mkt_value = sum(
            pos.shares * float(data[s].loc[date, "close"])
            for s, pos in positions.items()
            if date in data[s].index
        )
        snapshots.append((date, cash + mkt_value))

    # Close remaining positions
    last_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        if last_date in data[sym].index:
            price = float(data[sym].loc[last_date, "close"])
            days = (last_date - pos.entry_date).days
            pnl = (price / pos.entry_price - 1) * 100
            trades.append(FundTrade(sym, str(pos.entry_date)[:10], str(last_date)[:10], round(pnl, 2), days, "持仓中"))

    return compute_fund_metrics(snapshots, trades, label)


def compute_fund_metrics(snapshots, trades, label):
    """Compute fund-level performance metrics."""
    dates = [s[0] for s in snapshots]
    equity = np.array([s[1] for s in snapshots])

    if len(equity) < 2:
        return {"label": label, "total_return": 0, "ann_return": 0, "max_dd": 0,
                "sharpe": 0, "sortino": 0, "trades": 0, "win_rate": 0, "pf": 0}

    total_ret = (equity[-1] / equity[0] - 1) * 100
    days = (dates[-1] - dates[0]).days
    ann_ret = ((equity[-1] / equity[0]) ** (365.25 / days) - 1) * 100 if days > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = np.max(dd)

    # Daily returns
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    sortino = np.mean(daily_ret) / np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 0 and np.std(neg_ret) > 0 else 0

    # Trade metrics
    pnls = [t.pnl_pct for t in trades]
    wins = sum(1 for p in pnls if p > 0) if pnls else 0
    win_rate = wins / len(pnls) * 100 if pnls else 0
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_p / gross_l if gross_l > 0 else (float("inf") if gross_p > 0 else 0)

    return {
        "label": label,
        "total_return": round(total_ret, 1),
        "ann_return": round(ann_ret, 1),
        "max_dd": round(max_dd, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "pf": round(pf, 2),
        "final_equity": round(equity[-1], 0),
    }


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------

def rank_by_jojo(candidates, date):
    """A: Sort by jojo value descending (highest momentum first)."""
    return sorted(candidates, key=lambda x: x[1], reverse=True)

def rank_by_market_cap(candidates, date):
    """B: Sort by market cap descending (largest first)."""
    return sorted(candidates, key=lambda x: MARKET_CAPS.get(x[0], 0), reverse=True)

def rank_by_atr(candidates, date):
    """C: Sort by ATR% descending (highest volatility first)."""
    return sorted(candidates, key=lambda x: x[2], reverse=True)

# D: rank by historical profit factor - set up in main()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if not tickers:
        print("Failed to get S&P 500 list!")
        sys.exit(1)
    print(f"  Got {len(tickers)} tickers.")

    data = download_data(tickers, START_DATE)
    if not data:
        print("No data downloaded!")
        sys.exit(1)

    print("Pre-computing jojo values and ATR%...")
    jojo_cache, atr_cache = precompute(data)

    print("Computing historical profit factors...")
    pf_map = compute_historical_pf(data, jojo_cache)
    top10 = sorted(pf_map.items(), key=lambda x: -x[1])[:10]
    print("  Top 10 PF:", {k: round(v, 2) for k, v in top10})

    # Fetch market caps from yfinance (approximate, use last close * shares)
    # Use a simpler proxy: just use close price rank as tie-breaker
    # For proper market cap, we'd need FMP, but for ranking comparison this is fine
    # We'll use the MARKET_CAPS dict for known ones, 0 for others
    # Actually let's compute a rough proxy: average daily volume * close
    mktcap_proxy = {}
    for sym, df in data.items():
        mktcap_proxy[sym] = MARKET_CAPS.get(sym, 50)  # default 50B for unknown S&P500

    def rank_by_pf(candidates, date):
        """D: Sort by historical profit factor descending."""
        return sorted(candidates, key=lambda x: pf_map.get(x[0], 0), reverse=True)

    def rank_by_mktcap(candidates, date):
        """B: Sort by market cap proxy descending."""
        return sorted(candidates, key=lambda x: mktcap_proxy.get(x[0], 0), reverse=True)

    methods = [
        (rank_by_jojo, "A: jojo值降序"),
        (rank_by_mktcap, "B: 市值降序"),
        (rank_by_atr, "C: ATR%降序"),
        (rank_by_pf, "D: 历史盈亏比降序"),
    ]

    for max_pos in [5, 10, 20]:
        global MAX_POSITIONS
        MAX_POSITIONS = max_pos

        results = []
        for fn, label in methods:
            print(f"\nRunning: {label} (max_pos={max_pos})...")
            r = run_fund(data, jojo_cache, atr_cache, fn, label)
            results.append(r)

        print("\n" + "=" * 115)
        print(f"排序方案对比 | 策略1 | S&P500 ({len(data)}只) | {START_DATE}~今 | 初始${INITIAL_CAPITAL:,} | 最大仓位 {max_pos}")
        print("=" * 115)
        header = f"{'方案':<22s} {'总收益%':>8s} {'年化%':>7s} {'最大回撤%':>9s} {'Sharpe':>7s} {'Sortino':>8s} {'交易数':>6s} {'胜率%':>6s} {'盈亏比':>7s} {'终值':>12s}"
        print(header)
        print("-" * 115)
        for r in results:
            print(f"{r['label']:<22s} {r['total_return']:>8.1f} {r['ann_return']:>7.1f} {r['max_dd']:>9.1f} "
                  f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['trades']:>6d} {r['win_rate']:>6.1f} "
                  f"{r['pf']:>7.2f} {r['final_equity']:>12,.0f}")
        print("=" * 115)


if __name__ == "__main__":
    main()
