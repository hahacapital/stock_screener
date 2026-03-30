"""
jojo_quant 基金回测器 — portfolio-level backtest using jojo signals.

Simulates a fund that:
1. Scans all stocks daily for jojo buy/sell signals
2. Ranks buy candidates by rolling profit factor (past 3 years)
3. Manages positions with stop loss, optional regime filter, optional vol sizing
4. Tracks portfolio equity curve and computes fund-level metrics

TODO: Add periodic rebalancing — check holdings on a weekly/monthly basis,
      replace positions whose PF ranking has dropped with higher-ranked candidates.

Usage:
    python fund_backtest.py --strategy 1 --universe sp500+
    python fund_backtest.py --strategy 1 --universe sp500+ --regime-filter --vol-sizing
    python fund_backtest.py --compare   # run all 4 configs and compare
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from indicators import compute_jojo, _rma
from screener import EXTRA_TICKERS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_TICKERS = ["TSM", "GOOG", "AAPL", "TSLA", "MSFT", "PLTR", "ANET",
                  "NVDA", "HOOD", "MU", "RKLB", "AMZN", "META"]


# ---------------------------------------------------------------------------
# Data classes
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
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl_pct: float
    pnl_dollar: float
    holding_days: int
    exit_reason: str


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        return sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as e:
        print(f"[WARN] Could not fetch S&P 500 list: {e}")
        return []


def get_sp500_historical() -> tuple[list[str], dict[str, set]]:
    """Fetch current S&P 500 + historical changes from Wikipedia.

    Returns:
        (all_ever_tickers, membership_by_date)
        - all_ever_tickers: all tickers that were ever in S&P 500 (for data download)
        - membership_by_date: dict mapping date_str -> set of member tickers
          We store membership snapshots at each change date.
          To look up membership at a given date, use the most recent snapshot <= that date.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
    except Exception as e:
        print(f"[WARN] Could not fetch S&P 500 data: {e}")
        return [], {}

    # Current members
    current = set(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())

    # Parse changes table (columns are MultiIndex)
    if len(tables) < 2:
        return sorted(current), {}

    changes = tables[1]
    # Flatten MultiIndex columns
    changes.columns = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]
    changes["added_ticker"] = changes["added_ticker"].str.replace(".", "-", regex=False)
    changes["removed_ticker"] = changes["removed_ticker"].str.replace(".", "-", regex=False)

    # Parse dates and sort chronologically (newest first in table)
    change_records = []
    for _, row in changes.iterrows():
        try:
            dt = pd.to_datetime(row["date"])
            added = str(row["added_ticker"]).strip() if pd.notna(row["added_ticker"]) else ""
            removed = str(row["removed_ticker"]).strip() if pd.notna(row["removed_ticker"]) else ""
            if added or removed:
                change_records.append((dt, added, removed))
        except Exception:
            continue

    # Sort oldest first
    change_records.sort(key=lambda x: x[0])

    # Reconstruct historical membership by replaying changes backward from current
    # Start with current set, undo changes from newest to oldest
    members = set(current)
    all_ever = set(current)

    # Build snapshots: walk backward in time
    snapshots = []  # [(date, frozenset_of_members)]
    snapshots.append((pd.Timestamp("2099-01-01"), frozenset(members)))  # sentinel

    for dt, added, removed in reversed(change_records):
        # Undo: if ticker was added on this date, remove it (it wasn't there before)
        if added and added in members:
            members.discard(added)
        # Undo: if ticker was removed on this date, add it back (it was there before)
        if removed:
            members.add(removed)
            all_ever.add(removed)
        snapshots.append((dt, frozenset(members)))

    snapshots.sort(key=lambda x: x[0])

    print(f"  S&P 500 historical: {len(change_records)} changes, "
          f"{len(all_ever)} unique tickers ever in index")

    return sorted(all_ever), snapshots


def _get_sp500_members_at(snapshots: list, date: pd.Timestamp) -> set:
    """Get S&P 500 members at a given date using pre-built snapshots."""
    if not snapshots:
        return set()
    # Binary search for latest snapshot <= date
    result = snapshots[0][1]
    for snap_date, members in snapshots:
        if snap_date <= date:
            result = members
        else:
            break
    return set(result)


def get_universe(name: str, custom_tickers: str = "", historical: bool = False):
    """Get ticker list for the specified universe.

    If historical=True and name starts with 'sp500', returns
    (all_ever_tickers, snapshots) for survivorship-bias-free backtesting.
    Otherwise returns a plain list of tickers.
    """
    if name in ("sp500", "sp500+"):
        if historical:
            all_ever, snapshots = get_sp500_historical()
            if not all_ever:
                print("  Historical fetch failed, falling back to current members.")
                tickers = get_sp500_tickers() or list(REPORT_TICKERS)
                return tickers, []
            if name == "sp500+":
                existing = set(all_ever)
                all_ever = all_ever + [t for t in EXTRA_TICKERS if t not in existing]
            return all_ever, snapshots
        else:
            tickers = get_sp500_tickers()
            if not tickers:
                print("Falling back to report tickers.")
                tickers = list(REPORT_TICKERS)
            if name == "sp500+":
                existing = set(tickers)
                tickers += [t for t in EXTRA_TICKERS if t not in existing]
            return tickers
    elif name == "report":
        return list(REPORT_TICKERS) if not historical else (list(REPORT_TICKERS), [])
    elif name == "custom":
        t = [t.strip() for t in custom_tickers.split(",") if t.strip()]
        return t if not historical else (t, [])
    else:
        return list(REPORT_TICKERS) if not historical else (list(REPORT_TICKERS), [])


# ---------------------------------------------------------------------------
# Data download with parquet caching
# ---------------------------------------------------------------------------

def download_data(tickers: list[str], start: str, cache_dir: str = "",
                  no_cache: bool = False) -> dict[str, pd.DataFrame]:
    """Download OHLC data, with optional parquet caching."""
    if cache_dir and not no_cache:
        os.makedirs(cache_dir, exist_ok=True)
        return _download_cached(tickers, start, cache_dir)
    return _download_fresh(tickers, start)


def _download_fresh(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Download all tickers from yfinance."""
    print(f"  Downloading {len(tickers)} tickers from {start}...")
    batch_size = 200
    data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        pct = min(100, (i + len(batch)) / len(tickers) * 100)
        print(f"\r  Batch {i // batch_size + 1} ({len(batch)} tickers, {pct:.0f}%)...",
              end="", flush=True)
        try:
            raw = yf.download(batch, start=start, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        sub = raw[sym].dropna(how="all")
                        sub.columns = [c.lower() for c in sub.columns]
                        if len(sub) >= 60 and all(c in sub.columns for c in ["open", "high", "low", "close"]):
                            data[sym] = sub[["open", "high", "low", "close"]].copy()
                    except Exception:
                        pass
            else:
                raw.columns = [c.lower() for c in raw.columns]
                if len(raw) >= 60 and all(c in raw.columns for c in ["open", "high", "low", "close"]):
                    data[batch[0]] = raw[["open", "high", "low", "close"]].dropna(how="all").copy()
        except Exception as e:
            print(f"\n  [WARN] Batch failed: {e}")
        time.sleep(0.3)
    print(f"\r  Downloaded {len(data)} tickers.                              ")
    return data


def _download_cached(tickers: list[str], start: str, cache_dir: str) -> dict[str, pd.DataFrame]:
    """Download with parquet caching."""
    data = {}
    to_download = []

    for sym in tickers:
        path = os.path.join(cache_dir, f"{sym}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if len(df) >= 60:
                    data[sym] = df
                    continue
            except Exception:
                pass
        to_download.append(sym)

    if data:
        print(f"  Loaded {len(data)} tickers from cache.")

    if to_download:
        fresh = _download_fresh(to_download, start)
        for sym, df in fresh.items():
            data[sym] = df
            path = os.path.join(cache_dir, f"{sym}.parquet")
            try:
                df.to_parquet(path)
            except Exception:
                pass
        print(f"  Cached {len(fresh)} new tickers to {cache_dir}/")

    return data


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def precompute_jojo(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Compute jojo indicator for all tickers."""
    cache = {}
    for sym, df in data.items():
        try:
            cache[sym] = compute_jojo(df)
        except Exception:
            pass
    return cache


def precompute_atr_pct(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Compute ATR%(14) for all tickers using Wilder's RMA."""
    cache = {}
    for sym, df in data.items():
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = _rma(pd.Series(tr, index=df.index), 14)
        cache[sym] = atr / df["close"].astype(float) * 100
    return cache


def fetch_shares_outstanding(tickers: list[str]) -> dict[str, float]:
    """Fetch current shares outstanding for each ticker via yfinance.

    Used together with historical close prices to approximate historical
    market cap (close × shares).  Share counts change slowly, so this is
    a reasonable approximation that avoids look-ahead bias on price.
    """
    shares = {}
    batch_size = 200
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            ts = yf.Tickers(" ".join(batch))
            for sym in batch:
                try:
                    info = ts.tickers[sym].fast_info
                    s = getattr(info, "shares", 0) or 0
                    if s > 0:
                        shares[sym] = float(s)
                except Exception:
                    pass
        except Exception:
            pass
    return shares


def compute_historical_pf(data: dict[str, pd.DataFrame],
                          jojo_cache: dict[str, pd.Series],
                          strategy: int) -> dict[str, float]:
    """Compute historical profit factor for each ticker (full history).

    WARNING: This uses ALL historical data including future bars.
    Using this for candidate ranking in a backtest introduces look-ahead bias.
    Prefer build_rolling_pf_cache() with pf_window > 0 for backtesting.
    """
    import warnings
    warnings.warn(
        "compute_historical_pf uses full history including future data. "
        "This causes look-ahead bias when used for ranking in backtests. "
        "Use pf_window > 0 (rolling PF) instead.",
        UserWarning,
        stacklevel=2,
    )
    pf_map = {}
    for sym, df in data.items():
        if sym not in jojo_cache:
            continue
        pf_map[sym] = _calc_pf_from_series(jojo_cache[sym].values,
                                            df["close"].astype(float).values,
                                            strategy)
    return pf_map


def _calc_pf_from_series(jvals: np.ndarray, closes: np.ndarray, strategy: int) -> float:
    """Calculate profit factor from jojo and close arrays."""
    trades_pnl = []
    in_pos = False
    entry_price = 0.0

    if strategy == 1:
        for i in range(1, len(jvals)):
            if np.isnan(jvals[i]) or np.isnan(jvals[i - 1]):
                continue
            if not in_pos:
                if jvals[i] > 76 and jvals[i - 1] <= 76:
                    in_pos = True
                    entry_price = closes[i]
            else:
                if jvals[i] < 68 and jvals[i - 1] >= 68:
                    trades_pnl.append((closes[i] / entry_price - 1) * 100)
                    in_pos = False
    else:  # strategy 2
        for i in range(1, len(jvals)):
            if np.isnan(jvals[i]) or np.isnan(jvals[i - 1]):
                continue
            if not in_pos:
                if jvals[i - 1] < 28 and jvals[i] > jvals[i - 1]:
                    in_pos = True
                    entry_price = closes[i]
            else:
                if (jvals[i] > 51 and jvals[i - 1] <= 51) or \
                   (jvals[i] < 28 and jvals[i - 1] >= 28):
                    trades_pnl.append((closes[i] / entry_price - 1) * 100)
                    in_pos = False

    gross_p = sum(p for p in trades_pnl if p > 0)
    gross_l = abs(sum(p for p in trades_pnl if p < 0))
    return gross_p / gross_l if gross_l > 0 else (10.0 if gross_p > 0 else 0)


# ---------------------------------------------------------------------------
# Rolling PF computation
# ---------------------------------------------------------------------------

def build_rolling_pf_cache(data: dict[str, pd.DataFrame],
                           jojo_cache: dict[str, pd.Series],
                           strategy: int,
                           all_dates: list,
                           window_days: int = 756) -> dict[str, dict[str, float]]:
    """Pre-compute rolling PF for all tickers, refreshed monthly.

    Returns: {month_key: {ticker: pf_value}}
    where month_key = "YYYY-MM"
    """
    print(f"  Building rolling PF cache (window={window_days} days)...", end="", flush=True)

    # Group dates by month
    months = {}
    for d in all_dates:
        mk = f"{d.year}-{d.month:02d}"
        if mk not in months:
            months[mk] = d  # first trading day of month

    # For each month, compute PF using data up to that month's first day
    pf_cache = {}
    month_keys = sorted(months.keys())

    for mk in month_keys:
        cutoff = months[mk]
        month_pf = {}
        for sym in data:
            if sym not in jojo_cache:
                continue
            jojo = jojo_cache[sym]
            df = data[sym]
            # Get data before cutoff, last `window_days` bars
            mask = jojo.index < cutoff
            if mask.sum() < 60:
                continue
            j_before = jojo[mask].values[-window_days:]
            c_before = df["close"].astype(float).reindex(jojo[mask].index).values[-window_days:]
            if len(j_before) < 60:
                continue
            month_pf[sym] = _calc_pf_from_series(j_before, c_before, strategy)
        pf_cache[mk] = month_pf

    print(f" done ({len(pf_cache)} months)")
    return pf_cache


def get_rolling_pf(pf_cache: dict[str, dict[str, float]], date) -> dict[str, float]:
    """Get the PF map for a given date (uses that month's pre-computed values)."""
    mk = f"{date.year}-{date.month:02d}"
    return pf_cache.get(mk, {})


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def download_spx(start: str = "2010-01-01") -> pd.DataFrame:
    """Download SPX data. Returns DataFrame with 'close' column."""
    try:
        spx = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)
        if isinstance(spx.columns, pd.MultiIndex):
            spx = spx.droplevel("Ticker", axis=1)
        spx.columns = [c.lower() for c in spx.columns]
        return spx
    except Exception:
        return pd.DataFrame()


def build_regime_series(spx_df: pd.DataFrame = None, start: str = "2010-01-01") -> pd.Series:
    """Compute SMA(225), return bull/bear regime Series."""
    if spx_df is None or spx_df.empty:
        spx_df = download_spx(start)
    if spx_df.empty:
        return pd.Series(dtype=str)
    close = spx_df["close"].astype(float)
    sma = close.rolling(225).mean()
    regime = pd.Series("bear", index=close.index)
    regime[close >= sma] = "bull"
    return regime.ffill()


def get_trading_dates(spx_df: pd.DataFrame) -> set:
    """Return set of US market trading dates from SPX data."""
    return set(spx_df.index)


# ---------------------------------------------------------------------------
# Fund simulation engine
# ---------------------------------------------------------------------------

def run_fund(data: dict[str, pd.DataFrame],
             jojo_cache: dict[str, pd.Series],
             atr_cache: dict[str, pd.Series],
             strategy: int,
             max_positions: int = 5,
             stop_loss_pct: float = 20.0,
             min_atr_pct: float = 2.0,
             initial_capital: float = 1_000_000,
             start_date: str = "",
             end_date: str = "",
             pf_window: int = 756,
             regime: pd.Series = None,
             regime_filter: bool = False,
             vol_sizing: bool = False,
             pf_map_override: dict = None,
             rolling_pf_cache: dict = None,
             trading_dates: set = None,
             rank_method: str = "pf",
             shares_map: dict = None,
             sp500_snapshots: list = None) -> tuple[list, list]:
    """
    Run portfolio simulation.

    Returns (snapshots, trades)
    where snapshots = [(date, equity, cash, num_positions), ...]
    """
    # Build common date index — filter to US trading days only
    all_dates = sorted(set().union(*(df.index for df in data.values())))
    if trading_dates:
        all_dates = [d for d in all_dates if d in trading_dates]
    if start_date:
        start_ts = pd.Timestamp(start_date)
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        all_dates = [d for d in all_dates if d <= end_ts]

    if len(all_dates) < 2:
        return [], []

    # Build PF ranking
    if rolling_pf_cache is not None:
        rolling_pf = rolling_pf_cache
        static_pf = None
    elif pf_map_override is not None:
        # Static PF (legacy mode, pf_window=0)
        rolling_pf = None
        static_pf = pf_map_override
    elif pf_window > 0:
        rolling_pf = build_rolling_pf_cache(data, jojo_cache, strategy, all_dates, pf_window)
        static_pf = None
    else:
        # Full history PF
        rolling_pf = None
        static_pf = compute_historical_pf(data, jojo_cache, strategy)

    cash = initial_capital
    positions: dict[str, Position] = {}
    trades: list[FundTrade] = []
    snapshots = []

    def _get_price(sym, date, col="close"):
        """Get price for sym at date, or last available price before date."""
        df = data.get(sym)
        if df is None:
            return None
        c = col if col in df.columns else "close"
        if date in df.index:
            return float(df.loc[date, c])
        # Find last available price before date
        mask = df.index <= date
        if mask.any():
            return float(df.loc[mask].iloc[-1][c])
        return None

    def _close_position(sym, price, date, reason):
        nonlocal cash
        pos = positions.pop(sym)
        proceeds = pos.shares * price
        cash += proceeds
        days = (date - pos.entry_date).days
        pnl_pct = (price / pos.entry_price - 1) * 100
        pnl_dollar = proceeds - pos.shares * pos.entry_price
        trades.append(FundTrade(
            sym, str(pos.entry_date)[:10], round(pos.entry_price, 2),
            str(date)[:10], round(price, 2), pos.shares,
            round(pnl_pct, 2), round(pnl_dollar, 2), days, reason))

    # Pending orders: signals detected on day N, executed at day N+1 open
    pending_sells = []   # [(sym, reason)]
    pending_buys = []    # [(sym, strat, atr_val)]

    for di in range(1, len(all_dates)):
        date = all_dates[di]
        prev_date = all_dates[di - 1]

        # --- 1. Execute pending sell orders from yesterday at today's OPEN ---
        for sym, reason in pending_sells:
            if sym not in positions:
                continue
            price = _get_price(sym, date, "open")
            if price is None:
                price = _get_price(sym, date, "close")
            if price is not None:
                _close_position(sym, price, date, reason)
        pending_sells = []

        # --- 2. Execute pending buy orders from yesterday at today's OPEN ---
        # Determine effective max positions (regime filter) for allocation
        eff_max = max_positions
        if regime_filter and regime is not None and len(regime) > 0:
            idx = regime.index.get_indexer([date], method="ffill")
            if idx[0] >= 0 and regime.iloc[idx[0]] == "bear":
                eff_max = max(1, max_positions // 2)

        if pending_buys and len(positions) < eff_max:
            # Rank pending candidates
            candidates = pending_buys
            if rank_method == "pf":
                if rolling_pf is not None:
                    cur_pf = get_rolling_pf(rolling_pf, prev_date)
                else:
                    cur_pf = static_pf
                ranked = sorted(candidates, key=lambda x: cur_pf.get(x[0], 0), reverse=True)
            elif rank_method in ("mktcap", "mktcap_asc", "mktcap_mid"):
                _sh = shares_map or {}
                def _approx_mktcap(sym):
                    s = _sh.get(sym, 0)
                    if s <= 0 or sym not in data or prev_date not in data[sym].index:
                        return 0
                    return float(data[sym].loc[prev_date, "close"]) * s
                if rank_method == "mktcap_mid":
                    by_cap = sorted(candidates, key=lambda x: _approx_mktcap(x[0]))
                    mid = len(by_cap) // 2
                    ranked = sorted(by_cap, key=lambda x: abs(by_cap.index(x) - mid))
                else:
                    desc = (rank_method == "mktcap")
                    ranked = sorted(candidates, key=lambda x: _approx_mktcap(x[0]), reverse=desc)
            elif rank_method in ("jojo", "jojo_asc"):
                desc = (rank_method == "jojo")
                ranked = sorted(candidates, key=lambda x: x[2], reverse=desc)  # x[2] is atr_val, use jojo from cache
            else:
                ranked = candidates
            available_slots = eff_max - len(positions)

            equity = cash + sum(
                pos.shares * (_get_price(s, date) or pos.entry_price)
                for s, pos in positions.items()
            )

            selected = ranked[:available_slots]

            if vol_sizing and len(selected) > 0:
                inv_atrs = []
                for sym, strat, atr_val in selected:
                    inv_atrs.append(1.0 / max(atr_val, 0.5))
                total_inv = sum(inv_atrs)
                avail_equity = equity * len(selected) / eff_max

                for (sym, strat, atr_val), inv_a in zip(selected, inv_atrs):
                    if cash < 1000:
                        break
                    if sym in positions:
                        continue
                    price = _get_price(sym, date, "open")
                    if price is None:
                        price = _get_price(sym, date, "close")
                    if price is None:
                        continue
                    alloc = min(cash, avail_equity * inv_a / total_inv)
                    shares = int(alloc / price)
                    if shares <= 0:
                        continue
                    cost = shares * price
                    cash -= cost
                    positions[sym] = Position(sym, date, price, shares, strat)
            else:
                size_per_pos = equity / eff_max
                for sym, strat, atr_val in selected:
                    if cash < size_per_pos * 0.5:
                        break
                    if sym in positions:
                        continue
                    price = _get_price(sym, date, "open")
                    if price is None:
                        price = _get_price(sym, date, "close")
                    if price is None:
                        continue
                    alloc = min(cash, size_per_pos)
                    shares = int(alloc / price)
                    if shares <= 0:
                        continue
                    cost = shares * price
                    cash -= cost
                    positions[sym] = Position(sym, date, price, shares, strat)
        pending_buys = []

        # --- 3. Check stop losses at today's close (intraday stop) ---
        to_close = []
        for sym, pos in positions.items():
            price = _get_price(sym, date)
            if price is None:
                continue
            pnl = (price / pos.entry_price - 1) * 100
            if pnl <= -stop_loss_pct:
                to_close.append((sym, price, f"止损({stop_loss_pct:.0f}%)"))

        for sym, price, reason in to_close:
            _close_position(sym, price, date, reason)

        # --- 4. Detect sell signals → queue for next-day execution ---
        for sym, pos in list(positions.items()):
            if sym not in jojo_cache:
                continue
            jojo = jojo_cache[sym]
            if date not in jojo.index or prev_date not in jojo.index:
                continue
            j_today = jojo.loc[date]
            j_yest = jojo.loc[prev_date]
            if np.isnan(j_today) or np.isnan(j_yest):
                continue

            sell_reason = None
            if pos.strategy == 1:
                if j_today < 68 and j_yest >= 68:
                    sell_reason = "下穿68"
            elif pos.strategy == 2:
                if j_today > 51 and j_yest <= 51:
                    sell_reason = "上穿51"
                elif j_today < 28 and j_yest >= 28:
                    sell_reason = "再次下穿28"

            if sell_reason:
                pending_sells.append((sym, sell_reason))

        # --- 5. Detect buy signals → queue for next-day execution ---
        eligible_syms = set(data.keys())
        if sp500_snapshots:
            sp500_members = _get_sp500_members_at(sp500_snapshots, date)
            eligible_syms = sp500_members.intersection(eligible_syms) | {
                s for s in data if s.endswith("=F")}

        # Exclude positions and pending sells
        pending_sell_syms = {s for s, _ in pending_sells}
        for sym in eligible_syms:
            if sym in positions or sym in pending_sell_syms:
                continue
            if sym not in jojo_cache:
                continue
            jojo = jojo_cache[sym]
            if date not in jojo.index or prev_date not in jojo.index:
                continue
            j_today = jojo.loc[date]
            j_yest = jojo.loc[prev_date]
            if np.isnan(j_today) or np.isnan(j_yest):
                continue

            buy_signal = False
            strat = 0
            if strategy in (1, 3):
                if j_today > 76 and j_yest <= 76:
                    if sym in atr_cache and date in atr_cache[sym].index:
                        atr_val = atr_cache[sym].loc[date]
                        if not np.isnan(atr_val) and atr_val >= min_atr_pct:
                            buy_signal = True
                            strat = 1
            if not buy_signal and strategy in (2, 3):
                if j_yest < 28 and j_today > j_yest:
                    buy_signal = True
                    strat = 2

            if buy_signal:
                atr_val = 0.0
                if sym in atr_cache and date in atr_cache[sym].index:
                    av = atr_cache[sym].loc[date]
                    if not np.isnan(av):
                        atr_val = float(av)
                pending_buys.append((sym, strat, atr_val))

        # --- 6. Daily snapshot ---
        mkt_value = 0
        for s, pos in positions.items():
            p = _get_price(s, date)
            if p is not None:
                mkt_value += pos.shares * p
            else:
                mkt_value += pos.shares * pos.entry_price
        equity = cash + mkt_value
        snapshots.append((date, equity, cash, len(positions)))

    # Close remaining positions at last bar
    if all_dates:
        last_date = all_dates[-1]
        for sym, pos in list(positions.items()):
            if last_date in data[sym].index:
                price = float(data[sym].loc[last_date, "close"])
                _close_position(sym, price, last_date, "持仓中")

    return snapshots, trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_benchmark_metrics(spx_series: pd.Series, dates: list,
                               initial_capital: float) -> dict:
    """Compute SPX buy-and-hold benchmark metrics aligned to fund dates."""
    # Align SPX to fund dates
    spx_aligned = spx_series.reindex([d for d in dates], method="ffill").dropna()
    if len(spx_aligned) < 2:
        return {}
    spx_vals = spx_aligned.values
    spx_equity = initial_capital * spx_vals / spx_vals[0]

    total_ret = (spx_equity[-1] / initial_capital - 1) * 100
    d = (spx_aligned.index[-1] - spx_aligned.index[0]).days
    ann_ret = ((spx_equity[-1] / initial_capital) ** (365.25 / d) - 1) * 100 if d > 0 else 0

    peak = np.maximum.accumulate(spx_equity)
    dd = (peak - spx_equity) / peak * 100
    max_dd = float(np.max(dd))
    max_dd_date = spx_aligned.index[int(np.argmax(dd))]

    daily_ret = np.diff(spx_equity) / spx_equity[:-1]
    sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    sortino = float(np.mean(daily_ret) / np.std(neg_ret) * np.sqrt(252)) if len(neg_ret) > 0 and np.std(neg_ret) > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    eq_series = pd.Series(spx_equity, index=spx_aligned.index)
    monthly = eq_series.resample("ME").last().pct_change().dropna() * 100

    return {
        "final_equity": round(spx_equity[-1], 0),
        "total_return": round(total_ret, 1),
        "ann_return": round(ann_ret, 1),
        "max_drawdown": round(max_dd, 1),
        "max_dd_date": str(max_dd_date)[:10],
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "monthly_returns": monthly,
        "equity": spx_equity,
        "dates": spx_aligned.index.tolist(),
    }


def compute_fund_metrics(snapshots, trades, initial_capital, spx_series=None):
    """Compute comprehensive fund performance metrics."""
    if not snapshots:
        return {}

    dates = [s[0] for s in snapshots]
    equity = np.array([s[1] for s in snapshots])
    cash_arr = np.array([s[2] for s in snapshots])
    npos_arr = np.array([s[3] for s in snapshots])

    total_ret = (equity[-1] / initial_capital - 1) * 100
    days = (dates[-1] - dates[0]).days
    ann_ret = ((equity[-1] / initial_capital) ** (365.25 / days) - 1) * 100 if days > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = float(np.max(dd))
    max_dd_date = dates[int(np.argmax(dd))]

    # Daily returns
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    sortino = float(np.mean(daily_ret) / np.std(neg_ret) * np.sqrt(252)) if len(neg_ret) > 0 and np.std(neg_ret) > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    # Trade metrics
    pnls = [t.pnl_pct for t in trades]
    wins = sum(1 for p in pnls if p > 0) if pnls else 0
    win_rate = wins / len(pnls) * 100 if pnls else 0
    gross_p = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_p / gross_l if gross_l > 0 else (float("inf") if gross_p > 0 else 0)
    avg_pnl = float(np.mean(pnls)) if pnls else 0
    avg_holding = float(np.mean([t.holding_days for t in trades])) if trades else 0

    # Capital utilization
    invested = equity - cash_arr
    cap_util = float(np.mean(invested / equity * 100))

    # Monthly returns
    eq_series = pd.Series(equity, index=dates)
    monthly = eq_series.resample("ME").last().pct_change().dropna() * 100

    # Drawdown series
    dd_series = pd.Series(dd, index=dates)

    # Benchmark
    bench = {}
    if spx_series is not None:
        bench = _compute_benchmark_metrics(spx_series, dates, initial_capital)

    return {
        "initial_capital": initial_capital,
        "final_equity": round(equity[-1], 0),
        "total_return": round(total_ret, 1),
        "ann_return": round(ann_ret, 1),
        "max_drawdown": round(max_dd, 1),
        "max_dd_date": str(max_dd_date)[:10],
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "avg_pnl": round(avg_pnl, 2),
        "avg_holding": round(avg_holding, 1),
        "max_concurrent": int(np.max(npos_arr)) if len(npos_arr) > 0 else 0,
        "capital_utilization": round(cap_util, 1),
        "start_date": str(dates[0])[:10],
        "end_date": str(dates[-1])[:10],
        "trading_days": len(dates),
        "monthly_returns": monthly,
        "drawdown_series": dd_series,
        "equity_series": eq_series,
        "benchmark": bench,
    }


def metrics_one_line(m):
    """Return a compact one-line summary dict for comparison tables."""
    return {
        "total_return": m["total_return"],
        "ann_return": m["ann_return"],
        "max_dd": m["max_drawdown"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "trades": m["total_trades"],
        "win_rate": m["win_rate"],
        "pf": m["profit_factor"],
        "final_equity": m["final_equity"],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(metrics, config_str):
    """Print fund performance summary to console."""
    m = metrics
    print()
    print("=" * 70)
    print("  韭韭量化 基金回测结果")
    print("=" * 70)
    print(f"  {config_str}")
    print(f"  回测区间: {m['start_date']} ~ {m['end_date']} ({m['trading_days']} 交易日)")
    print("-" * 70)
    print(f"  初始资金:       ${m['initial_capital']:>14,.0f}")
    print(f"  终值:           ${m['final_equity']:>14,.0f}")
    print(f"  总收益:         {m['total_return']:>13.1f}%")
    print(f"  年化收益:       {m['ann_return']:>13.1f}%")
    print(f"  最大回撤:       {m['max_drawdown']:>13.1f}%  ({m['max_dd_date']})")
    print(f"  Sharpe:         {m['sharpe']:>13.2f}")
    print(f"  Sortino:        {m['sortino']:>13.2f}")
    print(f"  Calmar:         {m['calmar']:>13.2f}")
    print("-" * 70)
    print(f"  总交易数:       {m['total_trades']:>13d}")
    print(f"  胜率:           {m['win_rate']:>13.1f}%")
    print(f"  盈亏比:         {str(m['profit_factor']):>13s}")
    print(f"  平均收益:       {m['avg_pnl']:>13.2f}%")
    print(f"  平均持仓:       {m['avg_holding']:>13.1f} 天")
    print(f"  最大同时持仓:   {m['max_concurrent']:>13d}")
    print(f"  资金利用率:     {m['capital_utilization']:>13.1f}%")
    print("=" * 70)

    # Benchmark comparison
    bench = m.get("benchmark", {})
    if bench:
        print("\n  业绩基准 (SPX 500 买入持有):")
        print("-" * 70)
        print(f"  SPX 终值:       ${bench['final_equity']:>14,.0f}")
        print(f"  SPX 总收益:     {bench['total_return']:>13.1f}%")
        print(f"  SPX 年化收益:   {bench['ann_return']:>13.1f}%")
        print(f"  SPX 最大回撤:   {bench['max_drawdown']:>13.1f}%  ({bench['max_dd_date']})")
        print(f"  SPX Sharpe:     {bench['sharpe']:>13.2f}")
        alpha = m['ann_return'] - bench['ann_return']
        print(f"  超额年化(α):    {alpha:>+13.1f}%")
        print("=" * 70)

    # Monthly returns heatmap
    monthly = m.get("monthly_returns")
    if monthly is not None and len(monthly) > 0:
        print("\n月度收益率 (%) — 策略:")
        print("-" * 70)
        mdf = pd.DataFrame({"ret": monthly})
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        pivot.columns = [f"{c:>2d}月" for c in pivot.columns]
        yearly = mdf.groupby("year")["ret"].sum()
        pivot["  年度"] = yearly
        print(pivot.to_string(float_format=lambda x: f"{x:>6.1f}"))

    # Benchmark monthly heatmap
    bench_monthly = bench.get("monthly_returns") if bench else None
    if bench_monthly is not None and len(bench_monthly) > 0:
        print(f"\n月度收益率 (%) — SPX 基准:")
        print("-" * 70)
        bmdf = pd.DataFrame({"ret": bench_monthly})
        bmdf["year"] = bmdf.index.year
        bmdf["month"] = bmdf.index.month
        bpivot = bmdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        bpivot.columns = [f"{c:>2d}月" for c in bpivot.columns]
        byearly = bmdf.groupby("year")["ret"].sum()
        bpivot["  年度"] = byearly
        print(bpivot.to_string(float_format=lambda x: f"{x:>6.1f}"))

    # Drawdown summary
    dd_series = m.get("drawdown_series")
    if dd_series is not None and len(dd_series) > 0:
        print(f"\n最大回撤 Top 5:")
        print("-" * 70)
        # Find distinct drawdown episodes (peaks)
        dd_monthly = dd_series.resample("ME").max()
        top_dd = dd_monthly.nlargest(5)
        for i, (dt, val) in enumerate(top_dd.items(), 1):
            print(f"  {i}. {str(dt)[:7]}  回撤 {val:.1f}%")
    print()


def print_comparison(results: list[tuple[str, dict]]):
    """Print comparison table of multiple configurations."""
    print("\n" + "=" * 120)
    print("  配置对比")
    print("=" * 120)
    header = (f"{'配置':<35s} {'总收益%':>8s} {'年化%':>7s} {'最大回撤%':>9s} "
              f"{'Sharpe':>7s} {'Sortino':>8s} {'交易数':>6s} {'胜率%':>6s} "
              f"{'盈亏比':>7s} {'终值':>14s}")
    print(header)
    print("-" * 120)
    for label, r in results:
        pf_str = f"{r['pf']}" if isinstance(r['pf'], str) else f"{r['pf']:.2f}"
        print(f"{label:<35s} {r['total_return']:>8.1f} {r['ann_return']:>7.1f} "
              f"{r['max_dd']:>9.1f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['trades']:>6d} {r['win_rate']:>6.1f} {pf_str:>7s} "
              f"{r['final_equity']:>14,.0f}")
    print("=" * 120)


def generate_report(metrics, trades, config_str, output_dir="reports"):
    """Generate markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    m = metrics
    run_date = datetime.now().strftime("%Y-%m-%d")

    md = []
    md.append(f"# 韭韭量化 基金回测报告\n")
    md.append(f"**生成日期**: {run_date}")
    md.append(f"**配置**: {config_str}\n")

    md.append("## 业绩概览\n")
    md.append("| 指标 | 值 |")
    md.append("|------|------|")
    rows = [
        ("回测区间", f"{m['start_date']} ~ {m['end_date']}"),
        ("初始资金", f"${m['initial_capital']:,.0f}"),
        ("终值", f"${m['final_equity']:,.0f}"),
        ("总收益", f"{m['total_return']}%"),
        ("年化收益", f"{m['ann_return']}%"),
        ("最大回撤", f"{m['max_drawdown']}% ({m['max_dd_date']})"),
        ("Sharpe", f"{m['sharpe']}"),
        ("Sortino", f"{m['sortino']}"),
        ("Calmar", f"{m['calmar']}"),
        ("总交易数", f"{m['total_trades']}"),
        ("胜率", f"{m['win_rate']}%"),
        ("盈亏比", f"{m['profit_factor']}"),
        ("平均收益", f"{m['avg_pnl']}%"),
        ("平均持仓", f"{m['avg_holding']}天"),
        ("最大同时持仓", f"{m['max_concurrent']}"),
        ("资金利用率", f"{m['capital_utilization']}%"),
    ]
    for label, val in rows:
        md.append(f"| {label} | {val} |")

    # Benchmark comparison
    bench = m.get("benchmark", {})
    if bench:
        md.append("\n## 业绩基准 (SPX 500 买入持有)\n")
        md.append("| 指标 | 策略 | SPX |")
        md.append("|------|------|-----|")
        md.append(f"| 终值 | ${m['final_equity']:,.0f} | ${bench['final_equity']:,.0f} |")
        md.append(f"| 总收益 | {m['total_return']}% | {bench['total_return']}% |")
        md.append(f"| 年化收益 | {m['ann_return']}% | {bench['ann_return']}% |")
        md.append(f"| 最大回撤 | {m['max_drawdown']}% | {bench['max_drawdown']}% |")
        md.append(f"| Sharpe | {m['sharpe']} | {bench['sharpe']} |")
        md.append(f"| Sortino | {m['sortino']} | {bench['sortino']} |")
        alpha = round(m['ann_return'] - bench['ann_return'], 1)
        md.append(f"| **超额年化(α)** | **{alpha:+.1f}%** | — |")

    def _pivot_to_md(monthly_series):
        """Convert monthly return series to markdown table."""
        mdf = pd.DataFrame({"ret": monthly_series})
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        yearly = mdf.groupby("year")["ret"].sum()
        # Build markdown table manually
        cols = list(pivot.columns)
        header = "| 年份 | " + " | ".join(f"{c}月" for c in cols) + " | 年度 |"
        sep = "|------|" + "|".join("------:" for _ in cols) + "|------:|"
        rows = [header, sep]
        for yr in pivot.index:
            vals = []
            for c in cols:
                v = pivot.loc[yr, c]
                vals.append(f"{v:.1f}" if not pd.isna(v) else "")
            yr_total = yearly.get(yr, 0)
            rows.append(f"| {yr} | " + " | ".join(vals) + f" | {yr_total:.1f} |")
        return "\n".join(rows)

    # Monthly heatmap
    monthly = m.get("monthly_returns")
    if monthly is not None and len(monthly) > 0:
        md.append("\n## 月度收益率 (%) — 策略\n")
        md.append(_pivot_to_md(monthly))

    bench_monthly = bench.get("monthly_returns") if bench else None
    if bench_monthly is not None and len(bench_monthly) > 0:
        md.append("\n## 月度收益率 (%) — SPX 基准\n")
        md.append(_pivot_to_md(bench_monthly))

    # Drawdown top 5
    dd_series = m.get("drawdown_series")
    if dd_series is not None and len(dd_series) > 0:
        md.append("\n## 最大回撤 Top 5\n")
        md.append("| # | 月份 | 回撤% |")
        md.append("|---|------|-------|")
        dd_monthly = dd_series.resample("ME").max()
        top_dd = dd_monthly.nlargest(5)
        for i, (dt, val) in enumerate(top_dd.items(), 1):
            md.append(f"| {i} | {str(dt)[:7]} | {val:.1f}% |")

    # Top trades
    if trades:
        sorted_by_pnl = sorted(trades, key=lambda t: t.pnl_dollar, reverse=True)
        md.append("\n## Top 10 盈利交易\n")
        md.append("| # | Ticker | 买入 | 卖出 | 持仓天数 | 收益% | 收益$ | 原因 |")
        md.append("|---|--------|------|------|----------|-------|-------|------|")
        for i, t in enumerate(sorted_by_pnl[:10], 1):
            md.append(f"| {i} | {t.ticker} | {t.entry_date} @ {t.entry_price} "
                      f"| {t.exit_date} @ {t.exit_price} | {t.holding_days} "
                      f"| {t.pnl_pct:+.2f}% | ${t.pnl_dollar:+,.0f} | {t.exit_reason} |")

        md.append("\n## Top 10 亏损交易\n")
        md.append("| # | Ticker | 买入 | 卖出 | 持仓天数 | 收益% | 收益$ | 原因 |")
        md.append("|---|--------|------|------|----------|-------|-------|------|")
        for i, t in enumerate(sorted_by_pnl[-10:][::-1], 1):
            md.append(f"| {i} | {t.ticker} | {t.entry_date} @ {t.entry_price} "
                      f"| {t.exit_date} @ {t.exit_price} | {t.holding_days} "
                      f"| {t.pnl_pct:+.2f}% | ${t.pnl_dollar:+,.0f} | {t.exit_reason} |")

    report = "\n".join(md)
    path = os.path.join(output_dir, "fund_report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"  Report: {path} ({len(report)} chars)")
    return path


def export_csv(snapshots, trades, output_dir="reports"):
    """Export equity curve and trade log as CSV."""
    os.makedirs(output_dir, exist_ok=True)

    eq_path = os.path.join(output_dir, "fund_equity.csv")
    eq_df = pd.DataFrame(snapshots, columns=["date", "equity", "cash", "positions"])
    eq_df["date"] = eq_df["date"].astype(str).str[:10]
    eq_df.to_csv(eq_path, index=False)
    print(f"  Equity CSV: {eq_path} ({len(eq_df)} rows)")

    tr_path = os.path.join(output_dir, "fund_trades.csv")
    tr_df = pd.DataFrame([{
        "ticker": t.ticker, "entry_date": t.entry_date, "entry_price": t.entry_price,
        "exit_date": t.exit_date, "exit_price": t.exit_price, "shares": t.shares,
        "pnl_pct": t.pnl_pct, "pnl_dollar": t.pnl_dollar,
        "holding_days": t.holding_days, "exit_reason": t.exit_reason,
    } for t in trades])
    tr_df.to_csv(tr_path, index=False)
    print(f"  Trades CSV: {tr_path} ({len(tr_df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="韭韭量化 基金回测器")
    parser.add_argument("--strategy", type=int, default=1, choices=[1, 2, 3],
                        help="1=超买动量, 2=超卖反转, 3=both (default: 1)")
    parser.add_argument("--universe", type=str, default="sp500",
                        choices=["sp500", "sp500+", "report", "custom"],
                        help="Stock universe (default: sp500)")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated tickers for --universe custom")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Initial capital (default: 1000000)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max concurrent positions (default: 5)")
    parser.add_argument("--stop-loss", type=float, default=20.0,
                        help="Stop loss %% (default: 20)")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="Backtest start date (default: 2015-01-01)")
    parser.add_argument("--end", type=str, default="",
                        help="Backtest end date (default: today)")
    parser.add_argument("--pf-window", type=int, default=756,
                        help="Rolling PF window in trading days (default: 756 ~3yr, 0=full history)")
    parser.add_argument("--regime-filter", action="store_true",
                        help="Reduce positions in bear market (SPX < SMA225)")
    parser.add_argument("--vol-sizing", action="store_true",
                        help="ATR%% inverse position sizing (lower vol = bigger position)")
    parser.add_argument("--compare", action="store_true",
                        help="Run 4 configs and compare: base / +regime / +vol / +both")
    parser.add_argument("--cache-dir", type=str, default="data/",
                        help="Cache directory for OHLC data (default: data/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download (skip cache)")
    parser.add_argument("--output-dir", type=str, default="reports/",
                        help="Output directory (default: reports/)")
    parser.add_argument("--rank-method", type=str, default="pf",
                        choices=["pf", "mktcap", "mktcap_asc", "jojo", "jojo_asc"],
                        help="Ranking method: pf, mktcap (large first), mktcap_asc (small first), jojo (high first), jojo_asc (low first)")
    parser.add_argument("--rank-compare", action="store_true",
                        help="Compare all rank methods × TopN combinations")
    parser.add_argument("--historical", action="store_true",
                        help="Use historical S&P 500 membership (avoid survivorship bias)")
    args = parser.parse_args()

    strat_names = {1: "策略1(超买动量)", 2: "策略2(超卖反转)", 3: "策略1+2(全部)"}
    pf_label = f"滚动PF({args.pf_window}天)" if args.pf_window > 0 else "全量PF"

    print("=" * 70)
    print("  韭韭量化 基金回测器")
    print("=" * 70)
    print()

    # Step 1: Get universe
    print("[1/5] Loading universe...")
    sp500_snapshots = []
    if args.historical:
        result = get_universe(args.universe, args.tickers, historical=True)
        tickers, sp500_snapshots = result
        print(f"  Universe: {args.universe} (historical, {len(tickers)} tickers ever in index)")
    else:
        tickers = get_universe(args.universe, args.tickers)
        print(f"  Universe: {args.universe} ({len(tickers)} tickers)")

    # Step 2: Download data (need extra history for rolling PF)
    dl_start = "2010-01-01" if args.pf_window > 0 else args.start
    print(f"\n[2/5] Downloading OHLC data (from {dl_start})...")
    data = download_data(tickers, dl_start, args.cache_dir, args.no_cache)
    if not data:
        print("No data! Exiting.")
        sys.exit(1)

    # Step 3: Pre-compute
    print(f"\n[3/5] Pre-computing indicators for {len(data)} tickers...")
    jojo_cache = precompute_jojo(data)
    atr_cache = precompute_atr_pct(data)

    # Step 4: SPX data (for regime + trading dates)
    print("  Downloading SPX...")
    spx_df = download_spx("2010-01-01")
    spx_close = spx_df["close"].astype(float) if not spx_df.empty else None
    td = get_trading_dates(spx_df) if not spx_df.empty else None
    if td:
        print(f"  US trading dates: {len(td)} days")

    regime = None
    if args.regime_filter or args.compare:
        regime = build_regime_series(spx_df)
        if len(regime) > 0:
            current = regime.iloc[-1]
            print(f"  Current regime: {'牛市' if current == 'bull' else '熊市'}")

    # Step 5: Run
    common_kwargs = dict(
        data=data, jojo_cache=jojo_cache, atr_cache=atr_cache,
        strategy=args.strategy, max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss, initial_capital=args.capital,
        start_date=args.start, end_date=args.end,
        pf_window=args.pf_window,
        trading_dates=td,
        sp500_snapshots=sp500_snapshots if sp500_snapshots else None,
    )

    if args.rank_compare:
        print(f"\n[4/5] Running rank method × TopN comparison...")

        # Pre-build rolling PF cache once
        all_dates = sorted(set().union(*(df.index for df in data.values())))
        if td:
            all_dates = [d for d in all_dates if d in td]
        if args.start:
            start_ts = pd.Timestamp(args.start)
            all_dates = [d for d in all_dates if d >= start_ts]
        shared_rolling_pf = None
        shared_static_pf = None
        if args.pf_window > 0:
            shared_rolling_pf = build_rolling_pf_cache(
                data, jojo_cache, args.strategy, all_dates, args.pf_window)
        else:
            shared_static_pf = compute_historical_pf(data, jojo_cache, args.strategy)

        # Fetch shares outstanding for mktcap ranking
        print("  Fetching shares outstanding...")
        shares_map = fetch_shares_outstanding(list(data.keys()))
        print(f"  Got shares for {len(shares_map)} tickers.")

        top_ns = [3, 5, 8, 10]
        rank_methods = [
            ("滚动PF", "pf"),
            ("市值", "mktcap"),
            ("jojo指标", "jojo"),
        ]

        comparison = []
        for top_n in top_ns:
            for rank_label, rank_m in rank_methods:
                label = f"Top{top_n} | {rank_label}"
                print(f"  Running: {label}...")
                run_kwargs = dict(
                    data=data, jojo_cache=jojo_cache, atr_cache=atr_cache,
                    strategy=args.strategy, max_positions=top_n,
                    stop_loss_pct=args.stop_loss, initial_capital=args.capital,
                    start_date=args.start, end_date=args.end,
                    pf_window=0, trading_dates=td,
                    rank_method=rank_m, shares_map=shares_map,
                    sp500_snapshots=sp500_snapshots if sp500_snapshots else None,
                )
                if rank_m == "pf":
                    if shared_rolling_pf is not None:
                        run_kwargs["rolling_pf_cache"] = shared_rolling_pf
                    elif shared_static_pf is not None:
                        run_kwargs["pf_map_override"] = shared_static_pf

                snapshots, trades = run_fund(**run_kwargs)
                m = compute_fund_metrics(snapshots, trades, args.capital, spx_close)
                comparison.append((label, metrics_one_line(m)))

        print_comparison(comparison)
        print("\nDone.")
        sys.exit(0)

    elif args.compare:
        print(f"\n[4/5] Running 4 configurations...")

        # Pre-build rolling PF cache once (shared across all configs)
        all_dates = sorted(set().union(*(df.index for df in data.values())))
        if td:
            all_dates = [d for d in all_dates if d in td]
        if args.start:
            start_ts = pd.Timestamp(args.start)
            all_dates = [d for d in all_dates if d >= start_ts]
        shared_rolling_pf = None
        shared_static_pf = None
        if args.pf_window > 0:
            shared_rolling_pf = build_rolling_pf_cache(
                data, jojo_cache, args.strategy, all_dates, args.pf_window)
        else:
            shared_static_pf = compute_historical_pf(data, jojo_cache, args.strategy)

        configs = [
            ("基线 (滚动PF+等权)", dict(regime_filter=False, vol_sizing=False)),
            ("+熊市减仓", dict(regime=regime, regime_filter=True, vol_sizing=False)),
            ("+ATR%动态仓位", dict(regime_filter=False, vol_sizing=True)),
            ("+熊市减仓+ATR%动态仓位", dict(regime=regime, regime_filter=True, vol_sizing=True)),
        ]
        comparison = []
        best_snapshots, best_trades, best_metrics, best_label = None, None, None, None
        # Override common_kwargs to pass pre-built cache and skip internal rebuild
        compare_kwargs = dict(common_kwargs)
        compare_kwargs["pf_window"] = 0  # disable internal build
        if shared_rolling_pf is not None:
            compare_kwargs["rolling_pf_cache"] = shared_rolling_pf
        elif shared_static_pf is not None:
            compare_kwargs["pf_map_override"] = shared_static_pf

        for label, kw in configs:
            print(f"\n  Running: {label}...")
            snapshots, trades = run_fund(**compare_kwargs, **kw)
            m = compute_fund_metrics(snapshots, trades, args.capital, spx_close)
            comparison.append((label, metrics_one_line(m)))
            # Keep the best (highest Sharpe, then Sortino as tiebreaker) for full report
            cur_score = (m.get("sharpe", 0), m.get("sortino", 0))
            best_score = (best_metrics.get("sharpe", 0), best_metrics.get("sortino", 0)) if best_metrics else (-999, -999)
            if cur_score >= best_score:
                best_snapshots, best_trades, best_metrics, best_label = snapshots, trades, m, label

        print_comparison(comparison)

        # Generate report for best config
        print(f"\n[5/5] Generating output for best config: {best_label}")
        config_str = f"{strat_names[args.strategy]} | {args.universe} | {pf_label} | {best_label}"
        print_summary(best_metrics, config_str)
        generate_report(best_metrics, best_trades, config_str, args.output_dir)
        export_csv(best_snapshots, best_trades, args.output_dir)
    else:
        config_str = (f"{strat_names[args.strategy]} | {args.universe} | {pf_label} | "
                      f"资金${args.capital:,.0f} | 最大{args.max_positions}仓 | "
                      f"止损{args.stop_loss}%"
                      + (" | 熊市减仓" if args.regime_filter else "")
                      + (" | ATR%动态仓位" if args.vol_sizing else ""))

        print(f"\n[4/5] Running fund simulation...")
        extra_kwargs = {}
        if args.rank_method != "pf":
            extra_kwargs["rank_method"] = args.rank_method
            if args.rank_method in ("mktcap", "mktcap_asc"):
                print("  Fetching shares outstanding...")
                extra_kwargs["shares_map"] = fetch_shares_outstanding(list(data.keys()))
        snapshots, trades = run_fund(
            **common_kwargs,
            regime=regime if args.regime_filter else None,
            regime_filter=args.regime_filter,
            vol_sizing=args.vol_sizing,
            **extra_kwargs,
        )

        if not snapshots:
            print("No snapshots generated! Check date range.")
            sys.exit(1)

        print(f"\n[5/5] Generating output...")
        metrics = compute_fund_metrics(snapshots, trades, args.capital, spx_close)
        print_summary(metrics, config_str)
        generate_report(metrics, trades, config_str, args.output_dir)
        export_csv(snapshots, trades, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
