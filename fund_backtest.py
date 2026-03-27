"""
jojo_quant 基金回测器 — portfolio-level backtest using jojo signals.

Simulates a fund that:
1. Scans all stocks daily for jojo buy/sell signals
2. Ranks buy candidates by historical profit factor (top 5)
3. Manages positions with stop loss and equal-weight sizing
4. Tracks portfolio equity curve and computes fund-level metrics

Usage:
    python fund_backtest.py --strategy 1 --universe sp500
    python fund_backtest.py --strategy 1 --universe report --start 2020-01-01
    python fund_backtest.py --strategy 1 --universe custom --tickers TSLA,NVDA,AAPL
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from indicators import compute_jojo

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


def get_universe(name: str, custom_tickers: str = "") -> list[str]:
    """Get ticker list for the specified universe."""
    if name == "sp500":
        tickers = get_sp500_tickers()
        if not tickers:
            print("Falling back to report tickers.")
            return REPORT_TICKERS
        return tickers
    elif name == "report":
        return REPORT_TICKERS
    elif name == "custom":
        return [t.strip() for t in custom_tickers.split(",") if t.strip()]
    else:
        return REPORT_TICKERS


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
        import time
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
    """Compute ATR%(14) for all tickers."""
    cache = {}
    for sym, df in data.items():
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = pd.Series(tr, index=df.index).rolling(14).mean()
        cache[sym] = atr / df["close"].astype(float) * 100
    return cache


def compute_historical_pf(data: dict[str, pd.DataFrame],
                          jojo_cache: dict[str, pd.Series],
                          strategy: int) -> dict[str, float]:
    """Compute historical profit factor for each ticker."""
    pf_map = {}
    for sym, df in data.items():
        if sym not in jojo_cache:
            continue
        jvals = jojo_cache[sym].values
        closes = df["close"].astype(float).values
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
        pf_map[sym] = gross_p / gross_l if gross_l > 0 else (10.0 if gross_p > 0 else 0)
    return pf_map


# ---------------------------------------------------------------------------
# Fund simulation engine
# ---------------------------------------------------------------------------

def run_fund(data: dict[str, pd.DataFrame],
             jojo_cache: dict[str, pd.Series],
             atr_cache: dict[str, pd.Series],
             pf_map: dict[str, float],
             strategy: int,
             max_positions: int = 5,
             stop_loss_pct: float = 20.0,
             min_atr_pct: float = 2.0,
             initial_capital: float = 1_000_000,
             start_date: str = "",
             end_date: str = "") -> tuple[list, list, list]:
    """
    Run portfolio simulation.

    Returns (snapshots, trades, monthly_returns)
    where snapshots = [(date, equity, cash, num_positions), ...]
    """
    # Build common date index
    all_dates = sorted(set().union(*(df.index for df in data.values())))
    if start_date:
        start_ts = pd.Timestamp(start_date)
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        all_dates = [d for d in all_dates if d <= end_ts]

    if len(all_dates) < 2:
        return [], [], []

    cash = initial_capital
    positions: dict[str, Position] = {}
    trades: list[FundTrade] = []
    snapshots = []

    for di in range(1, len(all_dates)):
        date = all_dates[di]
        prev_date = all_dates[di - 1]

        # --- 1. Check stop losses ---
        to_close = []
        for sym, pos in positions.items():
            if date not in data.get(sym, pd.DataFrame()).index:
                continue
            price = float(data[sym].loc[date, "close"])
            pnl = (price / pos.entry_price - 1) * 100
            if pnl <= -stop_loss_pct:
                to_close.append((sym, price, f"止损({stop_loss_pct:.0f}%)"))

        for sym, price, reason in to_close:
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

            sell_reason = None
            if pos.strategy == 1:
                if j_today < 68 and j_yest >= 68:
                    sell_reason = "下穿68"
            elif pos.strategy == 2:
                if j_today > 51 and j_yest <= 51:
                    sell_reason = "上穿51"
                elif j_today < 28 and j_yest >= 28:
                    sell_reason = "再次下穿28"

            if sell_reason and date in data[sym].index:
                price = float(data[sym].loc[date, "close"])
                to_sell.append((sym, price, sell_reason))

        for sym, price, reason in to_sell:
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

        # --- 3. Check buy signals ---
        candidates = []
        for sym in data:
            if sym in positions:
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
            if strategy in (1, 3):  # 3 = "all"
                if j_today > 76 and j_yest <= 76:
                    # ATR% filter for S1
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
                candidates.append((sym, float(j_today), strat))

        # --- 4. Rank by historical PF and allocate ---
        if candidates and len(positions) < max_positions:
            ranked = sorted(candidates, key=lambda x: pf_map.get(x[0], 0), reverse=True)
            available_slots = max_positions - len(positions)

            # Current equity for sizing
            equity = cash + sum(
                pos.shares * float(data[s].loc[date, "close"])
                for s, pos in positions.items()
                if date in data[s].index
            )
            size_per_pos = equity / max_positions

            for sym, j_val, strat in ranked[:available_slots]:
                if cash < size_per_pos * 0.5:
                    break
                if date not in data[sym].index:
                    continue
                alloc = min(cash, size_per_pos)
                price = float(data[sym].loc[date, "close"])
                shares = int(alloc / price)
                if shares <= 0:
                    continue
                cost = shares * price
                cash -= cost
                positions[sym] = Position(sym, date, price, shares, strat)

        # --- 5. Daily snapshot ---
        mkt_value = sum(
            pos.shares * float(data[s].loc[date, "close"])
            for s, pos in positions.items()
            if date in data[s].index
        )
        equity = cash + mkt_value
        snapshots.append((date, equity, cash, len(positions)))

    # Close remaining positions at last bar
    if all_dates:
        last_date = all_dates[-1]
        for sym, pos in list(positions.items()):
            if last_date in data[sym].index:
                price = float(data[sym].loc[last_date, "close"])
                days = (last_date - pos.entry_date).days
                pnl_pct = (price / pos.entry_price - 1) * 100
                pnl_dollar = pos.shares * price - pos.shares * pos.entry_price
                trades.append(FundTrade(
                    sym, str(pos.entry_date)[:10], round(pos.entry_price, 2),
                    str(last_date)[:10], round(price, 2), pos.shares,
                    round(pnl_pct, 2), round(pnl_dollar, 2), days, "持仓中"))

    return snapshots, trades, []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_fund_metrics(snapshots, trades, initial_capital):
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

    # Monthly returns heatmap
    monthly = m.get("monthly_returns")
    if monthly is not None and len(monthly) > 0:
        print("\n月度收益率 (%):")
        print("-" * 70)
        # Pivot to year x month
        mdf = pd.DataFrame({"ret": monthly})
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        pivot.columns = [f"{m:>2d}月" for m in pivot.columns]
        # Add yearly total
        yearly = mdf.groupby("year")["ret"].sum()
        pivot["  年度"] = yearly
        print(pivot.to_string(float_format=lambda x: f"{x:>6.1f}"))
        print()


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

    # Monthly returns
    monthly = m.get("monthly_returns")
    if monthly is not None and len(monthly) > 0:
        md.append("\n## 月度收益率 (%)\n")
        mdf = pd.DataFrame({"ret": monthly})
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        pivot.columns = [f"{m:d}月" for m in pivot.columns]
        yearly = mdf.groupby("year")["ret"].sum()
        pivot["年度"] = yearly

        header = "| 年份 | " + " | ".join(pivot.columns) + " |"
        sep = "|------|" + "|".join(["------"] * len(pivot.columns)) + "|"
        md.append(header)
        md.append(sep)
        for year, row in pivot.iterrows():
            vals = " | ".join(f"{v:.1f}" if not pd.isna(v) else "" for v in row)
            md.append(f"| {year} | {vals} |")

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

    # Equity curve
    eq_path = os.path.join(output_dir, "fund_equity.csv")
    eq_df = pd.DataFrame(snapshots, columns=["date", "equity", "cash", "positions"])
    eq_df["date"] = eq_df["date"].astype(str).str[:10]
    eq_df.to_csv(eq_path, index=False)
    print(f"  Equity CSV: {eq_path} ({len(eq_df)} rows)")

    # Trade log
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
                        choices=["sp500", "report", "custom"],
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
    parser.add_argument("--cache-dir", type=str, default="data/",
                        help="Cache directory for OHLC data (default: data/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download (skip cache)")
    parser.add_argument("--output-dir", type=str, default="reports/",
                        help="Output directory (default: reports/)")
    args = parser.parse_args()

    strat_names = {1: "策略1(超买动量)", 2: "策略2(超卖反转)", 3: "策略1+2(全部)"}
    config_str = (f"{strat_names[args.strategy]} | {args.universe} | "
                  f"资金${args.capital:,.0f} | 最大{args.max_positions}仓 | "
                  f"止损{args.stop_loss}% | 排序:历史盈亏比")

    print("=" * 70)
    print("  韭韭量化 基金回测器")
    print("=" * 70)
    print(f"  {config_str}")
    print()

    # Step 1: Get universe
    print("[1/5] Loading universe...")
    tickers = get_universe(args.universe, args.tickers)
    print(f"  Universe: {args.universe} ({len(tickers)} tickers)")

    # Step 2: Download data
    print("\n[2/5] Downloading OHLC data...")
    data = download_data(tickers, args.start, args.cache_dir, args.no_cache)
    if not data:
        print("No data! Exiting.")
        sys.exit(1)

    # Step 3: Pre-compute
    print(f"\n[3/5] Pre-computing indicators for {len(data)} tickers...")
    jojo_cache = precompute_jojo(data)
    atr_cache = precompute_atr_pct(data)
    pf_map = compute_historical_pf(data, jojo_cache, args.strategy)
    top5 = sorted(pf_map.items(), key=lambda x: -x[1])[:5]
    print(f"  Top 5 PF: {', '.join(f'{k}({v:.2f})' for k, v in top5)}")

    # Step 4: Run simulation
    print(f"\n[4/5] Running fund simulation...")
    snapshots, trades, _ = run_fund(
        data, jojo_cache, atr_cache, pf_map,
        strategy=args.strategy,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )

    if not snapshots:
        print("No snapshots generated! Check date range.")
        sys.exit(1)

    # Step 5: Output
    print(f"\n[5/5] Generating output...")
    metrics = compute_fund_metrics(snapshots, trades, args.capital)
    print_summary(metrics, config_str)
    generate_report(metrics, trades, config_str, args.output_dir)
    export_csv(snapshots, trades, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
