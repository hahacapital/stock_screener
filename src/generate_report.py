"""
Generate comprehensive backtest report for jojo strategies.

Downloads data, runs backtests, tags bull/bear regime, computes enhanced
metrics, generates markdown report, commits to GitHub, uploads to S3.

Usage:
    python generate_report.py
    python generate_report.py --no-push --no-s3
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import Trade, run_backtest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

TICKERS = ["TSM", "GOOG", "AAPL", "TSLA", "MSFT", "PLTR", "ANET", "NVDA",
           "HOOD", "MU", "RKLB", "AMZN", "META"]
START_DATE = "2009-01-01"
SPX_SYMBOL = "^GSPC"
SMA_LENGTH = 225
REPORT_PATH = str(REPORTS_DIR / "backtest_report.md")
S3_DIR = "s3://hahacapital-jp/jojo_quant/reports/"  # ap-northeast-1 (Tokyo); was us-east-1 staking-ledger-bpt


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_data():
    """Download OHLC for all tickers + SPX."""
    print(f"[1/5] Downloading {len(TICKERS)} stocks from {START_DATE}...")
    raw = yf.download(TICKERS, start=START_DATE, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)

    stock_data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in TICKERS:
            try:
                sub = raw[sym].dropna(how="all")
                sub.columns = [c.lower() for c in sub.columns]
                if len(sub) >= 60 and all(c in sub.columns for c in ["open", "high", "low", "close"]):
                    stock_data[sym] = sub[["open", "high", "low", "close"]].copy()
                    print(f"  {sym}: {len(sub)} bars ({sub.index[0].strftime('%Y-%m-%d')} to {sub.index[-1].strftime('%Y-%m-%d')})")
                else:
                    print(f"  {sym}: SKIP (only {len(sub)} bars)")
            except Exception as e:
                print(f"  {sym}: ERROR ({e})")
    else:
        # Single ticker fallback
        raw.columns = [c.lower() for c in raw.columns]
        stock_data[TICKERS[0]] = raw[["open", "high", "low", "close"]].dropna(how="all")

    print(f"\n  Downloading SPX ({SPX_SYMBOL})...")
    spx = yf.download(SPX_SYMBOL, start=START_DATE, auto_adjust=True, progress=False)
    if isinstance(spx.columns, pd.MultiIndex):
        spx = spx.droplevel("Ticker", axis=1)
    spx.columns = [c.lower() for c in spx.columns]
    print(f"  SPX: {len(spx)} bars")

    return stock_data, spx


def build_regime(spx: pd.DataFrame) -> pd.Series:
    """Build bull/bear regime series from SPX close and SMA(225)."""
    close = spx["close"].astype(float)
    sma = close.rolling(SMA_LENGTH).mean()
    regime = pd.Series("bear", index=close.index)
    regime[close >= sma] = "bull"
    # Forward-fill to cover any gaps
    regime = regime.ffill()
    return regime


def get_trade_regime(trade: Trade, regime: pd.Series) -> str:
    """Look up market regime for a trade's entry date."""
    entry_str = str(trade.entry_date)[:10]
    try:
        entry_dt = pd.Timestamp(entry_str)
    except Exception:
        return "unknown"
    # Find nearest prior date in regime index
    idx = regime.index.get_indexer([entry_dt], method="ffill")
    if idx[0] >= 0:
        return regime.iloc[idx[0]]
    return "unknown"


# ---------------------------------------------------------------------------
# Enhanced metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[Trade]) -> dict:
    """Compute comprehensive performance metrics from a list of trades."""
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0,
            "median_pnl": 0, "profit_factor": 0, "max_win": 0, "max_loss": 0,
            "max_drawdown": 0, "avg_holding": 0, "median_holding": 0,
            "min_holding": 0, "max_holding": 0,
            "win_streak": 0, "loss_streak": 0, "reward_risk": 0,
        }

    pnls = [t.pnl_pct for t in trades]
    holdings = [t.holding_days for t in trades]
    wins = sum(1 for p in pnls if p > 0)

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    # Max drawdown (cumulative equity curve)
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Reward/risk ratio
    std = np.std(pnls) if len(pnls) > 1 else 0
    rr = np.mean(pnls) / std if std > 0 else 0

    return {
        "num_trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "avg_pnl": round(np.mean(pnls), 2),
        "median_pnl": round(float(np.median(pnls)), 2),
        "total_pnl": round(sum(pnls), 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "max_win": round(max(pnls), 2),
        "max_loss": round(min(pnls), 2),
        "max_drawdown": round(max_dd, 2),
        "avg_holding": round(np.mean(holdings), 1),
        "median_holding": round(float(np.median(holdings)), 1),
        "min_holding": min(holdings),
        "max_holding": max(holdings),
        "win_streak": max_win_streak,
        "loss_streak": max_loss_streak,
        "reward_risk": round(rr, 2),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def fmt_metric(val):
    """Format a metric value for display."""
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        if abs(val) >= 100:
            return f"{val:.1f}"
        return f"{val:.2f}"
    return str(val)


def metrics_table(overall: dict, bull: dict, bear: dict) -> str:
    """Generate a markdown table comparing Overall / Bull / Bear metrics."""
    labels = {
        "num_trades": "Trades",
        "win_rate": "Win rate (%)",
        "avg_pnl": "Avg PnL (%)",
        "median_pnl": "Median PnL (%)",
        "total_pnl": "Total PnL (%)",
        "profit_factor": "Profit factor",
        "max_win": "Max single win (%)",
        "max_loss": "Max single loss (%)",
        "max_drawdown": "Max drawdown (%)",
        "avg_holding": "Avg holding (days)",
        "median_holding": "Median holding (days)",
        "min_holding": "Min holding (days)",
        "max_holding": "Max holding (days)",
        "win_streak": "Max win streak",
        "loss_streak": "Max loss streak",
        "reward_risk": "Reward/risk ratio",
    }
    lines = ["| Metric | Overall | Bull | Bear |", "|------|------|------|------|"]
    for key, label in labels.items():
        o = fmt_metric(overall.get(key, ""))
        b = fmt_metric(bull.get(key, ""))
        be = fmt_metric(bear.get(key, ""))
        lines.append(f"| {label} | {o} | {b} | {be} |")
    return "\n".join(lines)


def per_stock_table(results: list[tuple[str, dict]]) -> str:
    """Generate per-stock summary table."""
    lines = [
        "| Stock | Trades | Win rate | Avg PnL | Total PnL | Max drawdown | Avg holding | Profit factor |",
        "|------|----------|------|----------|----------|----------|----------|--------|",
    ]
    results = sorted(results, key=lambda x: x[1]['profit_factor'] if x[1]['profit_factor'] != "inf" else float("inf"), reverse=True)
    for sym, m in results:
        lines.append(
            f"| {sym} | {m['num_trades']} | {m['win_rate']}% | {m['avg_pnl']}% "
            f"| {m['total_pnl']}% | {m['max_drawdown']}% | {m['avg_holding']} days "
            f"| {fmt_metric(m['profit_factor'])} |"
        )
    return "\n".join(lines)


def trade_detail_table(trades: list[Trade], regime: pd.Series) -> str:
    """Generate detailed trade list."""
    if not trades:
        return "*No trades*\n"
    regime_cn = {"bull": "Bull", "bear": "Bear", "unknown": "Unknown"}
    lines = [
        "| # | Entry | Exit | Holding days | PnL% | Regime | Exit reason |",
        "|---|------|------|----------|-------|----------|----------|",
    ]
    for i, t in enumerate(trades, 1):
        r = regime_cn.get(get_trade_regime(t, regime), "Unknown")
        lines.append(
            f"| {i} | {str(t.entry_date)[:10]} @ {t.entry_price} "
            f"| {str(t.exit_date)[:10]} @ {t.exit_price} "
            f"| {t.holding_days} | {t.pnl_pct:+.2f}% | {r} | {t.exit_reason} |"
        )
    return "\n".join(lines)


def _collect_strategy_section(all_results, regime, result_key):
    """Collect trades and compute metrics for a strategy variant across all stocks.

    result_key: index into each result tuple (e.g. 1 for r1, 2 for r2, etc.)
    Returns (m_all, m_bull, m_bear, per_stock, per_stock_bull, per_stock_bear)
    """
    all_trades, bull_trades, bear_trades = [], [], []
    per_stock, per_stock_bull, per_stock_bear = [], [], []

    for item in all_results:
        sym = item[0]
        r = item[result_key]
        trades = r.trades
        sym_bull, sym_bear = [], []

        for t in trades:
            reg = get_trade_regime(t, regime)
            all_trades.append(t)
            if reg == "bull":
                bull_trades.append(t)
                sym_bull.append(t)
            elif reg == "bear":
                bear_trades.append(t)
                sym_bear.append(t)

        per_stock.append((sym, compute_metrics(trades)))
        per_stock_bull.append((sym, compute_metrics(sym_bull)))
        per_stock_bear.append((sym, compute_metrics(sym_bear)))

    return (compute_metrics(all_trades), compute_metrics(bull_trades), compute_metrics(bear_trades),
            per_stock, per_stock_bull, per_stock_bear)


def _append_strategy_block(md, label, m_all, m_bull, m_bear, per_stock, per_stock_bull, per_stock_bear):
    """Append a strategy metrics block (summary + per-stock tables) to md."""
    md.append(f"### {label}\n")
    md.append("#### Summary Metrics\n")
    md.append(metrics_table(m_all, m_bull, m_bear))
    md.append("")
    md.append("#### Per-Stock Summary — Overall\n")
    md.append(per_stock_table(per_stock))
    md.append("")
    md.append("#### Per-Stock Summary — Bull\n")
    md.append(per_stock_table(per_stock_bull))
    md.append("")
    md.append("#### Per-Stock Summary — Bear\n")
    md.append(per_stock_table(per_stock_bear))
    md.append("")


def generate_report(all_results, regime, run_date):
    """Build the full markdown report.

    all_results: list of (sym, r1, r2, r1_opt, r2a_opt, r2b_opt) tuples
    """
    md = []
    md.append(f"# 韭韭量化 Backtest Report\n")
    md.append(f"**Generated**: {run_date}")
    md.append(f"**Period**: {START_DATE} to today")
    md.append(f"**Stock count**: {len(all_results)}")
    md.append(f"**Stocks**: {', '.join(item[0] for item in all_results)}\n")

    md.append("## Market Regime Definition\n")
    md.append(f"- **Bull**: S&P 500 close >= SMA({SMA_LENGTH})")
    md.append(f"- **Bear**: S&P 500 close < SMA({SMA_LENGTH})\n")

    # --- Strategy 1 ---
    md.append("---\n\n## Strategy 1: Overbought Momentum (Cross above 76 -> Cross below 68 | Stop loss 20% | ATR%>=2.0)\n")

    data_opt = _collect_strategy_section(all_results, regime, 3)  # r1_opt
    _append_strategy_block(md, "Summary", *data_opt)

    md.append("### Per-Stock Trade Details\n")
    for item in all_results:
        md.append(f"- [{item[0]}](stocks/{item[0]}.md)")
    md.append("")

    # --- Strategy 2 ---
    md.append("---\n\n## Strategy 2: Oversold Reversal (Below 28 turning up -> Cross above 51 or below 28 again)\n")

    data_a = _collect_strategy_section(all_results, regime, 4)  # r2a_opt
    _append_strategy_block(md, "Variant A (Stop loss 20% + SPX trend filter)", *data_a)

    data_b = _collect_strategy_section(all_results, regime, 5)  # r2b_opt
    _append_strategy_block(md, "Variant B (Stop loss 20% + Stock SMA120 filter)", *data_b)

    md.append("### Per-Stock Trade Details\n")
    for item in all_results:
        md.append(f"- [{item[0]}](stocks/{item[0]}.md)")
    md.append("")

    return "\n".join(md)


def _detail_section(md, label, r, regime):
    """Append a trade detail section for one strategy variant."""
    md.append(f"### {label}\n")
    if not r.trades:
        md.append("*No trades*\n")
        return
    m = compute_metrics(r.trades)
    md.append(f"**{m['num_trades']} trades | Win rate {m['win_rate']}% | "
              f"Total PnL {m['total_pnl']}% | Profit factor {fmt_metric(m['profit_factor'])}**\n")
    md.append(trade_detail_table(r.trades, regime))
    md.append("")


def generate_stock_detail(sym, r1, r2, r1_opt, r2a_opt, r2b_opt, regime, run_date):
    """Generate per-stock detail markdown with trade lists for all strategy variants."""
    md = []
    md.append(f"# {sym} Trade Details\n")
    md.append(f"**Generated**: {run_date}")
    md.append(f"[<- Back to summary report](../backtest_report.md)\n")

    md.append("## Strategy 1: Overbought Momentum (Cross above 76 -> Cross below 68 | Stop loss 20% | ATR%>=2.0)\n")
    _detail_section(md, "Trade Details", r1_opt, regime)

    md.append("## Strategy 2: Oversold Reversal (Below 28 turning up -> Cross above 51 or below 28 again)\n")
    _detail_section(md, "Variant A (Stop loss 20% + SPX trend filter)", r2a_opt, regime)
    _detail_section(md, "Variant B (Stop loss 20% + Stock SMA120 filter)", r2b_opt, regime)

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate jojo backtest report")
    parser.add_argument("--no-push", action="store_true", help="Skip git push")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    args = parser.parse_args()

    run_date = datetime.now().strftime("%Y-%m-%d")

    # Step 1: Download
    stock_data, spx = download_data()

    # Step 2: Build regime
    print(f"\n[2/5] Building market regime (SPX SMA({SMA_LENGTH}))...")
    regime = build_regime(spx)
    bull_pct = (regime == "bull").sum() / len(regime) * 100
    print(f"  Bull: {bull_pct:.1f}% | Bear: {100 - bull_pct:.1f}%")

    # Step 3: Run backtests
    print(f"\n[3/5] Running backtests on {len(stock_data)} stocks...")
    all_results = []
    for sym, df in stock_data.items():
        try:
            r1, r2, r1_opt, r2a_opt, r2b_opt = run_backtest(
                sym, df, regime=regime, optimized=True)
            all_results.append((sym, r1, r2, r1_opt, r2a_opt, r2b_opt))
            print(f"  {sym}: S1={r1.num_trades}→{r1_opt.num_trades}, "
                  f"S2={r2.num_trades}→A{r2a_opt.num_trades}/B{r2b_opt.num_trades}")
        except Exception as e:
            print(f"  {sym}: ERROR ({e})")

    # Step 4: Generate reports
    print(f"\n[4/5] Generating reports...")
    stocks_dir = REPORTS_DIR / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    # Main summary report
    report = generate_report(all_results, regime, run_date)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"  Main report: {REPORT_PATH} ({len(report)} chars)")

    # Per-stock detail files
    for sym, r1, r2, r1_opt, r2a_opt, r2b_opt in all_results:
        detail = generate_stock_detail(sym, r1, r2, r1_opt, r2a_opt, r2b_opt, regime, run_date)
        detail_path = stocks_dir / f"{sym}.md"
        with open(detail_path, "w") as f:
            f.write(detail)
        print(f"  {detail_path} ({len(detail)} chars)")

    # Step 5: Publish
    print(f"\n[5/5] Publishing...")
    if not args.no_push:
        try:
            subprocess.run(["git", "add", str(REPORTS_DIR)], check=True,
                           capture_output=True, cwd=REPO_ROOT)
            subprocess.run(
                ["git", "commit", "-m",
                 f"Update backtest report {run_date}\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"],
                check=True, capture_output=True, cwd=REPO_ROOT,
            )
            subprocess.run(["git", "push"], check=True, capture_output=True,
                           cwd=REPO_ROOT)
            print("  Pushed to GitHub.")
        except subprocess.CalledProcessError as e:
            print(f"  Git push failed: {e.stderr.decode() if e.stderr else e}")
    else:
        print("  Skipped git push (--no-push)")

    if not args.no_s3:
        try:
            subprocess.run(
                ["aws", "s3", "sync", str(REPORTS_DIR), S3_DIR],
                check=True, capture_output=True,
            )
            print(f"  Uploaded to {S3_DIR}")
        except subprocess.CalledProcessError as e:
            print(f"  S3 upload failed: {e.stderr.decode() if e.stderr else e}")
    else:
        print("  Skipped S3 upload (--no-s3)")

    print("\nDone.")


if __name__ == "__main__":
    main()
