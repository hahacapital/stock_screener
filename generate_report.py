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

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import Trade, run_backtest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS = ["TSM", "GOOG", "AAPL", "TSLA", "MSFT", "PLTR", "ANET", "NVDA",
           "HOOD", "MU", "RKLB", "AMZN", "META"]
START_DATE = "2009-01-01"
SPX_SYMBOL = "^GSPC"
SMA_LENGTH = 225
REPORT_PATH = "reports/backtest_report.md"
S3_DIR = "s3://staking-ledger-bpt/jojo_quant/reports/"


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
        "num_trades": "交易次数",
        "win_rate": "胜率 (%)",
        "avg_pnl": "平均收益 (%)",
        "median_pnl": "中位收益 (%)",
        "total_pnl": "累计收益 (%)",
        "profit_factor": "盈亏比",
        "max_win": "最大单笔盈利 (%)",
        "max_loss": "最大单笔亏损 (%)",
        "max_drawdown": "最大回撤 (%)",
        "avg_holding": "平均持仓 (天)",
        "median_holding": "中位持仓 (天)",
        "min_holding": "最短持仓 (天)",
        "max_holding": "最长持仓 (天)",
        "win_streak": "最大连胜",
        "loss_streak": "最大连亏",
        "reward_risk": "收益风险比",
    }
    lines = ["| 指标 | 整体 | 牛市 | 熊市 |", "|------|------|------|------|"]
    for key, label in labels.items():
        o = fmt_metric(overall.get(key, ""))
        b = fmt_metric(bull.get(key, ""))
        be = fmt_metric(bear.get(key, ""))
        lines.append(f"| {label} | {o} | {b} | {be} |")
    return "\n".join(lines)


def per_stock_table(results: list[tuple[str, dict]]) -> str:
    """Generate per-stock summary table."""
    lines = [
        "| 股票 | 交易次数 | 胜率 | 平均收益 | 累计收益 | 最大回撤 | 平均持仓 | 盈亏比 |",
        "|------|----------|------|----------|----------|----------|----------|--------|",
    ]
    results = sorted(results, key=lambda x: x[1]['profit_factor'] if x[1]['profit_factor'] != "inf" else float("inf"), reverse=True)
    for sym, m in results:
        lines.append(
            f"| {sym} | {m['num_trades']} | {m['win_rate']}% | {m['avg_pnl']}% "
            f"| {m['total_pnl']}% | {m['max_drawdown']}% | {m['avg_holding']}天 "
            f"| {fmt_metric(m['profit_factor'])} |"
        )
    return "\n".join(lines)


def trade_detail_table(trades: list[Trade], regime: pd.Series) -> str:
    """Generate detailed trade list."""
    if not trades:
        return "*无交易*\n"
    regime_cn = {"bull": "牛市", "bear": "熊市", "unknown": "未知"}
    lines = [
        "| # | 买入 | 卖出 | 持仓天数 | 收益% | 市场环境 | 卖出原因 |",
        "|---|------|------|----------|-------|----------|----------|",
    ]
    for i, t in enumerate(trades, 1):
        r = regime_cn.get(get_trade_regime(t, regime), "未知")
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
    md.append("#### 汇总指标\n")
    md.append(metrics_table(m_all, m_bull, m_bear))
    md.append("")
    md.append("#### 个股汇总 — 整体\n")
    md.append(per_stock_table(per_stock))
    md.append("")
    md.append("#### 个股汇总 — 牛市\n")
    md.append(per_stock_table(per_stock_bull))
    md.append("")
    md.append("#### 个股汇总 — 熊市\n")
    md.append(per_stock_table(per_stock_bear))
    md.append("")


def generate_report(all_results, regime, run_date):
    """Build the full markdown report.

    all_results: list of (sym, r1, r2, r1_opt, r2a_opt, r2b_opt) tuples
    """
    md = []
    md.append(f"# 韭韭量化 回测报告\n")
    md.append(f"**生成日期**: {run_date}")
    md.append(f"**回测区间**: {START_DATE} 至今")
    md.append(f"**股票数量**: {len(all_results)}")
    md.append(f"**股票列表**: {', '.join(item[0] for item in all_results)}\n")

    md.append("## 市场环境定义\n")
    md.append(f"- **牛市 (Bull)**: S&P 500 收盘价 >= SMA({SMA_LENGTH})")
    md.append(f"- **熊市 (Bear)**: S&P 500 收盘价 < SMA({SMA_LENGTH})\n")

    # --- Strategy 1 ---
    md.append("---\n\n## 策略1: 超买动量 (上穿76买入 → 下穿68卖出 | 止损20% | ATR%≥2.0)\n")

    data_opt = _collect_strategy_section(all_results, regime, 3)  # r1_opt
    _append_strategy_block(md, "汇总", *data_opt)

    md.append("### 个股交易明细\n")
    for item in all_results:
        md.append(f"- [{item[0]}](stocks/{item[0]}.md)")
    md.append("")

    # --- Strategy 2 ---
    md.append("---\n\n## 策略2: 超卖反转 (下穿28拐头买入 → 上穿51或再穿28卖出)\n")

    data_a = _collect_strategy_section(all_results, regime, 4)  # r2a_opt
    _append_strategy_block(md, "方案A (止损20% + SPX趋势过滤)", *data_a)

    data_b = _collect_strategy_section(all_results, regime, 5)  # r2b_opt
    _append_strategy_block(md, "方案B (止损20% + 个股SMA120过滤)", *data_b)

    md.append("### 个股交易明细\n")
    for item in all_results:
        md.append(f"- [{item[0]}](stocks/{item[0]}.md)")
    md.append("")

    return "\n".join(md)


def _detail_section(md, label, r, regime):
    """Append a trade detail section for one strategy variant."""
    md.append(f"### {label}\n")
    if not r.trades:
        md.append("*无交易*\n")
        return
    m = compute_metrics(r.trades)
    md.append(f"**{m['num_trades']}笔交易 | 胜率 {m['win_rate']}% | "
              f"累计收益 {m['total_pnl']}% | 盈亏比 {fmt_metric(m['profit_factor'])}**\n")
    md.append(trade_detail_table(r.trades, regime))
    md.append("")


def generate_stock_detail(sym, r1, r2, r1_opt, r2a_opt, r2b_opt, regime, run_date):
    """Generate per-stock detail markdown with trade lists for all strategy variants."""
    md = []
    md.append(f"# {sym} 交易明细\n")
    md.append(f"**生成日期**: {run_date}")
    md.append(f"[← 返回汇总报告](../backtest_report.md)\n")

    md.append("## 策略1: 超买动量 (上穿76买入 → 下穿68卖出 | 止损20% | ATR%≥2.0)\n")
    _detail_section(md, "交易明细", r1_opt, regime)

    md.append("## 策略2: 超卖反转 (下穿28拐头买入 → 上穿51或再穿28卖出)\n")
    _detail_section(md, "方案A (止损20% + SPX趋势过滤)", r2a_opt, regime)
    _detail_section(md, "方案B (止损20% + 个股SMA120过滤)", r2b_opt, regime)

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
    os.makedirs("reports/stocks", exist_ok=True)

    # Main summary report
    report = generate_report(all_results, regime, run_date)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"  Main report: {REPORT_PATH} ({len(report)} chars)")

    # Per-stock detail files
    for sym, r1, r2, r1_opt, r2a_opt, r2b_opt in all_results:
        detail = generate_stock_detail(sym, r1, r2, r1_opt, r2a_opt, r2b_opt, regime, run_date)
        detail_path = f"reports/stocks/{sym}.md"
        with open(detail_path, "w") as f:
            f.write(detail)
        print(f"  {detail_path} ({len(detail)} chars)")

    # Step 5: Publish
    print(f"\n[5/5] Publishing...")
    if not args.no_push:
        try:
            subprocess.run(["git", "add", "reports/"], check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"Update backtest report {run_date}\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"],
                check=True, capture_output=True,
            )
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print("  Pushed to GitHub.")
        except subprocess.CalledProcessError as e:
            print(f"  Git push failed: {e.stderr.decode() if e.stderr else e}")
    else:
        print("  Skipped git push (--no-push)")

    if not args.no_s3:
        try:
            subprocess.run(
                ["aws", "s3", "sync", "reports/", S3_DIR],
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
