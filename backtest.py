"""
Backtest — test historical performance of jojo strategies.

Strategy 1 (超买动量):
    BUY  when jojo crosses above 76
    SELL when jojo drops below 68

Strategy 2 (超卖反转):
    BUY  when jojo was below 28 and turns upward (today > yesterday)
    SELL when jojo crosses above 51, OR drops below 28 again

Usage:
    # Backtest single stock
    python backtest.py TSLA

    # Backtest multiple stocks
    python backtest.py TSLA NVDA HOOD AAPL

    # Backtest with custom history period
    python backtest.py TSLA --years 3

    # Backtest from a TradingView CSV
    python backtest.py --csv "/tmp/BATS_TSLA, 1D_35f99.csv" --label TSLA
"""

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_jojo, _rma


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single round-trip trade."""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    holding_days: int
    pnl_pct: float  # percentage return
    exit_reason: str


@dataclass
class StrategyResult:
    """Aggregated backtest result for one strategy on one stock."""
    symbol: str
    strategy: str
    trades: list[Trade] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        return wins / len(self.trades) * 100

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.pnl_pct for t in self.trades])

    @property
    def total_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl_pct for t in self.trades)

    @property
    def avg_holding(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.holding_days for t in self.trades])

    @property
    def min_holding(self) -> int:
        if not self.trades:
            return 0
        return min(t.holding_days for t in self.trades)

    @property
    def max_holding(self) -> int:
        if not self.trades:
            return 0
        return max(t.holding_days for t in self.trades)

    @property
    def max_win(self) -> float:
        if not self.trades:
            return 0.0
        return max(t.pnl_pct for t in self.trades)

    @property
    def max_loss(self) -> float:
        if not self.trades:
            return 0.0
        return min(t.pnl_pct for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_strategy1(dates, closes, therm_vals, *,
                       stop_loss_pct=None, atr_pct=None, min_atr_pct=None,
                       opens=None) -> list[Trade]:
    """
    Strategy 1 (超买动量):
        BUY:  jojo crosses above 76 (today > 76 AND yesterday <= 76)
        SELL: jojo drops below 68 (today < 68 AND yesterday >= 68)

    Signal uses bar i close to compute jojo, entry/exit at bar i+1 open
    to avoid look-ahead bias. Stop loss checked intraday at close.

    Optional optimizations:
        stop_loss_pct: fixed stop loss percentage (e.g. 20 means -20%)
        atr_pct + min_atr_pct: skip entry if ATR% < min_atr_pct
        opens: next-bar open prices for realistic entry/exit
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_idx = None
    # If opens not provided, fall back to closes (legacy behavior)
    exec_prices = opens if opens is not None else closes

    for i in range(1, len(therm_vals)):
        t_today = therm_vals[i]
        t_yesterday = therm_vals[i - 1]

        if np.isnan(t_today) or np.isnan(t_yesterday):
            continue

        if not in_position:
            # BUY signal: cross above 76 → enter at next bar open
            if t_today > 76 and t_yesterday <= 76:
                # ATR% filter
                if min_atr_pct is not None and atr_pct is not None:
                    if np.isnan(atr_pct[i]) or atr_pct[i] < min_atr_pct:
                        continue
                # Enter at next bar open to avoid look-ahead bias
                if i + 1 >= len(therm_vals):
                    break
                in_position = True
                entry_date = str(dates[i + 1])
                entry_price = exec_prices[i + 1]
                entry_idx = i + 1
        else:
            # Stop loss check at current close (intraday stop)
            if stop_loss_pct is not None:
                pnl_now = (closes[i] / entry_price - 1) * 100
                if pnl_now <= -stop_loss_pct:
                    exit_price = closes[i]
                    pnl = pnl_now
                    trades.append(Trade(
                        entry_date=entry_date,
                        entry_price=round(entry_price, 2),
                        exit_date=str(dates[i]),
                        exit_price=round(exit_price, 2),
                        holding_days=i - entry_idx,
                        pnl_pct=round(pnl, 2),
                        exit_reason=f"止损({stop_loss_pct}%)",
                    ))
                    in_position = False
                    continue

            # SELL signal: drop below 68 → exit at next bar open
            if t_today < 68 and t_yesterday >= 68:
                if i + 1 < len(therm_vals):
                    exit_price = exec_prices[i + 1]
                    exit_date = str(dates[i + 1])
                    exit_idx = i + 1
                else:
                    exit_price = closes[i]
                    exit_date = str(dates[i])
                    exit_idx = i
                pnl = (exit_price / entry_price - 1) * 100
                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=exit_date,
                    exit_price=round(exit_price, 2),
                    holding_days=exit_idx - entry_idx,
                    pnl_pct=round(pnl, 2),
                    exit_reason="下穿68",
                ))
                in_position = False

    # Close open position at last bar
    if in_position:
        exit_price = closes[-1]
        pnl = (exit_price / entry_price - 1) * 100
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=round(entry_price, 2),
            exit_date=str(dates[-1]),
            exit_price=round(exit_price, 2),
            holding_days=len(therm_vals) - 1 - entry_idx,
            pnl_pct=round(pnl, 2),
            exit_reason="持仓中",
        ))

    return trades


def backtest_strategy2(dates, closes, therm_vals, *,
                       stop_loss_pct=None, trend_filter=None,
                       opens=None) -> list[Trade]:
    """
    Strategy 2 (超卖反转):
        BUY:  jojo was below 28, then turns up (today > yesterday, yesterday < 28)
        SELL: jojo crosses above 51 OR drops below 28 again

    Signal uses bar i close to compute jojo, entry/exit at bar i+1 open
    to avoid look-ahead bias. Stop loss checked intraday at close.

    Optional optimizations:
        stop_loss_pct: fixed stop loss percentage (e.g. 20 means -20%)
        trend_filter: boolean array, only enter when True
        opens: next-bar open prices for realistic entry/exit
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_idx = None
    exec_prices = opens if opens is not None else closes

    for i in range(1, len(therm_vals)):
        t_today = therm_vals[i]
        t_yesterday = therm_vals[i - 1]

        if np.isnan(t_today) or np.isnan(t_yesterday):
            continue

        if not in_position:
            # BUY: was below 28 and turning up → enter at next bar open
            if t_yesterday < 28 and t_today > t_yesterday:
                # Trend filter
                if trend_filter is not None and not trend_filter[i]:
                    continue
                if i + 1 >= len(therm_vals):
                    break
                in_position = True
                entry_date = str(dates[i + 1])
                entry_price = exec_prices[i + 1]
                entry_idx = i + 1
        else:
            # Stop loss check at current close (intraday stop)
            if stop_loss_pct is not None:
                pnl_now = (closes[i] / entry_price - 1) * 100
                if pnl_now <= -stop_loss_pct:
                    exit_price = closes[i]
                    pnl = pnl_now
                    trades.append(Trade(
                        entry_date=entry_date,
                        entry_price=round(entry_price, 2),
                        exit_date=str(dates[i]),
                        exit_price=round(exit_price, 2),
                        holding_days=i - entry_idx,
                        pnl_pct=round(pnl, 2),
                        exit_reason=f"止损({stop_loss_pct}%)",
                    ))
                    in_position = False
                    continue

            # SELL: cross above 51 → exit at next bar open
            if t_today > 51 and t_yesterday <= 51:
                if i + 1 < len(therm_vals):
                    exit_price = exec_prices[i + 1]
                    exit_date = str(dates[i + 1])
                    exit_idx = i + 1
                else:
                    exit_price = closes[i]
                    exit_date = str(dates[i])
                    exit_idx = i
                pnl = (exit_price / entry_price - 1) * 100
                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=exit_date,
                    exit_price=round(exit_price, 2),
                    holding_days=exit_idx - entry_idx,
                    pnl_pct=round(pnl, 2),
                    exit_reason="上穿51",
                ))
                in_position = False
            # SELL: drop below 28 again → exit at next bar open
            elif t_today < 28 and t_yesterday >= 28:
                if i + 1 < len(therm_vals):
                    exit_price = exec_prices[i + 1]
                    exit_date = str(dates[i + 1])
                    exit_idx = i + 1
                else:
                    exit_price = closes[i]
                    exit_date = str(dates[i])
                    exit_idx = i
                pnl = (exit_price / entry_price - 1) * 100
                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=exit_date,
                    exit_price=round(exit_price, 2),
                    holding_days=exit_idx - entry_idx,
                    pnl_pct=round(pnl, 2),
                    exit_reason="再次下穿28",
                ))
                in_position = False

    # Close open position at last bar
    if in_position:
        exit_price = closes[-1]
        pnl = (exit_price / entry_price - 1) * 100
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=round(entry_price, 2),
            exit_date=str(dates[-1]),
            exit_price=round(exit_price, 2),
            holding_days=len(therm_vals) - 1 - entry_idx,
            pnl_pct=round(pnl, 2),
            exit_reason="持仓中",
        ))

    return trades


# ---------------------------------------------------------------------------
# Run backtest on one stock
# ---------------------------------------------------------------------------

def _compute_atr_pct(df: pd.DataFrame, closes: np.ndarray, start: int) -> np.ndarray:
    """Compute ATR%(14) = ATR(14) / Close * 100 using Wilder's RMA."""
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = _rma(pd.Series(tr), 14).values
    atr_pct = atr / close * 100
    return atr_pct[start:]


def _build_regime_filter(dates, regime: pd.Series) -> np.ndarray:
    """Build boolean array: True where regime is 'bull'."""
    result = np.ones(len(dates), dtype=bool)
    if regime is None:
        return result
    for i, d in enumerate(dates):
        try:
            dt = pd.Timestamp(str(d)[:10])
            idx = regime.index.get_indexer([dt], method="ffill")
            if idx[0] >= 0 and regime.iloc[idx[0]] != "bull":
                result[i] = False
        except Exception:
            pass
    return result


def _build_sma_filter(closes: np.ndarray, period: int = 50) -> np.ndarray:
    """Build boolean array: True where close >= SMA(period)."""
    sma = pd.Series(closes).rolling(period).mean().values
    return closes >= sma


def run_backtest(symbol: str, df: pd.DataFrame, therm_vals: np.ndarray = None,
                 *, regime: pd.Series = None, optimized: bool = False):
    """
    Run both strategies on a single stock.

    Parameters
    ----------
    symbol : ticker symbol (for display)
    df : OHLC DataFrame
    therm_vals : optional pre-computed jojo values (e.g. from TV CSV).
                 If None, computed from df using our formula.
    regime : optional SPX bull/bear regime Series (for optimized mode)
    optimized : if True, also run optimized versions and return 5 results

    Returns
    -------
    If optimized=False: (r1, r2)
    If optimized=True: (r1, r2, r1_opt, r2a_opt, r2b_opt)
    """
    if therm_vals is None:
        therm_vals = compute_jojo(df).values

    closes = df["close"].astype(float).values
    opens = df["open"].astype(float).values if "open" in df.columns else None
    dates = df.index if hasattr(df.index, 'strftime') else range(len(df))

    # Skip warmup period
    start = 0
    for i in range(len(therm_vals)):
        if not np.isnan(therm_vals[i]):
            start = i
            break

    dates = dates[start:]
    closes = closes[start:]
    therm_vals = therm_vals[start:]
    if opens is not None:
        opens = opens[start:]

    # Original strategies (entry/exit at next bar open to avoid look-ahead bias)
    trades1 = backtest_strategy1(dates, closes, therm_vals, opens=opens)
    trades2 = backtest_strategy2(dates, closes, therm_vals, opens=opens)

    r1 = StrategyResult(symbol=symbol, strategy="超买动量 (76→68)", trades=trades1)
    r2 = StrategyResult(symbol=symbol, strategy="超卖反转 (28→51)", trades=trades2)

    if not optimized:
        return r1, r2

    # Optimized strategies
    atr_pct = _compute_atr_pct(df, closes, start)
    regime_filter = _build_regime_filter(dates, regime)
    sma_filter = _build_sma_filter(closes, 120)

    # S1 optimized: stop loss + ATR% filter
    trades1_opt = backtest_strategy1(dates, closes, therm_vals,
                                     stop_loss_pct=20, atr_pct=atr_pct, min_atr_pct=2.0,
                                     opens=opens)
    # S2-A optimized: stop loss + SPX regime filter
    trades2a_opt = backtest_strategy2(dates, closes, therm_vals,
                                      stop_loss_pct=20, trend_filter=regime_filter,
                                      opens=opens)
    # S2-B optimized: stop loss + stock SMA(50) filter
    trades2b_opt = backtest_strategy2(dates, closes, therm_vals,
                                      stop_loss_pct=20, trend_filter=sma_filter,
                                      opens=opens)

    r1_opt = StrategyResult(symbol=symbol, strategy="超买动量优化 (止损20%+ATR≥2%)", trades=trades1_opt)
    r2a_opt = StrategyResult(symbol=symbol, strategy="超卖反转优化A (止损20%+SPX趋势)", trades=trades2a_opt)
    r2b_opt = StrategyResult(symbol=symbol, strategy="超卖反转优化B (止损20%+个股SMA120)", trades=trades2b_opt)

    return r1, r2, r1_opt, r2a_opt, r2b_opt


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(r: StrategyResult):
    """Pretty-print a strategy result."""
    print(f"\n{'=' * 65}")
    print(f"  {r.symbol} — {r.strategy}")
    print(f"{'=' * 65}")

    if r.num_trades == 0:
        print("  No trades.")
        return

    print(f"  Trades: {r.num_trades}  |  Win rate: {r.win_rate:.1f}%  "
          f"|  Avg PnL: {r.avg_pnl:+.2f}%  |  Total PnL: {r.total_pnl:+.2f}%")
    print(f"  Holding: avg {r.avg_holding:.1f}d / min {r.min_holding}d / max {r.max_holding}d  |  "
          f"Max win: {r.max_win:+.2f}%  |  Max loss: {r.max_loss:+.2f}%  |  "
          f"Profit factor: {r.profit_factor:.2f}")
    print()

    # Trade list
    rows = []
    for t in r.trades:
        rows.append({
            "entry_date": t.entry_date[:10],
            "entry_price": t.entry_price,
            "exit_date": t.exit_date[:10],
            "exit_price": t.exit_price,
            "days": t.holding_days,
            "pnl%": f"{t.pnl_pct:+.2f}",
            "exit_reason": t.exit_reason,
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="jojo backtest")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to backtest")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to TradingView CSV (uses jojo column directly)")
    parser.add_argument("--label", type=str, default=None,
                        help="Symbol label for CSV mode (default: filename)")
    parser.add_argument("--years", type=int, default=2,
                        help="Years of history for yfinance download (default: 2)")
    parser.add_argument("--use-tv", action="store_true",
                        help="When using --csv, use the jojo column instead of computing our own")
    args = parser.parse_args()

    if not args.symbols and not args.csv:
        parser.print_help()
        sys.exit(1)

    # --- CSV mode ---
    if args.csv:
        label = args.label or args.csv.split("/")[-1][:20]
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)

        # Normalize columns
        cols = [c.strip().lower() for c in df.columns]
        seen = {}
        for i, c in enumerate(cols):
            if c in seen:
                seen[c] += 1
                cols[i] = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
        df.columns = cols

        therm_vals = None
        if args.use_tv and "jojo" in df.columns:
            therm_vals = df["jojo"].values
            print(f"  Using TV jojo column ({np.sum(~np.isnan(therm_vals))} values)")
        else:
            print("  Computing jojo from OHLC...")

        r1, r2 = run_backtest(label, df, therm_vals)
        print_result(r1)
        print_result(r2)
        return

    # --- yfinance mode ---
    for sym in args.symbols:
        print(f"\nDownloading {sym} ({args.years}y history)...")
        try:
            df = yf.download(sym, period=f"{args.years}y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                print(f"  [SKIP] Not enough data for {sym}")
                continue
            # yfinance may return MultiIndex columns even for single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            df.columns = [c.lower() for c in df.columns]
            print(f"  Got {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            continue

        r1, r2 = run_backtest(sym, df)
        print_result(r1)
        print_result(r2)


if __name__ == "__main__":
    main()
