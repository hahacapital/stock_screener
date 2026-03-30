"""
jojo_quant (韭韭量化) — scan stocks and commodities for jojo signals.

Strategy 1 (超买动量): jojo crosses UP through 76 today (ATR% >= 2.0)
Strategy 2 (超卖反转): jojo was below 28 and turns upward today

Usage:
    python screener.py                        # scan both strategies
    python screener.py --strategy 1 --top 20  # strategy 1 only, top 20
    python screener.py --strategy 2           # strategy 2 only
"""

import argparse
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtest import run_backtest
from indicators import compute_jojo, _rma


# ---------------------------------------------------------------------------
# Extra tickers (commodities, not in NASDAQ/NYSE lists)
# ---------------------------------------------------------------------------

EXTRA_TICKERS = [
    # Commodities futures
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F",
]

EXTRA_TICKERS_SET = set(EXTRA_TICKERS)

# ---------------------------------------------------------------------------
# Chinese names for common tickers
# ---------------------------------------------------------------------------

CN_NAMES = {
    # US mega-caps
    "AAPL": "苹果", "MSFT": "微软", "GOOGL": "谷歌", "GOOG": "谷歌",
    "AMZN": "亚马逊", "NVDA": "英伟达", "META": "Meta", "TSLA": "特斯拉",
    "BRK-A": "伯克希尔", "BRK-B": "伯克希尔", "TSM": "台积电",
    "V": "Visa", "MA": "万事达", "JNJ": "强生", "WMT": "沃尔玛",
    "JPM": "摩根大通", "PG": "宝洁", "UNH": "联合健康", "HD": "家得宝",
    "XOM": "埃克森美孚", "CVX": "雪佛龙", "KO": "可口可乐", "PEP": "百事",
    "ABBV": "艾伯维", "MRK": "默克", "LLY": "礼来", "AVGO": "博通",
    "COST": "开市客", "ADBE": "Adobe", "CRM": "Salesforce", "NKE": "耐克",
    "NFLX": "奈飞", "DIS": "迪士尼", "PYPL": "PayPal", "INTC": "英特尔",
    "AMD": "AMD", "QCOM": "高通", "TXN": "德州仪器", "AMAT": "应用材料",
    "ORCL": "甲骨文", "IBM": "IBM", "CSCO": "思科", "UBER": "优步",
    "BA": "波音", "GE": "通用电气", "CAT": "卡特彼勒", "DE": "约翰迪尔",
    "GS": "高盛", "MS": "摩根士丹利", "BAC": "美国银行", "C": "花旗",
    "WFC": "富国银行", "SCHW": "嘉信理财", "BLK": "贝莱德",
    "PLTR": "Palantir", "ANET": "Arista", "HOOD": "Robinhood",
    "MU": "美光", "RKLB": "火箭实验室", "COIN": "Coinbase",
    "SNOW": "Snowflake", "SQ": "Block", "SHOP": "Shopify",
    "SPOT": "Spotify", "SNAP": "Snapchat", "PINS": "Pinterest",
    "BABA": "阿里巴巴", "JD": "京东", "PDD": "拼多多", "BIDU": "百度",
    "NIO": "蔚来", "XPEV": "小鹏", "LI": "理想", "BILI": "哔哩哔哩",
    "TME": "腾讯音乐", "ZTO": "中通快递", "VIPS": "唯品会",
    # Commodities
    "GC=F": "黄金期货", "SI=F": "白银期货", "CL=F": "原油期货",
    "NG=F": "天然气期货", "HG=F": "铜期货", "PL=F": "铂金期货",
}


# ---------------------------------------------------------------------------
# 1. Get all tickers (stocks + extras)
# ---------------------------------------------------------------------------

def get_exchange_tickers() -> tuple[list[str], dict[str, str]]:
    """Download NASDAQ + NYSE ticker lists from NASDAQ FTP.

    Returns (sorted_tickers, name_map) where name_map maps symbol -> security name.
    """
    tickers = set()
    name_map = {}
    urls = {
        "NASDAQ": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "NYSE": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }
    for exchange, url in urls.items():
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            for line in lines[1:]:
                parts = line.split("|")
                if not parts:
                    continue
                sym = parts[0].strip()
                if not sym or " " in sym or len(sym) > 6:
                    continue
                if exchange == "NASDAQ" and len(parts) > 6 and parts[6].strip() == "Y":
                    continue
                if exchange == "NYSE" and len(parts) > 6 and parts[6].strip() == "Y":
                    continue
                if any(c in sym for c in ["$", ".", "-"]):
                    continue
                tickers.add(sym)
                if len(parts) > 1:
                    name_map[sym] = parts[1].strip()
        except Exception as e:
            print(f"[WARN] Failed to fetch {exchange} list: {e}")

    if not tickers:
        print("[WARN] Could not fetch ticker lists, using fallback S&P500 list")
        tickers = _fallback_sp500()

    return sorted(tickers), name_map


def _fallback_sp500() -> set[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        return set(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"}


# ---------------------------------------------------------------------------
# 2. Download OHLC data in batches
# ---------------------------------------------------------------------------

def download_ohlc(tickers: list[str], days: int = 120, batch_size: int = 200) -> dict[str, pd.DataFrame]:
    end = datetime.now()
    start = end - timedelta(days=days)

    all_data = {}
    total = len(tickers)

    num_batches = (total + batch_size - 1) // batch_size
    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        pct = min(100, (i + len(batch)) / total * 100)
        print(f"  [{batch_num}/{num_batches}] Downloading {len(batch)} tickers ({pct:.0f}%)...")

        try:
            df = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"\n[WARN] Batch download failed: {e}")
            continue

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            for sym in batch:
                try:
                    sub = df[sym].dropna(how="all")
                    if len(sub) < 60:
                        continue
                    sub.columns = [c.lower() for c in sub.columns]
                    if all(c in sub.columns for c in ["open", "high", "low", "close"]):
                        all_data[sym] = sub[["open", "high", "low", "close"]].copy()
                except (KeyError, Exception):
                    pass
        else:
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 60 and all(c in df.columns for c in ["open", "high", "low", "close"]):
                all_data[batch[0]] = df[["open", "high", "low", "close"]].dropna(how="all").copy()

        time.sleep(0.5)

    print(f"  Done. Downloaded data for {len(all_data)} / {total} tickers.")
    return all_data


# ---------------------------------------------------------------------------
# 3. Detect signals
# ---------------------------------------------------------------------------

FMP_API_KEY = "F5UGdTRIUvC0ioFTuUcj3zmuUpBzVsEo"
FMP_PROFILE_URL = "https://financialmodelingprep.com/stable/profile"


def _fetch_fmp_profiles(symbols: list[str]) -> dict[str, dict]:
    """Fetch company profiles from FMP (name, sector, industry, marketCap).

    Free tier: 250 requests/day. Each symbol = 1 request.
    Skip EXTRA_TICKERS (FMP won't have them).
    """
    result = {}
    stock_symbols = [s for s in symbols if s not in EXTRA_TICKERS_SET]
    for i, sym in enumerate(stock_symbols):
        try:
            resp = requests.get(
                FMP_PROFILE_URL,
                params={"symbol": sym, "apikey": FMP_API_KEY},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    item = data[0]
                    result[sym] = {
                        "companyName": item.get("companyName", ""),
                        "sector": item.get("sector", ""),
                        "industry": item.get("industry", ""),
                        "marketCap": item.get("marketCap", 0) or 0,
                        "isEtf": item.get("isEtf", False),
                    }
                    continue
            elif resp.status_code == 429:
                print(f"\n  [WARN] FMP rate limit at {i}/{len(stock_symbols)}, stopping.", end="")
                break
        except Exception:
            pass
        result[sym] = {"companyName": "", "sector": "", "industry": "", "marketCap": 0, "isEtf": False}
    return result


def enrich_signals(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    """Add company name, Chinese name, sector, industry, market cap, and backtest data.

    - Stocks: FMP API for metadata, filter market_cap >= 1B
    - EXTRA_TICKERS: skip market cap filter
    """
    if df.empty:
        return df

    df = df.copy()
    tickers = df["ticker"].tolist()

    # FMP profiles (only for stocks, not extras)
    stock_tickers = [t for t in tickers if t not in EXTRA_TICKERS_SET]
    fmp_limit = 200
    fetch_tickers = stock_tickers[:fmp_limit] if len(stock_tickers) > fmp_limit else stock_tickers
    if fetch_tickers:
        print(f"  Fetching profiles for {len(fetch_tickers)} stock tickers (FMP)...", end="", flush=True)
        profiles = _fetch_fmp_profiles(fetch_tickers)
        print(" done.")
    else:
        profiles = {}

    df["name"] = df["ticker"].apply(
        lambda s: profiles.get(s, {}).get("companyName", "") or name_map.get(s, "")
    )
    df["name"] = df["name"].str.slice(0, 40)
    df["cn_name"] = df["ticker"].apply(lambda s: CN_NAMES.get(s, ""))
    df["sector"] = df["ticker"].apply(lambda s: profiles.get(s, {}).get("sector", ""))
    df["industry"] = df["ticker"].apply(lambda s: profiles.get(s, {}).get("industry", ""))
    df["market_cap"] = df["ticker"].apply(lambda s: profiles.get(s, {}).get("marketCap", 0))

    # For extras (commodities), set sector/industry
    for idx, row in df.iterrows():
        sym = row["ticker"]
        if sym in EXTRA_TICKERS_SET:
            if sym.endswith("=F"):
                df.at[idx, "sector"] = "Commodities"
                df.at[idx, "industry"] = "Futures"

    # Market cap filter: stocks must be >= 1B, extras exempt
    is_extra = df["ticker"].isin(EXTRA_TICKERS_SET)
    df = df[(df["market_cap"] >= 1e9) | is_extra].reset_index(drop=True)

    # Filter out ETFs
    etf_tickers = set()
    for sym in df["ticker"]:
        if sym not in EXTRA_TICKERS_SET:
            profile = profiles.get(sym, {})
            if profile.get("isEtf", False):
                etf_tickers.add(sym)
    if etf_tickers:
        df = df[~df["ticker"].isin(etf_tickers)].reset_index(drop=True)

    def fmt_cap(val):
        if not val or val == 0:
            return ""
        if val >= 1e12:
            return f"{val / 1e12:.1f}T"
        if val >= 1e9:
            return f"{val / 1e9:.1f}B"
        if val >= 1e6:
            return f"{val / 1e6:.0f}M"
        return f"{val:.0f}"

    df["mkt_cap_fmt"] = df["market_cap"].apply(fmt_cap)
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    return df


def _compute_trade_metrics(trades: list) -> dict:
    """Compute backtest metrics from a list of Trade objects."""
    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pnl": 0, "pf": 0, "max_dd": 0}

    pnls = [t.pnl_pct for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

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

    return {
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_pnl": round(sum(pnls), 1),
        "pf": round(pf, 2) if pf != float("inf") else "inf",
        "max_dd": round(max_dd, 1),
    }


def _run_backtest_for_signal(sym: str, strategy: int, regime: pd.Series = None) -> dict:
    """Download full history and run backtest for a single ticker.

    Returns dict with overall + regime-specific backtest metrics.
    """
    try:
        df = yf.download(sym, start="2009-01-01", auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)
        df.columns = [c.lower() for c in df.columns]

        r1, r2 = run_backtest(sym, df)
        r = r1 if strategy == 1 else r2

        # Overall metrics
        m_all = _compute_trade_metrics(r.trades)
        result = {
            "bt_trades": m_all["trades"],
            "bt_win_rate": m_all["win_rate"],
            "bt_total_pnl": m_all["total_pnl"],
            "bt_pf": m_all["pf"],
            "bt_max_dd": m_all["max_dd"],
        }

        # Bull/bear split
        if regime is not None and r.trades:
            bull_trades, bear_trades = [], []
            for t in r.trades:
                entry_str = str(t.entry_date)[:10]
                try:
                    entry_dt = pd.Timestamp(entry_str)
                    idx = regime.index.get_indexer([entry_dt], method="ffill")
                    reg = regime.iloc[idx[0]] if idx[0] >= 0 else "unknown"
                except Exception:
                    reg = "unknown"
                if reg == "bull":
                    bull_trades.append(t)
                elif reg == "bear":
                    bear_trades.append(t)

            m_bull = _compute_trade_metrics(bull_trades)
            m_bear = _compute_trade_metrics(bear_trades)
            for prefix, m in [("bull_", m_bull), ("bear_", m_bear)]:
                result[f"{prefix}trades"] = m["trades"]
                result[f"{prefix}win_rate"] = m["win_rate"]
                result[f"{prefix}total_pnl"] = m["total_pnl"]
                result[f"{prefix}pf"] = m["pf"]
                result[f"{prefix}max_dd"] = m["max_dd"]

        return result
    except Exception:
        return {}


def get_current_regime() -> tuple[str, pd.Series]:
    """Download SPX, compute SMA(225), return (current_regime, full_regime_series)."""
    try:
        spx = yf.download("^GSPC", start="2008-01-01", auto_adjust=True, progress=False)
        if isinstance(spx.columns, pd.MultiIndex):
            spx = spx.droplevel("Ticker", axis=1)
        spx.columns = [c.lower() for c in spx.columns]
        close = spx["close"].astype(float)
        sma = close.rolling(225).mean()
        regime = pd.Series("bear", index=close.index)
        regime[close >= sma] = "bull"
        regime = regime.ffill()
        current = regime.iloc[-1]
        return current, regime
    except Exception:
        return "unknown", pd.Series(dtype=str)


def add_backtest_data(df: pd.DataFrame, strategy: int, regime: pd.Series = None) -> pd.DataFrame:
    """Add historical backtest columns (overall + bull/bear) for each ticker."""
    if df.empty:
        return df

    df = df.copy()
    tickers = df["ticker"].tolist()
    print(f"  Running backtests for {len(tickers)} tickers...", end="", flush=True)

    bt_data = {}
    for sym in tickers:
        bt_data[sym] = _run_backtest_for_signal(sym, strategy, regime)

    all_cols = ["bt_trades", "bt_win_rate", "bt_total_pnl", "bt_pf", "bt_max_dd",
                "bull_trades", "bull_win_rate", "bull_total_pnl", "bull_pf", "bull_max_dd",
                "bear_trades", "bear_win_rate", "bear_total_pnl", "bear_pf", "bear_max_dd"]
    for col in all_cols:
        df[col] = df["ticker"].apply(lambda s, c=col: bt_data.get(s, {}).get(c, ""))

    print(" done.")
    return df


def scan_signals(all_data: dict[str, pd.DataFrame], strategy: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all stocks for signals.

    strategy: "1", "2", or "all"
    Returns (strategy1_df, strategy2_df)
    """
    s1_results = []
    s2_results = []
    errors = 0
    total = len(all_data)
    run_s1 = strategy in ("1", "all")
    run_s2 = strategy in ("2", "all")

    for idx, (sym, df) in enumerate(all_data.items()):
        if (idx + 1) % 500 == 0:
            print(f"\r  Scanning... {idx + 1}/{total}", end="", flush=True)

        try:
            therm = compute_jojo(df)
            vals = therm.values
            if len(vals) < 3:
                continue

            today = vals[-1]
            yesterday = vals[-2]
            if np.isnan(today) or np.isnan(yesterday):
                continue

            last_date = (df.index[-1].strftime("%Y-%m-%d")
                         if hasattr(df.index[-1], "strftime") else str(df.index[-1]))
            last_close = float(df["close"].iloc[-1])
            close_arr = df["close"].astype(float).values

            if run_s1:
                # Compute ATR%(14) for volatility filter
                high = df["high"].astype(float).values
                low = df["low"].astype(float).values
                prev_close = np.roll(close_arr, 1)
                prev_close[0] = close_arr[0]
                tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
                atr = _rma(pd.Series(tr), 14).values
                cur_atr_pct = atr[-1] / close_arr[-1] * 100 if close_arr[-1] > 0 else 0

                # Strategy 1: cross above 76 with position state tracking
                # Includes -20% stop loss to match backtest behavior
                in_pos = False
                signal_today = False
                entry_price_s1 = 0.0
                for k in range(1, len(vals)):
                    vk = vals[k]
                    vk1 = vals[k - 1]
                    if np.isnan(vk) or np.isnan(vk1):
                        continue
                    if not in_pos:
                        if vk > 76 and vk1 <= 76:
                            in_pos = True
                            entry_price_s1 = close_arr[k]
                            if k == len(vals) - 1:
                                signal_today = True
                    else:
                        # Stop loss exit: -20%
                        if entry_price_s1 > 0 and (close_arr[k] / entry_price_s1 - 1) * 100 <= -20:
                            in_pos = False
                        # Signal exit: cross below 68
                        elif vk < 68 and vk1 >= 68:
                            in_pos = False

                # ATR% filter
                if signal_today and cur_atr_pct < 2.0:
                    signal_today = False

                if signal_today:
                    s1_results.append({
                        "ticker": sym,
                        "date": last_date,
                        "close": round(last_close, 2),
                        "jojo": round(today, 2),
                        "prev": round(yesterday, 2),
                        "atr_pct": round(cur_atr_pct, 2),
                    })

            if run_s2:
                # Strategy 2: below 28 zone, turning up with position state tracking
                # Includes -20% stop loss to match backtest behavior
                in_pos2 = False
                signal_today2 = False
                recent_low2 = np.nan
                entry_price_s2 = 0.0
                for k in range(1, len(vals)):
                    vk = vals[k]
                    vk1 = vals[k - 1]
                    if np.isnan(vk) or np.isnan(vk1):
                        continue
                    if not in_pos2:
                        if vk1 < 28 and vk > vk1:
                            in_pos2 = True
                            entry_price_s2 = close_arr[k]
                            recent_low2 = float(np.nanmin(vals[max(0, k - 20):k + 1]))
                            if k == len(vals) - 1:
                                signal_today2 = True
                    else:
                        # Stop loss exit: -20%
                        if entry_price_s2 > 0 and (close_arr[k] / entry_price_s2 - 1) * 100 <= -20:
                            in_pos2 = False
                        # Signal exits
                        elif (vk > 51 and vk1 <= 51) or (vk < 28 and vk1 >= 28):
                            in_pos2 = False

                if signal_today2:
                    s2_results.append({
                        "ticker": sym,
                        "date": last_date,
                        "close": round(last_close, 2),
                        "jojo": round(today, 2),
                        "prev": round(yesterday, 2),
                        "recent_low": round(recent_low2, 2),
                    })

        except Exception:
            errors += 1

    print(f"\r  Scanned {total} tickers ({errors} errors).                    ")

    cols1 = ["ticker", "date", "close", "jojo", "prev", "atr_pct"]
    cols2 = ["ticker", "date", "close", "jojo", "prev", "recent_low"]

    df1 = pd.DataFrame(s1_results, columns=cols1)
    df2 = pd.DataFrame(s2_results, columns=cols2)

    if not df1.empty:
        df1 = df1.sort_values("jojo", ascending=False).reset_index(drop=True)
    if not df2.empty:
        df2 = df2.sort_values("jojo", ascending=True).reset_index(drop=True)

    return df1, df2


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="韭韭量化 jojo选股")
    parser.add_argument("--strategy", type=str, default="all", choices=["1", "2", "all"],
                        help="Which strategy to scan: 1, 2, or all (default: all)")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only top N results per strategy (default: all)")
    parser.add_argument("--days", type=int, default=120,
                        help="Calendar days of history to download (default: 120)")
    parser.add_argument("--batch", type=int, default=200,
                        help="Tickers per download batch (default: 200)")
    args = parser.parse_args()

    strat_desc = {"1": "Strategy 1 only", "2": "Strategy 2 only", "all": "All strategies"}
    print("=== 韭韭量化 jojo选股 ===")
    print(f"Mode: {strat_desc[args.strategy]}")
    print()

    print("[1/6] Fetching NASDAQ + NYSE ticker list...")
    tickers, name_map = get_exchange_tickers()
    # Add extra tickers (commodities)
    all_tickers = tickers + [t for t in EXTRA_TICKERS if t not in tickers]
    print(f"  Found {len(tickers)} stocks + {len(EXTRA_TICKERS)} extras = {len(all_tickers)} total.")
    print()

    print("[2/6] Detecting market regime (SPX SMA 225)...")
    current_regime, regime_series = get_current_regime()
    regime_cn = {"bull": "牛市", "bear": "熊市", "unknown": "未知"}
    print(f"  当前市场环境: {regime_cn.get(current_regime, current_regime)}")
    print()

    print("[3/6] Downloading OHLC data...")
    all_data = download_ohlc(all_tickers, days=args.days, batch_size=args.batch)
    print()

    print("[4/6] Scanning for signals...")
    s1, s2 = scan_signals(all_data, strategy=args.strategy)

    # Enrich with company info + market cap filter
    print("\n[5/6] Fetching company info & filtering (market cap >= 1B)...")
    s1 = enrich_signals(s1, name_map)
    s2 = enrich_signals(s2, name_map)

    if args.top > 0:
        s1 = s1.head(args.top)
        s2 = s2.head(args.top)

    # Add backtest data (with bull/bear split)
    print("\n[6/6] Running historical backtests...")
    if not s1.empty:
        s1 = add_backtest_data(s1, strategy=1, regime=regime_series)
    if not s2.empty:
        s2 = add_backtest_data(s2, strategy=2, regime=regime_series)

    # Choose regime-specific columns based on current market
    if current_regime == "bull":
        regime_prefix = "bull_"
        regime_label = "牛市"
    else:
        regime_prefix = "bear_"
        regime_label = "熊市"

    regime_bt_cols = [f"{regime_prefix}trades", f"{regime_prefix}win_rate",
                      f"{regime_prefix}total_pnl", f"{regime_prefix}pf", f"{regime_prefix}max_dd"]

    # Rename regime columns for display
    regime_display_names = {
        f"{regime_prefix}trades": f"{regime_label}_trades",
        f"{regime_prefix}win_rate": f"{regime_label}_win%",
        f"{regime_prefix}total_pnl": f"{regime_label}_pnl%",
        f"{regime_prefix}pf": f"{regime_label}_pf",
        f"{regime_prefix}max_dd": f"{regime_label}_dd%",
    }

    display_cols1 = ["ticker", "name", "cn_name", "industry", "mkt_cap_fmt",
                     "close", "jojo", "atr_pct",
                     "bt_trades", "bt_win_rate", "bt_total_pnl", "bt_pf", "bt_max_dd"] + regime_bt_cols
    display_cols2 = ["ticker", "name", "cn_name", "industry", "mkt_cap_fmt",
                     "close", "jojo", "recent_low",
                     "bt_trades", "bt_win_rate", "bt_total_pnl", "bt_pf", "bt_max_dd"] + regime_bt_cols

    def _rank_and_display(df, display_cols):
        """Rank all results by market cap descending."""
        if df.empty:
            return
        cols = [c for c in display_cols if c in df.columns]
        combined = df.sort_values("market_cap", ascending=False).copy()
        out = combined[cols].rename(columns=regime_display_names)
        print(out.to_string(index=False))

    if args.strategy in ("1", "all"):
        print()
        print("=" * 140)
        print(f"Strategy 1 — 超买动量 (cross above 76 | stop 20% | ATR%≥2.0) | 当前: {regime_label}")
        print("=" * 140)
        if s1.empty:
            print("  No signals today.")
        else:
            print(f"  Found {len(s1)} signal(s)，按市值排名:\n")
            _rank_and_display(s1, display_cols1)

    if args.strategy in ("2", "all"):
        print()
        print("=" * 140)
        print(f"Strategy 2 — 超卖反转 (below 28, turning up | stop 20%) | 当前: {regime_label}")
        print("=" * 140)
        if s2.empty:
            print("  No signals today.")
        else:
            print(f"  Found {len(s2)} signal(s)，按市值排名:\n")
            _rank_and_display(s2, display_cols2)

    print()
    print("Done.")
    return s1, s2


if __name__ == "__main__":
    main()
