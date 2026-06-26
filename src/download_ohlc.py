"""Download / maintain a local cache of US daily OHLC for jojo backtesting.

Universe = NASDAQ + NYSE (via ``screener.get_exchange_tickers``) + the
commodity futures listed in ``screener.EXTRA_TICKERS``. History = yfinance
``period="max"`` (each ticker back to its IPO). Layout = one parquet per
ticker under ``data/ohlc/``, plus ``_meta.parquet`` and ``_manifest.json``.

Usage:
    python3 download_ohlc.py --init                    # full download (1-2h)
    python3 download_ohlc.py --update                  # incremental
    python3 download_ohlc.py --init --limit 50 --no-s3 # smoke test
    python3 download_ohlc.py --update --no-s3
    python3 download_ohlc.py --rebuild-meta            # rescan parquet → meta
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import data_loader as dl
from screener import EXTRA_TICKERS, get_exchange_tickers

S3_DIR = "s3://hahacapital-jp/jojo_quant/ohlc/"  # ap-northeast-1 (Tokyo); was us-east-1 staking-ledger-bpt
OHLC_COLS = dl.OHLC_COLS
DELISTED_THRESHOLD = 10  # consecutive empty updates → status='delisted'


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def get_universe() -> list[str]:
    print("Fetching NASDAQ + NYSE ticker list...")
    tickers, _ = get_exchange_tickers()
    extras = [t for t in EXTRA_TICKERS if t not in tickers]
    universe = sorted(set(tickers)) + extras
    print(f"  {len(tickers)} stocks + {len(extras)} extras = {len(universe)} total")
    return universe


# ---------------------------------------------------------------------------
# yfinance batch parsing — same logic as screener.download_ohlc
# ---------------------------------------------------------------------------

def _parse_batch(df, batch: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for sym in batch:
            try:
                sub = df[sym].dropna(how="all")
                if sub.empty:
                    continue
                sub.columns = [c.lower() for c in sub.columns]
                if all(c in sub.columns for c in OHLC_COLS):
                    out[sym] = sub[OHLC_COLS].copy()
            except (KeyError, Exception):
                pass
    else:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if all(c in df.columns for c in OHLC_COLS):
            sub = df[OHLC_COLS].dropna(how="all")
            if not sub.empty:
                out[batch[0]] = sub
    return out


SPAC_FALLBACK_START = "1980-01-01"


def _iter_batches(tickers: list[str], batch_size: int, label: str = "",
                  **yf_kwargs):
    """Yield ``(batch_num, num_batches, batch, parsed_dict)`` per batch.

    Streaming form, so callers can write parquet & free memory between
    batches instead of buffering ~10k tickers worth of DataFrames in RAM
    (which is ~10 GB peak — too big for an 8 GB box).
    """
    total = len(tickers)
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        pct = min(100, (i + len(batch)) / total * 100)
        tag = f"{label} " if label else ""
        print(f"  {tag}[{batch_num}/{num_batches}] {len(batch)} tickers ({pct:.0f}%)...",
              flush=True)
        try:
            df = yf.download(
                batch,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
                **yf_kwargs,
            )
        except Exception as e:
            print(f"    [WARN] batch failed: {e}")
            yield batch_num, num_batches, batch, {}
            continue
        parsed = _parse_batch(df, batch)
        del df
        yield batch_num, num_batches, batch, parsed
        time.sleep(0.5)


def _download_in_batches(tickers: list[str], batch_size: int, **yf_kwargs
                         ) -> dict[str, pd.DataFrame]:
    """Buffered helper for small payloads (e.g. ``cmd_update`` daily increments).

    Do NOT use this for ``--init`` over the full universe: 10k tickers × full
    history ≈ 10 GB in memory. ``cmd_init`` uses ``_iter_batches`` directly.
    """
    out: dict[str, pd.DataFrame] = {}
    for _, _, _, parsed in _iter_batches(tickers, batch_size, **yf_kwargs):
        out.update(parsed)
    return out


def download_since(tickers: list[str], start: datetime,
                   batch_size: int = 200) -> dict[str, pd.DataFrame]:
    """Single shared start date — over-pull for tickers already past it; we
    slice each ticker individually downstream. Used by ``cmd_update``.
    """
    end = datetime.now() + timedelta(days=1)
    return _download_in_batches(
        tickers, batch_size,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )


# ---------------------------------------------------------------------------
# Cache write helpers
# ---------------------------------------------------------------------------

def _meta_row(df: pd.DataFrame, status: str = "active", fail_count: int = 0) -> dict:
    return {
        "first_date": df.index[0].strftime("%Y-%m-%d"),
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "num_bars": len(df),
        "status": status,
        "fail_count": fail_count,
    }


def upsert_ticker(sym: str, new_df: pd.DataFrame) -> tuple[int, int]:
    """Merge ``new_df`` into existing cache (concat + dedupe).

    Returns ``(added_bars, total_bars)``.
    """
    if new_df is None or new_df.empty:
        try:
            existing = dl.load_ohlc(sym)
            return 0, len(existing)
        except FileNotFoundError:
            return 0, 0

    new_df = new_df[OHLC_COLS].copy()
    new_df.index = pd.DatetimeIndex(new_df.index)
    try:
        existing = dl.load_ohlc(sym)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        added = len(combined) - len(existing)
    except FileNotFoundError:
        combined = new_df.sort_index()
        added = len(combined)

    dl.save_ohlc(sym, combined)
    return added, len(combined)


def rebuild_meta_from_files() -> pd.DataFrame:
    rows = []
    for sub in [dl.STOCKS_DIR, dl.EXTRAS_DIR]:
        if not sub.exists():
            continue
        for p in sub.glob("*.parquet"):
            sym = p.stem
            try:
                df = pd.read_parquet(p)
                if df.empty:
                    continue
                rows.append({"ticker": sym, **_meta_row(df)})
            except Exception as e:
                print(f"  [WARN] rebuild meta {sym}: {e}")
    if not rows:
        return pd.DataFrame(columns=dl.META_COLS).rename_axis("ticker")
    return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def sync_to_s3(s3_dir: str = S3_DIR) -> bool:
    print(f"Syncing {dl.DATA_DIR} → {s3_dir} ...")
    try:
        subprocess.run(
            ["aws", "s3", "sync", str(dl.DATA_DIR), s3_dir],
            check=True,
        )
        print("  S3 sync done.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] S3 sync failed: {e}")
        return False
    except FileNotFoundError:
        print("  [WARN] aws CLI not installed; skipping S3 sync.")
        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _stream_save(parsed: dict, meta_rows: dict, saved_set: set) -> int:
    """Write each ticker in a parsed batch to parquet, update meta_rows in
    place, and return the count saved. Failures are logged and skipped.
    """
    n = 0
    for sym, df in parsed.items():
        try:
            dl.save_ohlc(sym, df)
            meta_rows[sym] = _meta_row(df)
            saved_set.add(sym)
            n += 1
        except Exception as e:
            print(f"    [WARN] save {sym}: {e}")
    return n


def _flush_meta(meta_rows: dict) -> None:
    """Persist current ``meta_rows`` to ``_meta.parquet`` (atomic-ish overwrite)."""
    if not meta_rows:
        return
    meta = pd.DataFrame.from_dict(meta_rows, orient="index").rename_axis("ticker")
    dl.write_meta(meta)


def cmd_init(args) -> None:
    universe = get_universe()
    if args.limit > 0:
        universe = universe[: args.limit]
        print(f"  --limit {args.limit} → using {len(universe)} tickers")

    dl.ensure_dirs()
    meta_rows: dict[str, dict] = {}
    saved_set: set[str] = set()
    saved_total = 0

    # --- Pass 1: period='max' for full history, streamed ---
    print(f"\n[1/3] Streaming download + write for {len(universe)} tickers (period='max')...")
    for batch_num, num_batches, _, parsed in _iter_batches(
        universe, args.batch, label="main", period="max"
    ):
        n = _stream_save(parsed, meta_rows, saved_set)
        saved_total += n
        _flush_meta(meta_rows)  # crash-safe: meta always reflects what's on disk
        print(f"    saved {n} this batch  |  cumulative {saved_total}/{len(universe)}",
              flush=True)
        del parsed

    # --- Pass 2: SPAC retry with explicit start date ---
    missed = sorted(set(universe) - saved_set)
    if missed:
        print(f"\n  Retry pass: {len(missed)} tickers with start={SPAC_FALLBACK_START} "
              f"(period='max' rejected)...")
        for batch_num, num_batches, _, parsed in _iter_batches(
            missed, args.batch, label="retry", start=SPAC_FALLBACK_START
        ):
            n = _stream_save(parsed, meta_rows, saved_set)
            saved_total += n
            _flush_meta(meta_rows)
            print(f"    saved {n} this batch  |  cumulative {saved_total}/{len(universe)}",
                  flush=True)
            del parsed

    failed = sorted(set(universe) - saved_set)
    print(f"\n[2/3] Done streaming. Saved {saved_total} tickers, "
          f"{len(failed)} got nothing.")
    print(f"  meta written at {dl.META_PATH}")

    now = datetime.now().isoformat(timespec="seconds")
    dl.write_manifest({
        "last_init": now,
        "last_update": now,
        "ticker_count": saved_total,
        "failed_tickers": failed,
        "delisted_tickers": [],
    })
    print(f"  Manifest written.")

    if args.no_s3:
        print("\n[3/3] Skipped S3 sync (--no-s3).")
    else:
        print("\n[3/3] Syncing to S3...")
        sync_to_s3()


def cmd_update(args) -> None:
    meta = dl.read_meta()
    if meta.empty:
        print("No cache exists yet. Run with --init first.")
        sys.exit(1)

    targets = meta.index.tolist()
    if args.limit > 0:
        targets = targets[: args.limit]

    last_dates = pd.to_datetime(meta.loc[targets, "last_date"])
    earliest = last_dates.min()
    start = earliest + pd.Timedelta(days=1)
    print(f"Earliest cached last_date: {earliest.date()} → batch start {start.date()}")
    print(f"Updating {len(targets)} tickers...")

    new_data = download_since(targets, start.to_pydatetime(), batch_size=args.batch)
    print(f"  Got new bars for {len(new_data)}/{len(targets)} tickers.")

    print("\nMerging into cache...")
    added_total = 0
    nothing: list[str] = []
    for sym in targets:
        sym_last = pd.to_datetime(meta.loc[sym, "last_date"])
        new_df = new_data.get(sym)
        if new_df is not None and not new_df.empty:
            new_df = new_df[new_df.index > sym_last]
        if new_df is None or new_df.empty:
            nothing.append(sym)
            continue
        try:
            added, total = upsert_ticker(sym, new_df)
            added_total += added
            meta.loc[sym, "last_date"] = new_df.index[-1].strftime("%Y-%m-%d")
            meta.loc[sym, "num_bars"] = total
            meta.loc[sym, "fail_count"] = 0
            meta.loc[sym, "status"] = "active"
        except Exception as e:
            print(f"  [WARN] upsert {sym}: {e}")
            nothing.append(sym)

    for sym in nothing:
        if sym in meta.index:
            cur = int(meta.loc[sym, "fail_count"]) if pd.notna(meta.loc[sym, "fail_count"]) else 0
            meta.loc[sym, "fail_count"] = cur + 1
            if meta.loc[sym, "fail_count"] >= DELISTED_THRESHOLD:
                meta.loc[sym, "status"] = "delisted"

    dl.write_meta(meta)
    delisted = sorted(meta[meta["status"] == "delisted"].index.tolist())
    manifest = dl.read_manifest()
    manifest.update({
        "last_update": datetime.now().isoformat(timespec="seconds"),
        "ticker_count": len(meta),
        "failed_tickers": nothing,
        "delisted_tickers": delisted,
    })
    dl.write_manifest(manifest)

    print(f"\nUpdate done: +{added_total} bars across "
          f"{len(targets) - len(nothing)} tickers; "
          f"{len(nothing)} got nothing; "
          f"{len(delisted)} marked delisted (fail_count ≥ {DELISTED_THRESHOLD}).")

    if not args.no_s3:
        print()
        sync_to_s3()
    else:
        print("\nSkipped S3 sync (--no-s3).")


def cmd_rebuild_meta(args) -> None:
    meta = rebuild_meta_from_files()
    dl.write_meta(meta)
    print(f"Rebuilt meta: {len(meta)} tickers → {dl.META_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download/maintain US OHLC cache")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init", action="store_true",
                      help="Full download (period='max') for the entire universe")
    mode.add_argument("--update", action="store_true",
                      help="Incremental update from each ticker's last_date")
    mode.add_argument("--rebuild-meta", action="store_true",
                      help="Rescan parquet files and rebuild _meta.parquet")

    parser.add_argument("--limit", type=int, default=0,
                        help="Limit universe to first N tickers (smoke test)")
    parser.add_argument("--batch", type=int, default=200,
                        help="Tickers per yf.download batch (default 200)")
    parser.add_argument("--no-s3", action="store_true",
                        help="Skip S3 sync after init/update")

    args = parser.parse_args()

    if args.init:
        cmd_init(args)
    elif args.update:
        cmd_update(args)
    elif args.rebuild_meta:
        cmd_rebuild_meta(args)


if __name__ == "__main__":
    main()
