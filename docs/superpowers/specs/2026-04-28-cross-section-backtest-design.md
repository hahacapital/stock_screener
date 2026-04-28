# Cross-Section Backtest — Design

**Date:** 2026-04-28
**Author:** brainstorm session
**Status:** Approved, awaiting implementation plan

## Goal

Cross-sectional backtest of jojo Strategy 1 (超买动量) and Strategy 2 (超卖反转): for each market regime, determine which stocks historically fit the strategy best.

Output is **descriptive** (full-history retrospective), not prescriptive forward-selection.

## Decisions

| Topic | Decision |
|-------|----------|
| Regime dimensions | SPX trend state (3) × volatility quantile (3) = 9 buckets |
| Universe | (Cache ∩ (Russell 1000 ∪ S&P 500) ∩ ≥10y history) + commodity futures |
| Trade-regime tagging | By entry-date regime (Approach A: trade-level tag) |
| Ranking metric | `score = pf × √trades`, filter `trades ≥ 5`, tiebreak by `total_pnl` then `win_rate` |
| Look-ahead constraint | Hard requirement — every regime feature uses only data ≤ current bar |
| Output | `reports/cross_section_<date>.md` + `.csv`; GitHub push only (no S3) |

## Non-Goals

- Walk-forward / out-of-sample validation (could be a follow-up project).
- Forward stock-selection recommendations.
- Optimizing strategy parameters (jojo thresholds 76/68/28/51 stay fixed).
- New backtest engine — reuse `backtest.run_backtest`.

## Architecture

### New file: `cross_section.py`

Sibling to `screener.py` / `generate_report.py`. Pure orchestration.

```
cross_section.py
├── build_universe()                      # Russell1000 ∪ S&P500 ∩ cache ∩ ≥10y
├── load_or_fetch_spx()                   # cache to data/spx.parquet
├── build_regimes(spx_df) → DataFrame     # date × {trend_state, vol_bucket, regime}
├── lookup_regime(date, regimes) → str
├── classify_trades(trades, regimes)      # tag each trade
├── aggregate(records) → DataFrame        # per (ticker, strategy, regime)
├── rank(agg) → DataFrame                 # apply score, filters, sort
├── render_markdown(ranked) → str
├── write_csv(agg, path)
├── git_push(report_paths)                # commit + push
└── main()                                # CLI
```

### Reused modules (no changes required)

- `data_loader.load_ohlc(ticker)`, `data_loader.list_universe(min_bars=...)`
- `backtest.run_backtest(symbol, df)` → `(StrategyResult, StrategyResult)`
- `indicators.compute_jojo` (indirect via `run_backtest`)
- `screener.EXTRA_TICKERS` (commodity ticker list)

### Modules NOT changed

- `download_ohlc.py` — SPX cached separately by `cross_section.py` (avoids polluting main universe cache).
- `generate_report.py`, `fund_backtest.py`, `screener.py` — untouched.

## Universe Construction

Source order:
1. `data_loader.list_universe(min_bars=2520)` — tickers with ≥10y daily bars.
2. Intersect with `Russell 1000 ∪ S&P 500` from Wikipedia (`pd.read_html`).
3. Add `screener.EXTRA_TICKERS` (commodities) unconditionally if cache has ≥10y.
4. Drop tickers whose cache parquet fails to load.

Expected universe size: ~800–1000 stocks + 6 commodities.

**Why Wikipedia membership lists**: FMP free-tier (250 req/day) cannot cover ~1500 stocks. Wikipedia gives a current snapshot; cache filter ensures the stock has long history. Survivorship bias is acknowledged but acceptable for descriptive analysis (see Risks below).

## Regime Classification

### Dimension 1 — SPX trend state (3 categories)

Inputs: SPX close, SMA50, SMA200, SMA225 (all on SPX, not per-stock).

```
trend_state =
  "bull"     if close >= SMA225 AND SMA50 >= SMA200
  "bear"     if close <  SMA225 AND SMA50 <  SMA200
  "neutral"  otherwise
```

`neutral` covers mixed-signal periods (price above SMA225 but 50<200, or vice versa).

### Dimension 2 — Volatility bucket (3 categories)

```
log_ret    = log(close).diff()
vol_30d    = log_ret.rolling(30).std() * sqrt(252)            # annualized
rank_5y    = vol_30d.rolling(252*5, min_periods=252).rank(pct=True)
vol_bucket =
  "low_vol"   if rank_5y <= 0.33
  "mid_vol"   if 0.33 < rank_5y <= 0.67
  "high_vol"  if rank_5y > 0.67
```

**Rolling 5y rank, not full-sample quantile**: Volatility regimes drift across decades. Full-sample quantile is look-ahead. Rolling 5y reflects "high/mid/low under recent context" without peeking forward.

### Combined regime label

`regime = f"{trend_state}_{vol_bucket}"` → 9 categories:
`bull_low_vol`, `bull_mid_vol`, `bull_high_vol`, `neutral_*` (×3), `bear_*` (×3).

### Warm-up

- Need ≥225 SPX bars for trend (SMA225) and ≥252 for vol rank.
- SPX cache pulled from `2008-01-01`; effective start ~2009-01-02.
- Trades whose entry date predates the warm-up cutoff are tagged `"warmup"` and excluded from the aggregate.

## Trade Tagging

```python
def lookup_regime(entry_date_str: str, regimes: pd.DataFrame) -> str:
    dt = pd.Timestamp(entry_date_str[:10])
    idx = regimes.index.get_indexer([dt], method="ffill")
    if idx[0] < 0:
        return "warmup"
    return regimes.iloc[idx[0]]["regime"]
```

`method="ffill"` is safe — it walks backward to the most recent date ≤ `dt`. Trade entries are real trading days, so this is normally a direct hit. The fallback only matters at the warm-up boundary.

Each trade contributes one row:
```
{ticker, strategy, regime, entry_date, pnl_pct, holding_days}
```

A trade that opens in `bull_low_vol` and closes in `bear_high_vol` is tagged `bull_low_vol` (entry-date regime). This matches the strategy decision point and avoids cross-regime ambiguity.

## Aggregation & Ranking

### Per-group metrics — group key `(ticker, strategy, regime)`

| Column | Definition |
|--------|------------|
| `trades` | count of trades in group |
| `win_rate` | wins / trades × 100 |
| `total_pnl` | Σ pnl_pct |
| `avg_pnl` | mean pnl_pct |
| `pf` | gross_profit / gross_loss (Σ positive / abs(Σ negative)) |
| `max_dd` | running max-drawdown of equity curve from compounding pnl_pct in chronological entry order |
| `avg_holding` | mean holding_days |
| `score` | `pf × √trades` |

### Filters

1. `trades ≥ 5` — required for main ranking.
2. `pf` finite — `pf == inf` (no losses) goes to a separate `perfect_record` table per regime/strategy.

### Sort key

`score` desc, ties broken by `total_pnl` desc, then `win_rate` desc.

## Output

### Files

- `reports/cross_section_<YYYY-MM-DD>.md` — top-N per regime per strategy.
- `reports/cross_section_<YYYY-MM-DD>.csv` — full aggregate (no top-N filter, no `trades ≥ 5` filter, so it preserves everything for ad-hoc analysis).

### Markdown structure

```
# Cross-Section Report — 2026-04-28

## Universe
Stocks: 1043  |  Commodities: 6  |  Min history: 10y
Effective period: 2009-01-02 → 2026-04-28

## Regime time distribution (trading days)
| regime          | days |
|-----------------|------|
| bull_low_vol    | 1245 |
| bull_mid_vol    |  980 |
| ...             |      |

## Strategy 1 — 超买动量

### bull_low_vol  (top 30 by score, min trades=5)
| rank | ticker | name | cn_name | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|------|---------|--------|------|------------|----|---------|-------|
| 1    | NVDA   | NVIDIA | 英伟达 | 12 | 75.0 | 234.5 | 4.2 | -18.3 | 14.55 |
| ...  |        |      |         |        |      |            |    |         |       |

### bull_mid_vol
...

(repeat for all 9 regimes with at least one qualifying row)

## Strategy 2 — 超卖反转
(same structure)
```

### CSV schema

```
ticker, strategy, regime, trades, win_rate, total_pnl, avg_pnl,
pf, max_dd, avg_holding, score
```

### CLI

```bash
python3 cross_section.py                          # default: S1+S2, top 30, push
python3 cross_section.py --strategy 1 --top 50
python3 cross_section.py --min-trades 10
python3 cross_section.py --no-push                # local only
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--strategy {1,2,all}` | `all` | Which strategies to run |
| `--top N` | `30` | Top-N rows per regime in markdown (CSV is full) |
| `--min-trades N` | `5` | Override the 5-trade filter |
| `--limit N` | `0` (off) | Restrict universe to first N tickers (smoke test only) |
| `--no-push` | False | Skip git add/commit/push |

## Look-Ahead Audit

Every component proven to use only data ≤ current bar:

| Component | Lookback only? | Why |
|-----------|----------------|-----|
| Strategy entry/exit | ✓ | `backtest.py:168,193` enter at `i+1` open; signal computed from bar `i` |
| Stop-loss | ✓ | `backtest.py:174` checks close at bar `i` after entering at `i+1` |
| SMA50/200/225 | ✓ | Uses bars ≤ t, applied to regime[t] |
| 30d realized vol | ✓ | Uses bars ≤ t |
| 5y rolling vol rank | ✓ | `rolling(min_periods=252).rank(pct=True)` only ranks within window ending at t |
| Trade-regime tag | ✓ | `regimes.get_indexer([entry_date], method="ffill")` walks backward |
| Universe filter (Wikipedia membership) | △ | Snapshot of *current* index members → survivorship bias (see Risks) |

Test `test_build_regimes_no_lookahead` enforces: truncating SPX data to date `t` must produce identical regime[t]. Any future hand-edit that introduces a centered window or full-sample quantile is caught.

## Risks

1. **Survivorship bias from Wikipedia membership.** Current Russell 1000 / S&P 500 excludes delisted names. A delisted stock that fit Strategy 1 in 2010 won't appear in our table. Acceptable for descriptive ranking ("of stocks that survived to today, which fit best?"), but does inflate apparent edge. **Mitigation**: cross-check sample size per regime; document this caveat in the report header.

2. **Small-sample noise per regime.** Some regimes (e.g. `bear_low_vol`) are rare. Even with `trades ≥ 5`, ranking can be unstable. **Mitigation**: report aggregate trade counts per regime; users see when sample is thin.

3. **Entry-date regime tagging asymmetry.** A trade opening on the last day of `bull_low_vol` and exiting deep into `bear_high_vol` is fully credited to `bull_low_vol`. Across many trades this averages out, but individual outlier trades can distort one bucket's PF. **Mitigation**: report `avg_holding` per group so users can see if trades typically span regime boundaries.

4. **`pf = inf` (no losses) edge case.** A small handful of trades all positive yields infinite PF. We separate these into a `perfect_record` listing rather than letting them top the main rank.

5. **Wikipedia scrape fragility.** `pd.read_html` on Wikipedia can break if page structure changes. **Mitigation**: cache the membership list to `data/index_members.json` after first successful fetch; fail loudly with actionable error if both scrape and cache are unavailable.

## Testing

Add to `test_logic.py`:

1. `test_build_regimes_no_lookahead` — truncating SPX to date `t` does not change `regime[t]`.
2. `test_lookup_regime_ffill` — non-trading-day inputs ffill correctly.
3. `test_score_filters_low_trades` — groups with `trades < 5` excluded from ranked output.
4. `test_pf_inf_separated` — `pf == inf` rows go to `perfect_record`, not main rank.
5. `test_regime_buckets_distinct` — every (date, regime) is one of the 9 expected labels (no malformed strings).

Smoke test: `python3 cross_section.py --limit 50 --no-push` runs end-to-end in <30s, produces non-empty markdown + csv.

Sanity checks (manual):
- Regime time distribution matches known history (bear_high_vol concentrated in 2008/2020/2022).
- NVDA / TSLA top S1 in `bull_low_vol` or `bull_mid_vol` (matches priors).
- Trade counts per (strategy, regime) bucket all ≥ 50 in aggregate; flag any bucket below.

## Performance

~1000 stocks × `run_backtest` ≈ 0.1s per stock = ~100s single-process. Acceptable. If it grows, add `multiprocessing.Pool` later. No premature parallelism.

## Documentation Updates

After implementation:

- `README.md` — add `cross_section.py` usage section, link from main commands list.
- `CLAUDE.md` — add row to "项目文件" table; add command snippet to "可用命令".

## Open Questions

None. All decisions confirmed.
