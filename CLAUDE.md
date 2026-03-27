# jojo_quant (韭韭量化) - 温度计选股工具

基于温度计 (Thermometer) 复合动量指标的全市场扫描工具。扫描范围覆盖 NASDAQ + NYSE 全部股票、主要加密货币和商品期货。

## 可用命令

### 每日扫描

```bash
# 扫描策略1信号（超买动量）
python3 screener.py --strategy 1

# 扫描策略2信号（超卖反转）
python3 screener.py --strategy 2

# 扫描全部策略
python3 screener.py --strategy all

# 限制返回数量
python3 screener.py --strategy 1 --top 20
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--strategy` | `all` | `1`=超买动量, `2`=超卖反转, `all`=全部 |
| `--top N` | 全部 | 只显示前 N 个结果 |
| `--days N` | 120 | 下载 N 天历史数据用于信号检测 |
| `--batch N` | 200 | 每批下载的 ticker 数量 |

### 历史回测

```bash
# 回测单只或多只股票
python3 backtest.py TSLA NVDA HOOD --years 3

# 使用 TradingView CSV 数据回测
python3 backtest.py --csv data.csv --use-tv --label TSLA
```

### 生成回测报告

```bash
# 生成13只股票的完整回测报告（自动推送GitHub和S3）
python3 generate_report.py

# 只生成不推送
python3 generate_report.py --no-push --no-s3
```

## 策略说明

### 策略1: 超买动量
- **买入**: 温度计上穿 76
- **卖出**: 温度计下穿 68
- **过滤**: ATR%(14) >= 2.0（过滤低波动股）
- **止损**: -20%
- **适合**: 高波动股票（TSLA, NVDA, RKLB 等）

### 策略2: 超卖反转
- **买入**: 温度计在 28 以下拐头向上
- **卖出**: 温度计上穿 51，或再次下穿 28
- **止损**: -20%
- **适合**: 超卖反弹机会

## 输出格式

每个信号包含以下字段：

| 字段 | 说明 |
|------|------|
| ticker | 标的代码 |
| name | 英文名称 |
| cn_name | 中文名称（常见标的） |
| industry | 行业分类 |
| mkt_cap_fmt | 市值（格式化） |
| close | 最新收盘价 |
| thermometer | 温度计当前值 |
| atr_pct | ATR%（仅策略1） |
| bt_trades | 历史回测交易次数（2009年至今） |
| bt_win_rate | 历史胜率% |
| bt_total_pnl | 历史累计收益% |
| bt_pf | 盈亏比 (Profit Factor) |
| bt_max_dd | 最大回撤% |
| {regime}_trades | 当前市场环境下的交易次数 |
| {regime}_win% | 当前市场环境下的胜率 |
| {regime}_pnl% | 当前市场环境下的累计收益 |
| {regime}_pf | 当前市场环境下的盈亏比 |
| {regime}_dd% | 当前市场环境下的最大回撤 |

> **市场环境判断**: SPX 收盘价 >= SMA(225) 为牛市，< SMA(225) 为熊市。输出中 {regime} 会根据当前 SPX 状态自动替换为"牛市"或"熊市"，只展示当前环境对应的数据。

## 过滤规则

- 股票：市值 >= 1B USD，排除 ETF
- 加密货币和商品期货：不做市值过滤

## 覆盖范围

- **股票**: NASDAQ + NYSE 全部（约 6000+）
- **加密货币**: BTC, ETH, SOL, XRP, BNB, ADA, DOGE 等 30+ 主流币种
- **商品期货**: 黄金(GC=F), 白银(SI=F), 原油(CL=F), 天然气(NG=F), 铜(HG=F), 铂金(PL=F)

## 项目文件

| 文件 | 说明 |
|------|------|
| `screener.py` | 全市场扫描工具 |
| `backtest.py` | 历史回测引擎 |
| `indicators.py` | 温度计指标计算（纯 pandas/numpy） |
| `generate_report.py` | 批量回测报告生成 |
| `thermometer.pine` | TradingView Pine Script 版本 |

## 依赖

```bash
pip install yfinance pandas numpy requests
```
