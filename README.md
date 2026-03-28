# jojo_quant (韭韭量化) - jojo指标选股工具

每日扫描 NASDAQ + NYSE 全部股票和商品期货，基于jojo指标指标的两种策略选股：

- **策略1 (超买动量)**：上穿 76 买入，回落下穿 68 卖出（ATR%≥2.0 过滤，20%止损）
- **策略2 (超卖反转)**：下穿 28 后拐头向上买入，上穿 51 或再次下穿 28 卖出（20%止损）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 每日选股
python screener.py --strategy 1 --top 20   # 策略1
python screener.py --strategy 2 --top 20   # 策略2
python screener.py                          # 全部策略

# 历史回测
python backtest.py TSLA NVDA HOOD --years 3
python backtest.py --csv your_data.csv --use-tv  # 使用 TradingView 导出数据

# 生成回测报告
python generate_report.py

# 基金回测（Top5 组合）
python fund_backtest.py --strategy 1 --universe sp500+
python fund_backtest.py --compare --universe sp500+   # 对比4种配置
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `indicators.py` | jojo指标指标核心计算（纯 pandas / numpy，无需外部 TA 库） |
| `screener.py` | 全市场扫描（股票+期货），附带整体+当前市场环境回测数据 |
| `backtest.py` | 历史回测引擎，支持优化版策略（止损+趋势过滤+波动率过滤） |
| `generate_report.py` | 批量回测报告生成（牛熊市分别统计，推送 GitHub + S3） |
| `fund_backtest.py` | 基金组合回测（Top5 持仓，滚动PF排名选股，支持熊市减仓/ATR%动态仓位） |
| `jojo.pine` | TradingView Pine Script v6 版本，与 Python 代码完全一致 |
| `CLAUDE.md` | OpenClaw skill 配置文件 |
| `requirements.txt` | Python 依赖 |

## 扫描输出字段

每个信号包含：基本信息（ticker, 英文名, 中文名, 行业, 市值, 收盘价, jojo指标值）+ 整体历史回测（交易次数, 胜率, 累计收益, 盈亏比, 最大回撤）+ **当前市场环境回测**（根据 SPX vs SMA225 自动判断牛/熊市，只展示对应环境的历史数据）。

## 过滤规则

- 股票：市值 ≥ 1B USD，排除 ETF
- 商品期货：不做市值过滤

## 覆盖范围

- **股票**: NASDAQ + NYSE 全部（约 6000+）
- **商品期货**: 黄金(GC=F), 白银(SI=F), 原油(CL=F), 天然气(NG=F), 铜(HG=F), 铂金(PL=F)

## 基金回测 (fund_backtest)

模拟一个 Top-N 持仓基金组合，每日扫描 jojo 买入/卖出信号，自动选股、开仓、止损。

### 核心功能

1. **多种排名方式**：市值最大/中间/最小、jojo 指标值、滚动 Profit Factor
2. **历史 S&P 500 成分股**：从 Wikipedia 变更表重建当时的成分股，消除幸存者偏差
3. **SPX 基准对比**：自动计算超额收益(α)、月度热力图、回撤分析
4. **可选配置**：熊市减仓、ATR% 动态仓位、止损比例、持仓数量

### 回测结果（历史成分股，无幸存者偏差）

| 配置 | 年化% | 最大回撤% | Sharpe |
|------|-------|-----------|--------|
| Top10 市值最大 | +7.3 | 25.3 | 0.55 |
| Top10 市值中间 | +1.8 | 34.9 | 0.19 |
| Top10 市值最小 | +0.3 | 38.9 | 0.10 |
| SPX 基准 | +10.8 | 33.9 | 0.66 |

> **注意**：使用当前成分股回测会因幸存者偏差严重高估小盘股收益（Top1 市值最小从 +38.5% 降至 -13.7%）。务必使用 `--historical` 标志。

```bash
# 排名方式 × TopN 对比（推荐加 --historical）
python fund_backtest.py --rank-compare --universe sp500+ --historical

# 4 种配置对比（基线/熊市减仓/ATR%动态仓位/全部叠加）
python fund_backtest.py --compare --universe sp500+

# 单独运行
python fund_backtest.py --strategy 1 --universe sp500+ --max-positions 10 --rank-method mktcap --historical
```

## jojo指标指标详解

jojo指标是一个**复合动量震荡指标**，将 6 个子指标归一化到 0–100 后加权合成，再用 EMA 平滑。最终输出值在 0–100 之间：

- **> 76**：超买区域（策略1买入线）
- **68**：策略1卖出线
- **51**：中线（策略2卖出线）
- **< 28**：超卖区域（策略2买入区）

### 计算公式

```
index_raw = RSI × 0.1 + WR × 0.2 + CMO × 0.1 + KD × 0.3 + TSI × 0.2 + ADXRSI × 0.1
jojo指标 = EMA(index_raw, 3)
```

### 6 个子指标

#### 1. RSI — 相对强弱指数 (Relative Strength Index)

- **权重**：10%
- **参数**：`length = 14`
- **范围**：0 – 100
- **含义**：衡量近期涨幅与跌幅的相对强度。RSI > 70 通常视为超买，< 30 视为超卖。
- **公式**：
  ```
  RS = RMA(涨幅, 14) / RMA(跌幅, 14)
  RSI = 100 - 100 / (1 + RS)
  ```
  其中 RMA 是 Wilder 平滑（指数移动平均的变体，alpha = 1/length）。

#### 2. WR — 威廉指标 (Williams %R)

- **权重**：20%
- **参数**：`length = 14`
- **原始范围**：-100 – 0（标准 Williams %R）
- **归一化**：加 100 后变为 0 – 100
- **含义**：衡量收盘价在近期最高价和最低价之间的位置。值越高表示越接近区间顶部（偏强）。
- **公式**：
  ```
  WR = -100 × (最高价 - 收盘价) / (最高价 - 最低价) + 100
  ```
  其中最高价/最低价取过去 14 根 K 线的滚动最大/最小值。

> **注意**：归一化后 WR 与 Stochastic %K 的公式完全相同：`100 × (close - lowest) / (highest - lowest)`。

#### 3. CMO — 钱德动量震荡指标 (Chande Momentum Oscillator)

- **权重**：10%
- **参数**：`length = 14`
- **原始范围**：-100 – 100
- **归一化**：`(CMO + 100) / 2` 变为 0 – 100
- **含义**：类似 RSI，但直接用涨跌幅度差值与总量的比率，对动量变化更敏感。
- **公式**：
  ```
  sum_gain = SUM(涨幅, 14)
  sum_loss = SUM(跌幅, 14)
  CMO = 100 × (sum_gain - sum_loss) / (sum_gain + sum_loss)
  归一化 = (CMO + 100) / 2
  ```

#### 4. KD — 随机指标 %K (Stochastic %K)

- **权重**：30%（最高权重）
- **参数**：`length = 14`
- **范围**：0 – 100
- **含义**：衡量收盘价在近期价格通道中的相对位置。%K > 80 表示接近通道顶部（强势），< 20 表示接近底部（弱势）。这是jojo指标中权重最大的子指标，对短期价格位置变化最敏感。
- **公式**：
  ```
  %K = 100 × (收盘价 - 14日最低价) / (14日最高价 - 14日最低价)
  ```

#### 5. TSI — 真实强度指数 (True Strength Index)

- **权重**：20%
- **参数**：`short_length = 7, long_length = 14`
- **原始范围**：-1 – 1（Pine Script 的 `ta.tsi()` 返回值）
- **归一化**：`(TSI + 1) / 2 × 100` 变为 0 – 100
- **含义**：双重 EMA 平滑的动量指标，比 RSI 更平滑，能更好地反映趋势方向和强度，同时过滤掉短期噪音。
- **公式**：
  ```
  diff = close - close[1]                      # 每日价格变动
  double_smooth    = EMA(EMA(diff, 14), 7)     # 变动的双重平滑
  abs_double_smooth = EMA(EMA(|diff|, 14), 7)  # 绝对变动的双重平滑
  TSI_raw = double_smooth / abs_double_smooth   # [-1, 1]
  TSI = (TSI_raw + 1) / 2 × 100                # [0, 100]
  ```

#### 6. ADXRSI — ADX 的 RSI（方向性过滤器）

- **权重**：10%
- **参数**：`DI length = 14, ADX smoothing = 18, RSI length = 14`
- **范围**：0 – 100
- **含义**：先计算 ADX（平均趋向指数，衡量趋势强度），再对 ADX 取 RSI，最后根据 K 线方向（阳线/阴线）调整符号。这使得jojo指标能区分上涨趋势和下跌趋势中的 ADX 强度。
- **公式**：
  ```
  # 1. DMI (方向运动指标)
  +DM = max(high - high[1], 0)  当 high变动 > low变动 时
  -DM = max(low[1] - low, 0)    当 low变动 > high变动 时
  ATR = RMA(TrueRange, 14)
  +DI = 100 × RMA(+DM, 14) / ATR
  -DI = 100 × RMA(-DM, 14) / ATR

  # 2. ADX (平均趋向指数)
  DX  = 100 × |+DI - -DI| / (+DI + -DI)
  ADX = RMA(DX, 18)

  # 3. ADXRSI (方向性调整)
  sign = +1（阳线）或 -1（阴线）
  ADXRSI = (RSI(ADX, 14) × sign + 100) / 2
  ```

### 平滑方法

jojo指标中使用了两种平滑方法，均与 TradingView 的实现完全一致：

| 方法 | 公式 | 用途 |
|------|------|------|
| **RMA (Wilder 平滑)** | `rma[i] = val × (1/n) + rma[i-1] × (1 - 1/n)` | RSI 内部的涨跌幅平滑、ATR、DI、ADX |
| **EMA (指数移动平均)** | `ema[i] = val × (2/(n+1)) + ema[i-1] × (1 - 2/(n+1))` | TSI 内部的双重平滑、最终 index 的 EMA(3) |

两者均以前 N 个值的 **SMA（简单平均）** 作为种子值初始化，这是匹配 TradingView 计算结果的关键。

## 依赖

- **yfinance** — 从 Yahoo Finance 批量下载 OHLC 数据
- **pandas** — 数据处理与时间序列操作
- **numpy** — 数值计算
- **requests** — 从 NASDAQ 网站获取股票列表

> 无需 `pandas_ta` 或其他技术分析库，所有指标均从头实现。
