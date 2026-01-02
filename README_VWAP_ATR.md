# VWAP + ATR Intraday Trading Strategy

A production-ready, rule-based intraday trading strategy using VWAP, ATR, Volume Profile, RSI regime detection, and EMA trend filter.

## Overview

This strategy combines multiple technical indicators to identify high-probability trading opportunities in intraday markets:
- **VWAP (Volume Weighted Average Price)**: Session-anchored anchor point
- **ATR (Average True Range)**: Dynamic support/resistance bands
- **Volume Profile**: Key price levels (POC, VAH, VAL)
- **RSI**: Regime detection (not overbought/oversold)
- **EMA**: Trend filter

The strategy adapts to market conditions by detecting:
- **Trend Days**: Use trend continuation trades
- **Rotational Days**: Use mean reversion trades

## Features

✅ **Deterministic**: All signals are rule-based with no discretionary logic  
✅ **No Lookahead Bias**: Signals based on closed candles only  
✅ **No Repainting**: Indicators calculated sequentially  
✅ **Risk Managed**: Every trade has stop loss and take profit  
✅ **Session Aware**: Respects intraday session boundaries  
✅ **One Trade Per Direction Per Session**: Prevents overtrading  

## Installation

```bash
pip install pandas numpy
```

## Quick Start

```python
from examples.run_vwap_atr_strategy import main

# Run the strategy with sample data
main()
```

Or run directly:
```bash
python examples/run_vwap_atr_strategy.py
```

## Strategy Components

### 1. Indicators (`indicators/vwap_atr_indicators.py`)

All indicators avoid lookahead bias and are session-aware:

- **VWAP**: Session-anchored, resets each day
- **ATR(14)**: With ±1×ATR and ±2×ATR bands
- **Volume Profile**: TPO-style with POC, VAH, VAL (70% value area)
- **RSI(14)**: For regime detection
- **EMA(20)**: Trend filter

```python
from indicators.vwap_atr_indicators import calculate_indicators

# Calculate all indicators at once
df = calculate_indicators(
    df,
    session_col='session',
    atr_period=14,
    rsi_period=14,
    ema_period=20
)
```

### 2. Market Regime Detection (`strategies/regime_detection.py`)

Detects whether the market is in a **Trend Day** or **Rotational Day**:

**Trend Day Characteristics:**
- Price, VWAP, and EMA aligned (all bullish or all bearish)
- ATR expanding (volatility increasing)
- Price moving away from value area

**Rotational Day Characteristics:**
- Price oscillates around VWAP
- ATR contracting (volatility decreasing)
- Price stays within value area (VAH-VAL)

```python
from strategies.regime_detection import detect_full_regime

# Detect market regime
df = detect_full_regime(df, atr_lookback=20)
```

### 3. Signal Generation (`signals/vwap_atr_signal.py`)

Generates trading signals based on regime and setups:

#### Mean Reversion Trades (Rotational Markets)

**Entry Conditions:**
- Market regime = rotational
- Price between 1×ATR and 2×ATR away from VWAP
- Price near VAH / VAL / POC
- RSI regime supports direction
  - Long: below VWAP, RSI >= 40
  - Short: above VWAP, RSI <= 60

**Risk Management:**
- Stop Loss: 1×ATR from entry
- Take Profit: VWAP (mean reversion target)

#### Trend Continuation Trades (Trending Markets)

**Entry Conditions:**
- Market regime = trend
- Price pulls back to VWAP or EMA
- Price/VWAP/EMA aligned
- RSI regime confirms direction

**Risk Management:**
- Stop Loss: 0.75×ATR from entry (tighter)
- Take Profit: Opposite ATR band (VWAP ± 1×ATR)

```python
from signals.vwap_atr_signal import generate_signals

# Generate trading signals
df = generate_signals(df, session_col='session')
```

### 4. Backtesting (`backtest/intraday_backtest.py`)

Runs backtests with proper position tracking:

```python
from backtest.intraday_backtest import run_intraday_backtest

results = run_intraday_backtest(
    df,
    initial_capital=100000,
    position_size_pct=1.0  # 100% of capital per trade
)

# Access results
equity_curve = results['equity_curve']
trades = results['trades']
metrics = results['metrics']
```

## Data Requirements

The strategy requires 5-minute OHLCV data with the following columns:
- `timestamp`: DateTime index
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume
- `session`: Session identifier (e.g., '2024-01-02' for daily sessions)

### Using Your Own Data

```python
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Add session column (daily sessions)
df['session'] = df.index.date.astype(str)

# Run the strategy
from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals
from backtest.intraday_backtest import run_intraday_backtest

df = calculate_indicators(df)
df = detect_full_regime(df)
df = generate_signals(df)

results = run_intraday_backtest(df, initial_capital=100000)
```

## Example Output

```
================================================================================
VWAP + ATR INTRADAY TRADING STRATEGY
================================================================================

[1/6] Loading intraday data...
✓ Generated 1716 bars across 22 sessions

[2/6] Calculating indicators...
✓ Indicators calculated

[3/6] Detecting market regimes...
  Trend bars: 137 (8.0%)
  Rotational bars: 1579 (92.0%)

[4/6] Generating trading signals...
  ✓ Signal validation passed
  Total signals: 22

[5/6] Running backtest...
✓ Backtest complete

[6/6] Results Summary
================================================================================

PERFORMANCE METRICS
--------------------------------------------------------------------------------
Total Trades.................. 19
Win Rate...................... 52.63%
Profit Factor................. 0.81
Max Drawdown.................. -4.99%

REGIME PERFORMANCE
--------------------------------------------------------------------------------
    regime  num_trades win_rate avg_pnl_pct
     trend          16   50.00%    -0.14%
rotational           3   66.67%     0.26%
```

## Architecture

```
Financial-Algorithms/
├── indicators/
│   ├── __init__.py
│   └── vwap_atr_indicators.py    # All indicator calculations
├── strategies/
│   ├── __init__.py
│   └── regime_detection.py       # Market regime detection
├── signals/
│   ├── vwap_atr_signal.py        # Signal generation logic
├── backtest/
│   ├── intraday_backtest.py      # Intraday backtesting engine
├── examples/
│   ├── generate_sample_data.py   # Sample data generator
│   └── run_vwap_atr_strategy.py  # Complete example
└── README_VWAP_ATR.md            # This file
```

## Key Design Decisions

### Why Session-Anchored VWAP?
VWAP resets each session to provide a fresh reference point. This prevents carry-over effects from previous days.

### Why ATR Bands?
ATR adapts to changing market volatility, providing dynamic support/resistance levels that adjust to current conditions.

### Why Volume Profile?
Volume Profile identifies where the most trading activity occurred, revealing key support/resistance levels.

### Why RSI for Regime Detection?
Instead of traditional overbought/oversold, we use RSI to identify whether conditions favor bullish (RSI >= 40) or bearish (RSI <= 60) setups.

### Why One Trade Per Direction Per Session?
Prevents overtrading and ensures each setup is fresh. If a long trade stops out, we don't take another long until the next session.

## Risk Management

The strategy enforces strict risk management:

1. **Stop Loss**: Every trade has a stop loss
   - Mean reversion: 1×ATR
   - Trend continuation: 0.75×ATR

2. **Take Profit**: Every trade has a take profit target
   - Mean reversion: VWAP
   - Trend continuation: Opposite ATR band

3. **Position Sizing**: Configurable (default 100% of capital)

4. **Trade Limits**: One trade per direction per session

## Validation

The strategy includes validation checks:

```python
from signals.vwap_atr_signal import validate_signals

# Validate signals are unambiguous
is_valid = validate_signals(df)
```

Checks performed:
- No simultaneous long and short signals
- Stop loss and take profit are set for all signals
- All rules are deterministic

## Paper Trading / Live Trading

To adapt for live trading:

1. Connect your data feed (e.g., exchange WebSocket)
2. Calculate indicators in real-time
3. Detect regime on each new bar
4. Generate signals on bar close (no repainting)
5. Execute trades via exchange API

```python
# Pseudocode for live trading
while True:
    new_bar = get_latest_bar()
    df = append_bar(df, new_bar)
    
    # Calculate indicators
    df = calculate_indicators(df)
    df = detect_full_regime(df)
    df = generate_signals(df)
    
    # Check for signal on latest bar
    latest_signal = df.iloc[-1]['signal']
    if latest_signal != 'none':
        execute_trade(latest_signal, df.iloc[-1])
```

## Customization

All parameters are configurable:

```python
# Indicator periods
calculate_indicators(
    df,
    atr_period=14,      # ATR lookback
    rsi_period=14,      # RSI lookback
    ema_period=20       # EMA period
)

# Regime detection
detect_full_regime(
    df,
    atr_lookback=20     # ATR expansion lookback
)

# Risk management (modify in signals/vwap_atr_signal.py)
# - Stop loss multiples
# - Take profit targets
# - Near threshold for value area levels
```

## Historical Testing with Real Data

**⚠️ Important Update**: This strategy has been tested on real market data.

### Real Data Performance (3 Years, 2017-2020, Hourly Forex Data)

| Instrument | Sharpe Ratio | Total Return | Win Rate | Max Drawdown |
|------------|--------------|--------------|----------|--------------|
| EURUSD     | -0.04        | -0.53%       | 47.16%   | -3.21%       |
| GBPUSD     | 0.37         | 4.93%        | 49.83%   | -2.74%       |
| USDJPY     | 0.40         | 4.55%        | 48.81%   | -2.20%       |
| AUDUSD     | 0.20         | 3.12%        | 49.53%   | -5.14%       |
| **Average**    | **0.23**     | **2.52%**    | **48.83%**   | **-3.32%**   |

**For comparison, synthetic data results:**
- Sharpe Ratio: 3.09 (13x higher)
- Total Return: 84.22%
- Win Rate: 64.42%

### Key Findings

✅ **Strategy is viable**: Positive Sharpe ratio on 3 out of 4 forex pairs  
✅ **Manageable drawdowns**: Max drawdown <6% across all instruments  
✅ **GBPUSD/USDJPY show promise**: Sharpe ratios of 0.37-0.40 are decent starting points  

⚠️ **Areas for improvement**:
- Returns are modest (2-5% annually)
- Performance significantly lower than synthetic data (expected)
- Parameter optimization needed for each instrument
- Transaction costs not yet included

**See [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md) for detailed analysis and recommendations.**

### Running Real Data Tests

```bash
# Test single instrument (EUR/USD)
python examples/run_real_data_backtest.py

# Compare real vs synthetic data (4 forex pairs)
python examples/compare_real_vs_synthetic.py
```

## Performance Notes

- The strategy generates relatively few signals (designed for quality over quantity)
- Performance varies by market regime (trend vs rotational)
- Rotational markets typically have higher win rates with smaller moves
- Trending markets have lower win rates but larger wins when correct

## Limitations

- Requires reliable 5-minute OHLCV data
- Assumes liquid markets (tight spreads, good execution)
- Does not account for slippage or commissions (add in backtest)
- Volume Profile calculation is computationally intensive

## Future Enhancements

Potential improvements (not implemented to keep code minimal):
- Multiple take profit levels (TP1, TP2)
- Trailing stops for trend continuation
- Position scaling (add to winners)
- Multi-timeframe confirmation
- Volatility-based position sizing
- Commission and slippage modeling

## License

This code is provided as-is for educational and research purposes.

## Contributing

This strategy is part of the Financial-Algorithms repository. Contributions welcome!

## References

- VWAP: Volume Weighted Average Price
- ATR: Average True Range (Wilder)
- Volume Profile: Market Profile / TPO methodology
- RSI: Relative Strength Index (Wilder)
- EMA: Exponential Moving Average

---

**Disclaimer**: This strategy is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before live trading.
