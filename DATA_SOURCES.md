# Data Sources and Backtesting Methodology

## Overview

The VWAP + ATR strategy has been tested using **synthetic data** to demonstrate its functionality and characteristics. The synthetic data is designed to replicate realistic market behavior while allowing for reproducible testing.

## Data Sources

### Current Implementation: Synthetic Data

All backtests in this repository use **synthetically generated OHLCV data** created by the following scripts:

1. **`examples/generate_sample_data.py`**
   - Generates 5-minute intraday bars
   - Used by `examples/run_vwap_atr_strategy.py`
   - Creates realistic intraday patterns with trending and rotational behavior

2. **`examples/run_hourly_backtest.py`**
   - Generates hourly bars for 3-year period (2021-2024)
   - Includes momentum, trend cycles, and volatility
   - ~5,400 bars across ~780 trading sessions

3. **`examples/compare_timeframes.py`**
   - Generates data for multiple timeframes (5min, 15min, 30min, 1hr, 2hr, 4hr)
   - Adjusts volatility appropriately per timeframe
   - Tests strategy across different time horizons

### Synthetic Data Characteristics

The synthetic data generator creates realistic market behavior:

**Price Dynamics:**
- Random walk with configurable volatility
- Momentum/autocorrelation (30% from previous bar)
- Longer-term trend cycles (sine wave patterns)
- Mean reversion around session averages

**OHLC Relationships:**
- Valid high/low relationships (high ≥ open/close, low ≤ open/close)
- Realistic intrabar ranges (1.5-3x open-close range)
- Proper candlestick formations

**Volume:**
- Higher volume during market open/close
- Volume increases with price volatility
- Realistic volume distributions

**Session Structure:**
- Daily sessions (9:30 AM - 4:00 PM EST)
- Weekend exclusion
- Proper session boundaries for VWAP calculation

### Why Synthetic Data?

1. **Reproducibility**: Same data generates consistent results for testing
2. **No External Dependencies**: No need for API keys or data subscriptions
3. **Full Control**: Can create specific market conditions for testing
4. **Educational Purpose**: Demonstrates strategy mechanics clearly
5. **Fast Iteration**: Quick to generate without download delays

### Limitations of Synthetic Data

⚠️ **Important Considerations:**

- **Not Real Market Data**: Does not capture actual market microstructure
- **Simplified Behavior**: Real markets have more complexity (gaps, news events, liquidity changes)
- **Optimistic Results**: Synthetic data may not include adverse conditions
- **No Slippage/Commissions**: Perfect execution assumed
- **Pattern Recognition**: Strategy not overfit to specific patterns

## Using Real Data

### Adapting for Real Market Data

The strategy is designed to work with any OHLCV data source. To use real data:

```python
import pandas as pd
from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals
from backtest.intraday_backtest import run_intraday_backtest

# Load your data (example with CSV)
df = pd.read_csv('your_real_data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Required columns: open, high, low, close, volume
# Add session column (for daily sessions)
df['session'] = df.index.date.astype(str)

# Run strategy
df = calculate_indicators(df)
df = detect_full_regime(df)
df = generate_signals(df)

# Backtest
results = run_intraday_backtest(df, initial_capital=100000)
```

### Recommended Data Sources for Real Testing

**Free Sources:**
1. **Yahoo Finance** (`yfinance` library)
   - Free intraday data (limited history)
   - Easy Python integration
   ```python
   import yfinance as yf
   ticker = yf.Ticker("SPY")
   df = ticker.history(period="1mo", interval="5m")
   ```

2. **Alpha Vantage**
   - Free API with rate limits
   - Intraday and daily data
   - Requires API key

3. **Polygon.io**
   - Free tier available
   - High-quality data
   - Good for stocks, forex, crypto

**Paid Sources (for serious backtesting):**
1. **Interactive Brokers** - Historical data API
2. **Norgate Data** - Survivorship-bias free data
3. **QuantConnect** - Cloud-based backtesting with data
4. **Databento** - Professional market data

### Data Requirements

For accurate backtesting, ensure your data has:

✅ **Clean OHLCV**: No missing values or invalid bars  
✅ **Proper Timestamps**: Sorted chronologically  
✅ **Session Boundaries**: Clear session starts/ends for VWAP  
✅ **Sufficient History**: At least 20+ bars for indicator warmup  
✅ **Consistent Timeframe**: No gaps in expected bars  
✅ **Realistic Volume**: Required for VWAP and Volume Profile  

## Backtest Results Interpretation

### Synthetic Data Results

**Current Performance (Synthetic Data):**

| Timeframe | Sharpe Ratio | Total Return | Win Rate | Max DD | Trades/Year |
|-----------|--------------|--------------|----------|--------|-------------|
| 5-minute  | ~6.5-7.0     | ~100%        | ~78%     | ~1%    | ~160        |
| 15-minute | ~5.0-5.5     | ~115%        | ~75%     | ~2%    | ~155        |
| 30-minute | ~3.8-4.2     | ~105%        | ~70%     | ~2%    | ~130        |
| 1-hour    | ~2.3-2.8     | ~70-100%     | ~63%     | ~5%    | ~110        |
| 2-hour    | ~2.5-3.0     | ~130%        | ~68%     | ~6%    | ~85         |
| 4-hour    | ~1.8-2.0     | ~50%         | ~65%     | ~5%    | ~40         |

**Best Timeframe**: 5-minute bars show highest Sharpe ratio (~7.0) with synthetic data

### Expected Results with Real Data

⚠️ **Realistic Expectations for Real Market Data:**

Based on industry standards for intraday mean reversion strategies:

- **Sharpe Ratio**: 1.0-2.5 (good to excellent)
- **Win Rate**: 55-65% (realistic for mean reversion)
- **Max Drawdown**: 5-15% (normal range)
- **Profit Factor**: 1.3-2.0 (sustainable)
- **Total Return**: 15-40% annually (after costs)

Real data will likely show:
- Lower Sharpe ratios than synthetic data
- More consecutive losses
- Larger drawdowns
- Impact of slippage and commissions
- Market regime changes (trending vs. choppy periods)

## Performance Monitoring

### Recommended Metrics to Track

1. **Risk-Adjusted Returns**
   - Sharpe Ratio (>1.5 is good)
   - Sortino Ratio (downside risk)
   - Calmar Ratio (return/max drawdown)

2. **Trade Statistics**
   - Win rate by regime (trend vs. rotational)
   - Average trade duration
   - Trades per day/week
   - Profit factor

3. **Drawdown Analysis**
   - Maximum drawdown
   - Average drawdown
   - Recovery time

4. **Consistency**
   - Monthly returns distribution
   - Rolling Sharpe ratio
   - Stability across market conditions

## Validation Approach

### Before Live Trading

1. **Backtest on Real Data**: Test with at least 1-2 years of real market data
2. **Walk-Forward Analysis**: Test on out-of-sample periods
3. **Paper Trading**: Run strategy in real-time without real money
4. **Slippage/Commission**: Add realistic costs to backtest
5. **Different Market Conditions**: Test in trending and choppy markets
6. **Multiple Instruments**: Test on different stocks/ETFs

### Risk Checks

Before deploying with real capital:
- ✅ Maximum position size per trade
- ✅ Daily loss limits
- ✅ Circuit breakers for unusual behavior
- ✅ Connection monitoring
- ✅ Order execution verification

## Conclusion

The current implementation uses **synthetic data for demonstration purposes**. While the synthetic results are promising (Sharpe ~4-7), **real market performance will differ**.

**For production use:**
1. Test with real historical data first
2. Paper trade for at least 1-2 months
3. Start with small position sizes
4. Monitor performance closely
5. Adjust parameters based on real results

The strategy's adaptive regime detection and risk management make it suitable for real markets, but proper validation is essential.

---

**Data Source Summary:**
- **Current**: Synthetic/Generated data (for demonstration)
- **Recommended**: Real market data from brokers or data providers
- **Expectation**: Real Sharpe ratios will be 1.5-3.0 (lower than synthetic)
- **Best Timeframe**: 5-30 minute bars (based on synthetic testing)
