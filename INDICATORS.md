# Available Trading Indicators

This document provides a comprehensive list of technical indicators available for trading strategies in this repository.

## Overview

The indicators are implemented using the Python `ta` library and custom implementations. They are categorized into four main types:

1. **Trend Indicators** - Identify market direction and momentum
2. **Momentum Indicators** - Measure the speed of price changes
3. **Volatility Indicators** - Measure price fluctuation ranges
4. **Volume Indicators** - Analyze trading volume patterns

---

## Trend Technical Indicators

### Simple Moving Average (SMA)
- **Description**: Average of closing prices over a specified period
- **Usage**: Identifies support/resistance levels and trend direction
- **Documentation**: [Investopedia - SMA](https://www.investopedia.com/terms/s/sma.asp)
- **Implementation**: `ta.trend.sma_indicator()`
- **Strategies**: Moving Averages Crossover Strategy, Volume Weighted Moving Average Strategy

### Exponential Moving Average (EMA)
- **Description**: Weighted average giving more importance to recent prices
- **Usage**: More responsive to recent price changes than SMA
- **Documentation**: [Investopedia - EMA](https://www.investopedia.com/terms/e/ema.asp)
- **Implementation**: `ta.trend.ema_indicator()`
- **Strategies**: Technical Indicators Strategies

### Volume Weighted Moving Average (VWMA)
- **Description**: Moving average weighted by volume
- **Usage**: Gives more weight to periods with higher trading volume
- **Documentation**: [Investopedia - VWAP/VWMA](https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp)
- **Implementation**: Custom implementation in `volume.py`
- **Strategies**: Volume Weighted Moving Average Strategy (VWSMA)

### Moving Average Convergence Divergence (MACD)
- **Description**: Trend-following momentum indicator showing relationship between two EMAs
- **Usage**: Identifies trend changes and momentum
- **Documentation**: [Investopedia - MACD](https://www.investopedia.com/terms/m/macd.asp)
- **Implementation**: `ta.trend.macd()` and `ta.trend.macd_signal()`
- **Strategies**: Stochastic MACD Strategy

### Average Directional Movement Index (ADX)
- **Description**: Measures the strength of a trend (regardless of direction)
- **Usage**: Determines if market is trending or ranging
- **Documentation**: [Investopedia - ADX](https://www.investopedia.com/terms/a/adx.asp)
- **Implementation**: `ta.trend.adx()`
- **Strategies**: ADX Strategy, CCI ADX Strategy

### Commodity Channel Index (CCI)
- **Description**: Measures deviation from average price
- **Usage**: Identifies overbought/oversold levels and reversals
- **Documentation**: [Investopedia - CCI](https://www.investopedia.com/terms/c/commoditychannelindex.asp)
- **Implementation**: `ta.trend.cci()`
- **Strategies**: CCI ADX Strategy

### Parabolic Stop And Reverse (Parabolic SAR)
- **Description**: Provides potential stop-loss levels
- **Usage**: Identifies potential reversal points and trailing stops
- **Documentation**: [Investopedia - Parabolic SAR](https://www.investopedia.com/trading/introduction-to-parabolic-sar/)
- **Implementation**: `ta.trend.sar()`
- **Strategies**: SAR Stochastic Strategy

---

## Momentum Technical Indicators

### Relative Strength Index (RSI)
- **Description**: Measures magnitude of recent price changes (0-100 scale)
- **Usage**: Identifies overbought (>70) and oversold (<30) conditions
- **Documentation**: [Investopedia - RSI](https://www.investopedia.com/terms/r/rsi.asp)
- **Implementation**: `ta.momentum.rsi()`
- **Strategies**: RSI Strategy, Bollinger Bands RSI Strategy, OBV Bollinger Bands RSI Strategy

### Stochastic Oscillator (SR)
- **Description**: Compares closing price to price range over time
- **Components**: %K and %D lines
- **Usage**: Identifies overbought/oversold conditions and momentum
- **Documentation**: [Investopedia - Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp)
- **Implementation**: `ta.momentum.stoch()` and `ta.momentum.stoch_signal()`
- **Strategies**: SAR Stochastic Strategy, Stochastic MACD Strategy

### Williams %R (WR)
- **Description**: Momentum indicator measuring overbought/oversold levels
- **Usage**: Similar to Stochastic but scaled from 0 to -100
- **Documentation**: [Investopedia - Williams %R](https://www.investopedia.com/terms/w/williamsr.asp)
- **Implementation**: `ta.momentum.wr()`
- **Strategies**: Williams %R Stochastic Strategy

---

## Volatility Indicators

### Average True Range (ATR)
- **Description**: Measures market volatility
- **Usage**: Determines stop-loss levels and position sizing
- **Documentation**: [Investopedia - ATR](https://www.investopedia.com/terms/a/atr.asp)
- **Implementation**: `ta.volatility.average_true_range()`
- **Strategies**: Technical Indicators Strategies

### Bollinger Bands (BB)
- **Description**: Volatility bands placed above and below a moving average
- **Components**: Upper Band, Middle Band (SMA), Lower Band
- **Usage**: Identifies overbought/oversold conditions and volatility expansion/contraction
- **Documentation**: [Investopedia - Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
- **Implementation**: `ta.volatility.bollinger_hband()` and `ta.volatility.bollinger_lband()`
- **Strategies**: Bollinger Bands RSI Strategy, OBV Bollinger Bands RSI Strategy

---

## Volume Indicators

### On-Balance Volume (OBV)
- **Description**: Cumulative volume indicator based on price direction
- **Usage**: Confirms price trends and predicts potential reversals
- **Documentation**: [Investopedia - OBV](https://www.investopedia.com/terms/o/onbalancevolume.asp)
- **Implementation**: `ta.volume.on_balance_volume()`
- **Strategies**: OBV Bollinger Bands RSI Strategy

---

## Custom Indicators

### Trend Indicator
- **Description**: Custom indicator detecting market trends using SMA and standard deviation
- **Values**: 'Uptrend', 'Downtrend', 'Range'
- **Calculation**: 
  - Computes percentage change (pc) of 150-day SMA
  - Computes 150-day standard deviation (sc)
  - If pc > sc: 'Uptrend'
  - If pc < -sc: 'Downtrend'
  - Otherwise: 'Range'
- **Implementation**: Custom function in `VWMA-SMA-MeanReversion.py`
- **Strategies**: Used in almost every strategy

---

## Trading Strategies Using These Indicators

The following strategies are implemented in the repository:

1. **Moving Averages Crossover Strategy** - Uses SMA (20, 50 days) and Trend
2. **SAR Stochastic Strategy** - Uses Parabolic SAR, Stochastic Oscillator (%K, %D), and Trend
3. **Stochastic MACD Strategy** - Uses Stochastic Oscillator and MACD
4. **RSI Strategy** - Uses RSI and Trend
5. **Bollinger Bands RSI Strategy** - Uses Bollinger Bands and RSI
6. **OBV Bollinger Bands RSI Strategy** - Uses OBV, Bollinger Bands, RSI, and Trend
7. **ADX Strategy** - Uses ADX and Trend
8. **CCI ADX Strategy** - Uses CCI, ADX, and Trend (reversal strategy)
9. **Williams %R Stochastic Strategy** - Uses Williams %R
10. **Volume Weighted Moving Average Strategy (VWSMA)** - Uses VWMA and custom z-score (mean-reversion strategy)

---

## Implementation Files

- **Main Strategy Implementation**: `Financial-Algorithm Contents/Equity/Technical Indicators/technicalindicators_strategies.py`
- **Volume Indicators (Custom)**: `Financial-Algorithm Contents/Equity/Technical Indicators/volume.py`
- **VWSMA Strategy**: `Financial-Algorithm Contents/Equity/Technical Indicators/VWMA-SMA-MeanReversion.py`
- **Simple Signal Example**: `signals/sma_signal.py`

---

## Installation

To use these indicators, install the required Python library:

```bash
pip install ta
```

For a complete list of dependencies:

```bash
pip install -r python-requirements.txt
```

---

## Technical Analysis Library Reference

Most indicators are implemented using the `ta` library:
- **Documentation**: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
- **GitHub**: https://github.com/bukosabino/ta

---

## Usage Example

```python
import ta
import pandas as pd
from pandas_datareader import data as pdr

# Load price data
ticker = "SPY"
data = pdr.get_data_yahoo(ticker, "2020-01-01", "2023-01-01")

# Calculate RSI
data['RSI'] = ta.momentum.rsi(data['Close'], n=14)

# Calculate MACD
data['MACD'] = ta.trend.macd(data['Close'])
data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])

# Calculate Bollinger Bands
data['BB_High'] = ta.volatility.bollinger_hband(data['Close'])
data['BB_Low'] = ta.volatility.bollinger_lband(data['Close'])

print(data[['Close', 'RSI', 'MACD', 'BB_High', 'BB_Low']].tail())
```

---

## Additional Resources

For more detailed information about each strategy, including:
- Strategy logic and rules
- Signal generation
- Backtesting results
- Performance metrics

Please refer to the comprehensive README in `Financial-Algorithm Contents/README.md`

---

## Contributing

If you implement new indicators or strategies, please update this documentation accordingly.
