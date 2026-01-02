# Quick Reference: Trading Indicators

## Summary

This repository provides **14 technical indicators** across 4 categories for building trading strategies.

---

## At a Glance

### 📈 Trend Indicators (7)
| Indicator | Abbreviation | Key Use Case |
|-----------|-------------|--------------|
| Simple Moving Average | SMA | Trend direction, support/resistance |
| Exponential Moving Average | EMA | Responsive trend tracking |
| Volume Weighted Moving Average | VWMA | Volume-weighted trends |
| Moving Average Convergence Divergence | MACD | Momentum and trend changes |
| Average Directional Movement Index | ADX | Trend strength measurement |
| Commodity Channel Index | CCI | Overbought/oversold detection |
| Parabolic Stop And Reverse | SAR | Stop-loss and reversal points |

### 💨 Momentum Indicators (3)
| Indicator | Abbreviation | Key Use Case |
|-----------|-------------|--------------|
| Relative Strength Index | RSI | Overbought/oversold (0-100) |
| Stochastic Oscillator | SR | Price momentum and range |
| Williams %R | WR | Similar to Stochastic (0 to -100) |

### 📊 Volatility Indicators (2)
| Indicator | Abbreviation | Key Use Case |
|-----------|-------------|--------------|
| Average True Range | ATR | Volatility measurement |
| Bollinger Bands | BB | Volatility bands and extremes |

### 📦 Volume Indicators (1)
| Indicator | Abbreviation | Key Use Case |
|-----------|-------------|--------------|
| On-Balance Volume | OBV | Volume-based trend confirmation |

### 🎯 Custom Indicators (1)
| Indicator | Key Use Case |
|-----------|--------------|
| Trend | Market regime detection (Uptrend/Downtrend/Range) |

---

## Popular Combinations

The repository includes **10 pre-built strategies** that combine indicators:

1. **SMA 20/50 + Trend** → Moving Averages Crossover
2. **SAR + Stochastic + Trend** → SAR Stochastic Strategy
3. **Stochastic + MACD** → Stochastic MACD Strategy
4. **RSI + Trend** → RSI Strategy
5. **Bollinger Bands + RSI** → BB RSI Strategy
6. **OBV + BB + RSI + Trend** → Multi-indicator Strategy
7. **ADX + Trend** → ADX Trend Strategy
8. **CCI + ADX + Trend** → CCI ADX Reversal Strategy
9. **Williams %R** → Williams %R Strategy
10. **VWMA + Z-score + Trend** → Mean Reversion Strategy

---

## Implementation

All indicators are available through:
- **Python `ta` library**: Most standard indicators
- **Custom implementations**: VWMA and Trend indicator
- **Strategy file**: `Financial-Algorithm Contents/Equity/Technical Indicators/technicalindicators_strategies.py`

---

## Getting Started

```bash
# Install the technical analysis library
pip install ta

# Or install all dependencies
pip install -r python-requirements.txt
```

For detailed documentation on each indicator, see [INDICATORS.md](INDICATORS.md)

---

## Example Usage

```python
import ta
import pandas as pd

# Assuming you have a DataFrame with OHLCV data
df = your_price_data

# Trend Indicators
df['SMA_20'] = ta.trend.sma_indicator(df['Close'], n=20)
df['MACD'] = ta.trend.macd(df['Close'])
df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

# Momentum Indicators  
df['RSI'] = ta.momentum.rsi(df['Close'], n=14)
df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])

# Volatility Indicators
df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])

# Volume Indicators
df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
```

---

## Need More Details?

- **Full Indicator Documentation**: [INDICATORS.md](INDICATORS.md)
- **Strategy Details & Backtests**: [Financial-Algorithm Contents/README.md](Financial-Algorithm%20Contents/README.md)
- **Code Examples**: Check the `signals/` directory for simple examples
