# Financial Algorithms

A collection of profitable trading algorithms and trading strategies designed to be extensible and optimizable.

## Quick Links

- **[Available Trading Indicators](INDICATORS.md)** - Complete list of technical indicators available for trading strategies
- **[Detailed Documentation](Financial-Algorithm%20Contents/README.md)** - Comprehensive guide to all strategies and implementations

## What's Available?

### Trading Indicators

This repository provides access to a comprehensive set of technical indicators for building trading strategies:

- **Trend Indicators**: SMA, EMA, VWMA, MACD, ADX, CCI, Parabolic SAR
- **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R
- **Volatility Indicators**: ATR, Bollinger Bands
- **Volume Indicators**: OBV (On-Balance Volume)

See [INDICATORS.md](INDICATORS.md) for complete details on each indicator.

### Trading Strategies

10+ implemented strategies including:
- Moving Average Crossovers
- Mean Reversion Strategies
- Momentum-based Strategies
- Multi-indicator Combined Strategies

### Market Coverage

- **Equity Markets**: Technical indicators, fundamental trading, NLP trading, deep learning
- **Forex Markets**: Kalman Filter pairs trading

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/boriso-ctrl/Financial-Algorithms.git
cd Financial-Algorithms

# Install dependencies
pip install -r python-requirements.txt
```

### Quick Example

```python
from data_loader import load_daily_prices
from signals.sma_signal import sma_signal

# Load price data
tickers = ["AAPL"]
prices = load_daily_prices(tickers)

# Generate trading signal
signal = sma_signal(prices["AAPL"])
print(signal.tail())
```

## Documentation

- **[INDICATORS.md](INDICATORS.md)** - Lists all available technical indicators with descriptions and usage
- **[Financial-Algorithm Contents/README.md](Financial-Algorithm%20Contents/README.md)** - Detailed documentation of all strategies, backtesting results, and methodologies

## Requirements

- Python 3.7+
- Key libraries: `pandas`, `ta`, `pandas_datareader`, `scikit-learn`, `tensorflow`, `simfin`

See `python-requirements.txt` for complete dependencies.

## Repository Structure

```
Financial-Algorithms/
├── INDICATORS.md                     # List of available indicators
├── README.md                         # This file
├── signals/                          # Trading signal implementations
├── backtest/                         # Backtesting framework
├── data_loader.py                    # Data loading utilities
├── Financial-Algorithm Contents/     # Detailed strategies and documentation
│   ├── README.md                     # Comprehensive documentation
│   ├── Equity/
│   │   ├── Technical Indicators/
│   │   ├── Fundamental Trading/
│   │   ├── NLP Trading/
│   │   ├── Robust Strategies/
│   │   └── Deep Learning Trading/
│   └── Forex/
└── python-requirements.txt           # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to:
- Implement new indicators
- Develop new trading strategies
- Improve existing strategies
- Add tests and documentation

## License

This project is licensed under the MIT License.

## Acknowledgments

This repository builds on various trading strategies and research from the quantitative finance community.
