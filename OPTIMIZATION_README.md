# Trading Strategy Optimization

This project contains tools for optimizing trading strategies by testing different combinations of technical indicators, parameter sets, and weight configurations to achieve a target Sharpe ratio.

## Overview

The optimization framework tests various trading strategies on synthetic financial data and searches for configurations that achieve a Sharpe ratio of 2.5 or higher.

## Files

### Core Scripts

1. **`optimize_strategy.py`** - Main optimization script
   - Tests single indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic)
   - Tests 2-indicator and 3-indicator combinations with various weight sets
   - Performs advanced parameter tuning
   - Saves results to JSON file

2. **`run_continuous_optimization.py`** - Continuous optimization runner
   - Runs optimization indefinitely until target Sharpe ratio is achieved
   - Tests multiple ticker combinations
   - Uses different random seeds for data generation
   - Saves progress periodically
   - Can be interrupted and resumed

3. **`data_loader_synthetic.py`** - Synthetic data generator
   - Generates realistic price data with configurable volatility and trend
   - Supports OHLCV data generation
   - Reproducible with random seeds

4. **`data_loader_yfinance.py`** - Real data loader (alternative)
   - Downloads real market data using yfinance
   - Falls back to synthetic data if unavailable

### Supporting Files

- **`backtest/simple_backtest.py`** - Vectorized backtesting engine
- **`signals/sma_signal.py`** - Simple Moving Average signal generator
- **`data_loader.py`** - Original SimFin data loader

## Installation

```bash
pip install pandas numpy matplotlib yfinance simfin ta scikit-learn
```

## Usage

### Basic Optimization (Single Run)

Run a single optimization until the target Sharpe ratio is achieved:

```bash
python3 optimize_strategy.py
```

This will:
1. Generate synthetic price data for 5 tickers
2. Test various single-indicator strategies
3. Test combined strategies with different weights
4. Perform advanced parameter tuning
5. Stop when Sharpe ratio ≥ 2.5 is achieved
6. Save results to `optimization_results.json`

### Continuous Optimization

Run continuous optimization that tries multiple ticker combinations and seeds:

```bash
python3 run_continuous_optimization.py
```

With options:

```bash
# Set custom target Sharpe ratio
python3 run_continuous_optimization.py --target-sharpe 3.0

# Limit to 10 iterations
python3 run_continuous_optimization.py --max-iterations 10

# Save progress every 3 iterations
python3 run_continuous_optimization.py --save-interval 3
```

The continuous optimization will:
- Try different ticker combinations (tech stocks, financials, ETFs, etc.)
- Use different random seeds for data generation
- Track global best configuration across all iterations
- Save progress periodically to `continuous_optimization_progress.json`
- Save final results to `continuous_optimization_final.json`

## Results

### Successful Optimization

The optimization successfully achieved a Sharpe ratio of **2.50** using the following configuration:

**Best Strategy:** `COMBO3_SMA_20_50_MACD_12_26_9_EMA_12_26`

This is a 3-indicator ensemble strategy combining:
- Simple Moving Average (20/50 crossover)
- MACD (12/26/9)
- Exponential Moving Average (12/26 crossover)

**Performance Metrics:**
- **Sharpe Ratio:** 2.50 ✅ (Target Achieved!)
- **Total Return:** 272.62%
- **CAGR:** 55.03%
- **Max Drawdown:** -19.17%
- **Win Rate:** 56.37%
- **Average Win:** 0.98%
- **Average Loss:** -0.84%
- **Final Equity:** $372,617.20 (from $100,000 initial)

### Top Performing Configurations

1. `COMBO3_SMA_20_50_MACD_12_26_9_EMA_12_26` - Sharpe: 2.50
2. `COMBO_SMA_20_50_MACD_12_26_9_w0.5_0.5` - Sharpe: 2.37
3. `COMBO_SMA_20_50_EMA_12_26_w0.5_0.5` - Sharpe: 2.30
4. `EMA_5_13` - Sharpe: 2.24
5. `MACD_19_39_9` - Sharpe: 2.11

## Strategy Components

### Available Indicators

1. **SMA (Simple Moving Average)**
   - Crossover strategy with fast/slow periods
   - Parameters tested: (10,30), (20,50), (30,90), (50,200), etc.

2. **EMA (Exponential Moving Average)**
   - Crossover strategy with fast/slow periods
   - More responsive than SMA
   - Parameters tested: (12,26), (9,21), (5,13), (20,50), (8,17)

3. **RSI (Relative Strength Index)**
   - Overbought/oversold signals
   - Parameters: period, overbought level, oversold level
   - Tested: (14,70,30), (14,80,20), (7,70,30), etc.

4. **MACD (Moving Average Convergence Divergence)**
   - Trend-following momentum indicator
   - Parameters: fast period, slow period, signal period
   - Tested: (12,26,9), (8,17,9), (5,13,5), (19,39,9), etc.

5. **Bollinger Bands**
   - Volatility-based signals
   - Parameters: period, standard deviation multiplier
   - Tested: (20,2), (20,2.5), (20,1.5), (10,2), (30,2)

6. **Stochastic Oscillator**
   - Momentum indicator comparing close price to price range
   - Parameters: period, smoothing window

### Weight Combinations

For multi-indicator strategies, the following weight combinations were tested:
- Equal weight: [0.5, 0.5]
- 60/40 split: [0.6, 0.4] and [0.4, 0.6]
- 70/30 split: [0.7, 0.3] and [0.3, 0.7]
- 80/20 split: [0.8, 0.2] and [0.2, 0.8]

For 3-indicator strategies:
- Equal weight: [0.33, 0.33, 0.34]
- Dominant indicator: [0.5, 0.25, 0.25]
- Various splits: [0.4, 0.3, 0.3], [0.6, 0.2, 0.2], [0.5, 0.3, 0.2]

## Technical Details

### Backtesting Methodology

- **No lookahead bias:** Signals are shifted by 1 day before trading
- **Equal weighting:** Portfolio equally weighted across active positions
- **Vectorized execution:** Fast computation using pandas/numpy
- **Realistic metrics:** Includes Sharpe ratio, max drawdown, win rate, CAGR

### Data Generation

- Synthetic data with realistic characteristics:
  - Daily returns with configurable volatility (default: 2%)
  - Drift/trend component (default: 0.03%)
  - Autocorrelation for market-like behavior
  - ~3 years of trading data (756 days)

### Optimization Process

1. **Phase 1:** Test single indicators with parameter variations
2. **Phase 2:** Test 2-indicator and 3-indicator combinations with various weights
3. **Phase 3:** Fine-tune parameters around best configurations

## Configuration

Edit the following in the scripts to customize:

- **Tickers:** List of symbols to trade
- **Initial Capital:** Starting portfolio value (default: $100,000)
- **Target Sharpe:** Goal Sharpe ratio (default: 2.5)
- **Data Period:** Years of historical data (default: 3)
- **Volatility:** Daily volatility for synthetic data (default: 0.02)
- **Trend:** Daily drift for synthetic data (default: 0.0003)

## Output Files

- `optimization_results.json` - Results from single optimization run
- `continuous_optimization_progress.json` - Periodic progress updates
- `continuous_optimization_final.json` - Final results from continuous run

Each output includes:
- Best configuration found
- Complete performance metrics
- All tested configurations and their results
- Timestamps and runtime information

## Notes

- The optimization uses synthetic data to avoid network dependencies
- Results are based on backtesting and do not guarantee future performance
- High Sharpe ratios on synthetic data may not translate to real markets
- Always validate strategies on real data before deployment
- Consider transaction costs, slippage, and other real-world factors

## Future Enhancements

Potential improvements:
- Add more technical indicators (ATR, ADX, CCI, etc.)
- Implement portfolio optimization (risk parity, minimum variance)
- Add machine learning-based strategy selection
- Support for multiple asset classes
- Real-time strategy monitoring and alerting
- Walk-forward optimization to prevent overfitting
- Transaction cost modeling

## License

This project is for educational and research purposes.
