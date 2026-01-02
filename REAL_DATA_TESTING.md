# Testing with Real Historical Data

This guide explains how to test the VWAP + ATR strategy with real historical market data instead of synthetic data.

## Quick Start

### Option 1: CSV Files (Recommended)

If you have historical data in CSV format or can download it:

```bash
# Test with your CSV file
python examples/run_csv_backtest.py --csv your_data.csv --interval 5m

# Generate sample CSV (for testing the functionality)
python examples/generate_sample_csv.py
python examples/run_csv_backtest.py --csv examples/sample_data/sample_5min_data.csv
```

### Option 2: yfinance (Direct Download)

For automatic downloading from Yahoo Finance:

```bash
# Install yfinance
pip install yfinance

# Test with SPY (last 60 days, 5-minute bars)
python examples/run_real_data_backtest.py --ticker SPY --period 60d --interval 5m

# Test multiple tickers
python examples/run_real_data_backtest.py --multi
```

**Note**: yfinance limits intraday data to the last 60 days.

## CSV Data Format

### Required Columns

Your CSV must have these columns (case-insensitive):
- `timestamp` (or `datetime`, `date`, `time`)
- `open`
- `high`
- `low`
- `close`
- `volume`

### Example CSV Format

```csv
timestamp,open,high,low,close,volume
2024-01-02 09:30:00,450.50,451.20,450.30,451.00,1000000
2024-01-02 09:35:00,451.00,451.50,450.80,451.20,950000
2024-01-02 09:40:00,451.20,451.80,451.00,451.50,1100000
```

### Timestamp Formats Supported

- `2024-01-02 09:30:00`
- `2024-01-02 09:30:00.000`
- `2024-01-02T09:30:00`
- `01/02/2024 09:30`
- Unix timestamps (will be auto-converted)

## Where to Get Real Data

### Free Sources

1. **Yahoo Finance** (via website)
   - Go to finance.yahoo.com
   - Search for ticker (e.g., SPY)
   - Click "Historical Data"
   - Select time period and interval
   - Download CSV

2. **yfinance** (Python library)
   ```python
   import yfinance as yf
   ticker = yf.Ticker("SPY")
   df = ticker.history(period="60d", interval="5m")
   df.to_csv("SPY_5min.csv")
   ```

3. **Alpha Vantage**
   - Free API key at alphavantage.co
   - Supports intraday and daily data
   - Limited to 5 calls per minute (free tier)

4. **Polygon.io**
   - Free tier available
   - Good quality data
   - Supports stocks, forex, crypto

### Paid Sources (Better Quality)

1. **Interactive Brokers**
   - Historical data API
   - Excellent for serious backtesting
   - Requires account

2. **Norgate Data**
   - Survivorship-bias free
   - Professional quality
   - ~$30-50/month

3. **QuantConnect**
   - Cloud platform with data
   - Free tier available
   - Paid for more features

4. **Databento**
   - Professional market data
   - Pay per use
   - High quality

## Usage Examples

### Basic CSV Test

```bash
# Download data from Yahoo Finance manually, save as SPY_5min.csv
python examples/run_csv_backtest.py --csv SPY_5min.csv --interval 5m
```

### Custom Timestamp Column

If your CSV uses a different column name:

```bash
python examples/run_csv_backtest.py \
  --csv mydata.csv \
  --interval 5m \
  --timestamp-col "DateTime"
```

### Different Timeframes

```bash
# 15-minute bars
python examples/run_csv_backtest.py --csv data_15min.csv --interval 15m

# 1-hour bars
python examples/run_csv_backtest.py --csv data_hourly.csv --interval 1h

# Daily bars
python examples/run_csv_backtest.py --csv data_daily.csv --interval 1d
```

### Multiple Tickers (yfinance)

```bash
# Test on SPY, QQQ, IWM
python examples/run_real_data_backtest.py --multi
```

## Understanding Results

### Expected Performance with Real Data

Real market data will show **different results** than synthetic data:

| Metric | Synthetic Data | Real Data (Expected) |
|--------|----------------|---------------------|
| Sharpe Ratio | 4.0-7.0 | 0.5-2.5 |
| Win Rate | 70-80% | 50-65% |
| Profit Factor | 3.0-5.0 | 1.2-2.0 |
| Max Drawdown | 1-3% | 5-15% |

### Why Real Data Differs

1. **Market Complexity**: Real markets have gaps, news events, regime changes
2. **Slippage**: Order execution isn't perfect
3. **Commissions**: Trading costs reduce profits
4. **Market Impact**: Large orders move prices
5. **Liquidity**: Not all price levels have volume
6. **Adverse Selection**: Getting filled often means you're on wrong side

### Realistic Expectations

For a **good intraday mean reversion strategy** with real data:
- Sharpe Ratio: **1.0-2.0** is excellent
- Win Rate: **55-60%** is good
- Max Drawdown: **5-10%** is normal
- Annual Return: **15-30%** (after costs) is very good

A Sharpe ratio of **1.5+ with real data** indicates a strong strategy.

## Adding Transaction Costs

Real trading has costs. Modify the backtest to include them:

### Typical Costs

- **Commissions**: $0-1 per trade (most brokers now $0)
- **Spread**: 0.01-0.05% per trade (bid-ask spread)
- **Slippage**: 0.01-0.10% per trade (market impact)
- **Total**: ~0.02-0.15% per round trip

### Simple Cost Model

Deduct 0.05% per trade (0.025% entry + 0.025% exit):

```python
# In backtest, after calculating PnL:
cost_per_trade = 0.0005  # 0.05% round trip
trade_cost = shares * entry_price * cost_per_trade
pnl -= trade_cost
```

For 100 trades/year with $100k capital:
- Cost per trade: ~$50
- Annual costs: ~$5,000
- Impact on return: -5%

## Validation Checklist

Before going live, validate with real data:

- [ ] Test on at least 3-6 months of recent data
- [ ] Test on multiple tickers (SPY, QQQ, IWM, etc.)
- [ ] Test on both trending and choppy periods
- [ ] Add realistic transaction costs
- [ ] Verify Sharpe ratio > 1.0 after costs
- [ ] Check max drawdown is acceptable
- [ ] Analyze worst losing streaks
- [ ] Review trade distribution (avoid overfitting)
- [ ] Paper trade for 2-4 weeks
- [ ] Start with small position sizes

## Common Issues

### Issue: "Could not download data"

**Solutions**:
- Check internet connection
- Verify ticker symbol is correct
- Try different time period
- Use CSV method instead
- Check yfinance is installed: `pip install yfinance`

### Issue: "No valid data after indicator calculation"

**Solutions**:
- Data may be too short (need 20+ bars for indicators)
- Check for NaN values in OHLCV columns
- Verify data has proper timestamps
- Ensure volume > 0

### Issue: "Sharpe ratio is negative or very low"

This is **normal with real data**! Consider:
- Strategy may not work in current market conditions
- Need more data for accurate assessment
- Transaction costs may be too high
- May need parameter optimization
- Different timeframe might work better

### Issue: "No trades generated"

**Causes**:
- Market regime doesn't meet entry criteria
- Volatility too low (ATR bands too tight)
- Price not reaching entry zones
- Need longer time period

**Solutions**:
- Test longer period (more sessions)
- Try different ticker (more volatile)
- Review signal generation logic
- Check if indicators calculated correctly

## Comparing Results

### Synthetic vs Real Data

When you test with real data, create a comparison:

```python
# Run both tests
synthetic_sharpe = 6.5  # From synthetic backtest
real_sharpe = 1.8       # From real data backtest

print(f"Synthetic Sharpe: {synthetic_sharpe:.2f}")
print(f"Real Data Sharpe: {real_sharpe:.2f}")
print(f"Realism Factor: {real_sharpe / synthetic_sharpe:.2%}")
```

A **realism factor of 25-40%** is typical (real Sharpe is 25-40% of synthetic).

### Different Tickers

Results vary by ticker:
- **SPY**: Large cap, liquid, lower volatility → Lower Sharpe
- **QQQ**: Tech-heavy, more volatile → Higher Sharpe (but higher risk)
- **IWM**: Small cap, higher volatility → Variable results
- **Individual Stocks**: Much higher variance

## Next Steps

After testing with real data:

1. **If Sharpe > 1.5**: Strategy looks promising
   - Paper trade for 2-4 weeks
   - Monitor real-time performance
   - Verify signal generation works live

2. **If Sharpe 0.8-1.5**: Strategy is marginal
   - Consider optimizing parameters
   - Try different timeframes
   - Add more filters
   - Reduce position sizes

3. **If Sharpe < 0.8**: Strategy needs work
   - Market regime may not suit strategy
   - Try different ticker/market
   - Reconsider strategy logic
   - More testing needed

## Example Workflow

Complete testing workflow:

```bash
# 1. Generate sample CSV to verify script works
python examples/generate_sample_csv.py
python examples/run_csv_backtest.py --csv examples/sample_data/sample_5min_data.csv

# 2. Download real data from Yahoo Finance
# Save as SPY_5min.csv

# 3. Test with real data
python examples/run_csv_backtest.py --csv SPY_5min.csv --interval 5m

# 4. Compare results
# Synthetic: Sharpe ~6.5
# Real: Sharpe ~1.5-2.0 (if good)

# 5. Test multiple timeframes
python examples/run_csv_backtest.py --csv SPY_15min.csv --interval 15m
python examples/run_csv_backtest.py --csv SPY_1h.csv --interval 1h

# 6. If results good (Sharpe > 1.5), proceed to paper trading
```

## Resources

- **yfinance docs**: https://github.com/ranaroussi/yfinance
- **Yahoo Finance**: https://finance.yahoo.com
- **Alpha Vantage**: https://www.alphavantage.co
- **Polygon.io**: https://polygon.io
- **Data format guide**: See `DATA_SOURCES.md`

---

**Remember**: Real market performance will always be lower than synthetic data. A Sharpe ratio of 1.5-2.0 with real data is excellent and indicates a robust strategy worth paper trading.
