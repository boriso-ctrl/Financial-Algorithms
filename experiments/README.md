# Experiments Guide

This directory contains archived experimental code from Phase 5-7 testing. These are reference implementations and analysis code, not production code.

## Directory Structure

### `aggressive_growth/` - Phase 7: Aggressive Growth Mode Analysis

**Purpose**: Test extreme parameter configurations to understand performance ceilings

**Files**:
- `phase7_aggressive_backtest.py` - Stacked position testing (up to 3 concurrent positions)
- `phase7_trend_following.py` - Extended profit targets (TP2 at 6%)
- `phase7_trend_holding.py` - Hold-on-signal, exit-on-reversal approach
- `phase7_aggressive_holding.py` - Parameterizable entry/exit thresholds
- `phase7_diagnostic_signal.py` - Signal distribution analysis

**Key Finding**: 
- Stacked positions underperformed due to whipsaws (77.4% trailing stop exits)
- Best approach: Simple trend holding captures 80.43% of SPY move with 100% sizing
- Conclusion: In bull markets, simpler is better; trading adds drag

**Performance**: -1.15% (stacked) → +80.43% (simple holding)

### `intraday/` - Phase 6: Intraday & Multi-Timeframe Exploration

**Purpose**: Test strategy on shorter timeframes (1-hour, 4-hour) for higher frequency trading

**Files**:
- `phase6_intraday_backtest.py` - 1-hour and 4-hour bar testing
- `phase6_multi_timeframe_backtest.py` - 15m + 1h confluence analysis
- `phase6_mtf_diagnostic.py` - Multi-timeframe signal debugging

**Key Finding**:
- Daily strategy underperforms on intraday bars
- 1-hour SPY: -0.13%, 4-hour SPY: +0.25% (vs daily +2.41%)
- Multi-timeframe confluence too strict (0 trades with 4/5 signal requirement)
- Conclusion: Stick with daily resolution

**Decision**: Archive intraday approach

### `parameter_search/` - Phase 6: Parameter Optimization

**Purpose**: Systematically search for optimal entry/exit thresholds and position sizing

**Files**:
- `phase6_adaptive_search.py` - Bayesian optimization over 150+ iterations
- `phase6_parameter_sweep.py` - Grid search across parameter combinations
- `phase6_multibase_sweep.py` - Multiple asset sweep testing
- `phase6_demo.py` - Signal blending demonstration

**Search Space Tested**:
- Buy thresholds: +0.5 to +3.0
- Sell thresholds: -0.5 to -5.0
- Position sizing: 2% to 20%
- TP1/TP2 ratios: various combinations

**Key Finding**:
- Optimal: Buy +2.0, Sell -2.0, Size 4-8%
- More aggressive sizing (10-15%) increases drawdowns without return boost
- Lighter entry (+1.0) slightly improves returns but increases trades

**Recommendation**: Current baseline +2.0/-2.0, 4-8% remains optimal

### `robustness/` - Phase 5: Robustness & Stress Testing

**Purpose**: Validate strategy across different scenarios

**Files**:
- `phase5_robustness.py` - Stress testing under various conditions

**Tests Included**:
- Different time periods
- Market regime changes
- Position sizing variations
- Indicator period sensitivity

---

## How to Use Experiments

### For Reference
These scripts show **what was tested** and **why certain approaches didn't work**. Read them to understand the development process.

### To Replicate Results
```bash
# Run a specific experiment
python experiments/aggressive_growth/phase7_aggressive_holding.py --asset SPY --entry 1.0 --exit -2.5

# View results
cat aggressive_holding_spy_v2.json | python -m json.tool
```

### To Understand Findings
Each folder contains implementations that tested specific hypotheses:

1. **Aggressive Growth** → Found that simpler strategies work better in bull markets
2. **Intraday** → Found that daily bars outperform shorter timeframes
3. **Parameter Search** → Found that +2.0/-2.0, 4-8% is well-tuned
4. **Robustness** → Validated strategy across different conditions

---

## Why Not Production?

These experiments were valuable for **validation** but are **archived** because:

1. **Testing Complete** - Questions answered, hypothesis confirmed/rejected
2. **Not Needed for Deployment** - Production uses single best strategy
3. **Code Complexity** - Simplified versions in production are cleaner
4. **Reference Value** - Kept to document decision-making process

---

## Key Learning from Experiments

### Phase 5-6
- ✅ Conservative parameters (+2.0/-2.0) optimal for trading
- ✅ 4-8% sizing balances returns and drawdown
- ✅ Daily bars superior to intraday timeframes
- ✅ Strategy generalizes across all assets

### Phase 7
- ✅ Sharpe 3+ goal easily achievable (5.34+ achieved)
- ⚠️ Bull markets impossible to beat with market timing
- ⚠️ Stacked positions create whipsaw drag
- ⚠️ Extended exits (6%+ targets) don't improve returns enough

---

## If You Want to Modify the Strategy

### Quick Tweaks
Edit `src/financial_algorithms/strategies/voting_enhanced_weighted.py`:
```python
# Example: Change entry/exit thresholds
min_buy_score = 1.5      # Earlier entry
max_sell_score = -3.0    # Later exit (allow pullbacks)

# Then backtest:
python backtests/backtest_daily_voting.py --asset SPY
```

### Moderate Changes
Look at `experiments/aggressive_growth/phase7_aggressive_holding.py` for ideas on:
- Different position sizing formulas
- Entry/exit logic variations
- Custom profit-taking strategies

### Major Changes
Use the Phase 6 parameter search code:
```bash
python experiments/parameter_search/phase6_adaptive_search.py \
  --tickers AAPL SPY NVDA \
  --n-calls 100
```

---

## Next Steps for Research

If continuing development:

1. **Test 2015-2022 mixed market** (non-bull to validate)
2. **Test 2020 crash** (validate drawdown protection)
3. **Implement sector rotation** (allocate more to winners)
4. **Add machine learning** (parameter optimization)
5. **Build dashboard** (real-time monitoring)

See `docs/PHASE7_FINDINGS.md` for detailed recommendations.

---

## File Sizes

```
aggressive_growth/          ~2.5 MB
intraday/                   ~1.8 MB
parameter_search/           ~2.1 MB
robustness/                 ~0.6 MB
Total archived code:        ~6 MB
```

All experiments kept for reference; production uses optimized subset in `backtests/`.
