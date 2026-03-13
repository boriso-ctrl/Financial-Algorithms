## LOOKAHEAD BIAS BUG - BEFORE & AFTER

### The Issue You Caught

**Your question:** "Does the trading algorithm backtest in REAL TIME or does it know what price action is coming?"

**Answer:** It was cheating. The backtest knew future prices. You were right.

---

## Side-by-Side Comparison

### BEFORE (Lookahead Bug)

**Code (intraday_engine.py, lines 252-254):**
```python
def backtest(self, df_bars, signal_func, close_on_bar_n=None):
    # Generate signals BEFORE processing bars
    signals = signal_func(df_bars)  # ❌ PROBLEM: df_bars = entire dataset!
    df_bars['signal'] = signals
    
    # Process bars
    for symbol, group in df_bars.groupby('symbol'):
        for idx, row in group.iterrows():
            signal = row['signal']  # Already computed with FUTURE data!
```

**What was happening:**
- signal_func receives 2,000 bars (entire 16-hour session)
- Computing SMA on 2,000 bars = knows prices 0-1999
- When at bar 100, SMA includes bars 101-1999 (future!)
- RSI extremes detected with perfect foresight
- Result: Signals perfectly predict price moves

**Results:**
```
Crypto Backtest (BTC/ETH):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sharpe Ratio:      6.37 ✅ Looks amazing!
Total Return:      3.34%
Win Rate:          50%
Trades:            32
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

💬 **"WHOA! Sharpe 6.37?? Let's deploy this!"**

---

### AFTER (Walk-Forward Fix)

**Code (intraday_engine.py, lines 248-278):**
```python
def backtest(self, df_bars, signal_func, close_on_bar_n=None, min_history=50):
    # Process bars sequentially
    for idx in range(len(df_bars)):
        row = df_bars.iloc[idx]
        bars_so_far = idx - symbol_start_idx[symbol] + 1
        
        if bars_so_far >= min_history:
            # Create historical slice: ONLY past data
            hist_data = df_bars[(df_bars['symbol']==symbol) & (df_bars.index <= idx)]
            
            # Signal computed on HISTORICAL DATA ONLY
            signal = signal_func(hist_data)  # ✅ CORRECT: No future data!
        else:
            signal = 0  # Warm-up period
        
        # Execute trade with THIS signal
        self._try_entry(symbol, signal, price, size, timestamp)
```

**What happens now:**
- signal_func receives only bars 0-100 (no lookahead)
- Computing SMA on 100 bars = uses only past data
- At bar 100, prices 101+ are completely unknown
- Signals made WITHOUT knowing the future
- Result: Realistic trading performance

**Results:**
```
Crypto Backtest (BTC/ETH):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sharpe Ratio:     -42.89 ❌ Brutal honesty
Total Return:      -3.17%  (Losing money!)
Win Rate:          26.7%   (Below coin flip)
Trades:            30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

💬 **"Oh no. This strategy is worthless."** ← But this is GOOD news!

---

## What Changed

| Component | Before | After |
|-----------|--------|-------|
| Signal Generation | Upfront, full dataset | Per-bar, historical only |
| Data Available | *All future prices* ❌ | *Only past prices* ✅ |
| Warm-up Period | None | 50 bars minimum |
| Trading Realism | Fake | Real |
| Useful for... | Disaster | Honest evaluation |

---

## Why This Matters

**Trading CANNOT work with lookahead bias:**
```
Real world:                        Your old backtest:
─────────────────────────────     ─────────────────────────────
Current bar arrives                You saw the entire price chart
↓ Make decision with              ↓ Made decisions knowing
   ONLY past data                    ALL future data
↓ You are often wrong             ↓ You were always right
↓ Real Sharpe: -5 to +1           ↓ Fake Sharpe: +6
```

When you deploy on real prices, you DON'T get to see the future. So backtests that rely on future data are worthless.

---

## Detection Checklist

### How to spot lookahead bias:

- [ ] Sharpe > 3.0 on short timeframes (1-5 min)?
- [ ] Win rate consistently > 55%?
- [ ] All signals computed before processing bars?
- [ ] Indicators computed on full dataset?

**If YES to ANY:**  Probably lookahead bias

### How to prevent it:

- [x] Process bars sequentially (walk-forward)
- [x] Signal function gets ONLY historical bars
- [x] Minimum warm-up period (50-100 bars)
- [x] Test on out-of-sample data
- [x] Cross-validate on different time periods

---

## What's Still Valid

Some code/concepts are still good (just poorly tested):

✅ **Multi-timeframe ensemble** – Good idea, needs real testing  
✅ **Regime detection** – Good idea, filters too strict atm  
✅ **Position sizing** – Conceptually sound, needs tuning  
✅ **Intraday engine** – Framework now correct  
✅ **Walk-forward backtest** – NOW the gold standard  

---

## The Path Forward

### Phase 6 Status: RESET

The original Phase 6 goals were based on lookahead bias results:

```
OLD TARGET:  Sharpe 3.0 on intraday (FAKE, based on lookahead)
ACTUAL RESULTS: Sharpe -42.89 on intraday (REAL, walk-forward)
```

### New Task: Build a REAL Sharpe 3.0 Strategy

Options:

**Option A: Fix Phase 6 intraday**
- Start over with honest walk-forward design
- Better signal generation
- Stronger regime filters
- Expected: 0.5-1.0 Sharpe (realistic for intraday)

**Option B: Stick with Phase 3-4 daily strategies**
- Those already use walk-forward validation ✅
- Proven Sharpe 1.65 on daily data
- Lower trading frequency = lower costs
- Expected: 1.5-2.0 Sharpe (realistic for daily)

**Option C: Hybrid multi-timeframe**
- Use Phase 3/4 signals as base (daily, proven)
- Add intraday filters (regime detection from Phase 6)
- Trade only when multiple timeframes agree
- Expected: 0.8-1.2 Sharpe (conservative)

---

## Files Changed

✅ `src/financial_algorithms/backtest/intraday_engine.py`
- Line 248: Removed upfront signal generation
- Lines 248-278: Added walk-forward loop
- Added: min_history warm-up period
- Added: Per-bar historical data slicing

✅ `scripts/phase6_demo.py`
- Updated signal_func signature (now takes hist_data, returns single signal)
- Updated demo_crypto_backtest
- Updated demo_stock_backtest

✅ `LOOKAHEAD_BIAS_FIX.md` (new file)
- Detailed explanation with code examples

---

## Key Takeaway

**Before:** Sharpe 6.37 (knew the future)  
**After:** Sharpe -42.89 (didn't know the future)  
**Status:** BUG FIXED ✅

This is exactly what should happen when you add proper walk-forward validation. If your backtest got WORSE after fixing lookahead bias, that's actually a sign your testing framework is now trustworthy.

Now strategy optimization can actually help (instead of just overfitting to future prices).

