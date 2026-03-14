# URGENT: VOTING STRATEGY TUNING RESULTS
**Date:** March 11, 2026 | **Status:** ⚠️ CRITICAL FINDINGS

---

## Summary

Comprehensive parameter tuning and validation has identified a **critical issue**: the voting strategy is generating **negative Sharpe ratios across all tested configurations**, even with optimal buy_threshold tuning.

### Key Discovery from Threshold Testing

```
Threshold    Avg Sharpe    Avg Trades    Best Case    Worst Case
────────────────────────────────────────────────────────────────
1            -0.84         40 trades     MSFT +0.57   AAPL -1.44
2            -0.40         25 trades     MSFT +0.88   AAPL -1.22  ← BEST
2.5          -0.40         25 trades     MSFT +0.88   AAPL -1.22
3             -0.55         15 trades     MSFT +0.85   SPY -1.38
3.5          -0.56         15 trades     MSFT +0.85   SPY -1.38
4            -1.04         10 trades     [All negative]
5            -1.44         5 trades      [All negative]
```

**Finding:** Even optimal threshold (2-2.5) only achieves -0.40 Sharpe. Only MSFT consistently profitable.

---

## What This Means

✅ **Architecture works:** Voting system, position sizing, score calculation all verified  
✅ **Generates trades:** 25-40 trades/year depending on parameters  
❌ **Not profitable:** Negative Sharpe on most assets  
❌ **Not solving problem:** Tuning threshold alone insufficient  

---

## Root Causes (Analysis)

### 1. **Indicator Selection Issue**
The 8 indicators may not be complementary:
- RSI + Stochastic both measure momentum (redundant)
- MACD is also momentum
- Creates over-voting on momentum, under-voting on other factors

### 2. **Win Rate Problem**
Current across all thresholds: **35-40% win rate**
- Breakeven needs ~50% with current risk/reward
- Suggests entry signal quality is poor
- Not filtering false signals effectively

### 3. **Exit Logic Failure**
- Exiting at score ≤ 0 may be too aggressive
- Closes winners prematurely
- Doesn't protect losers adequately

### 4. **Position Sizing Mismatch**
- Using 0-4% sizing based on score
- But if entry quality is poor, even small positions lose
- May need risk scaling based on confidence vs. equality across all signals

---

## Recommended Path Forward

### Option A: Quick Fix (48 hours) - RECOMMENDED
Modify exit logic and try alternate indicators:

1. **Change exit condition:**
   ```python
   # OLD: Exit when score <= 0
   # NEW: Exit when (score < -1.5) OR (bar_count > 5) OR (loss > 2%)
   ```

2. **Test with indicator alternatives:**
   - Swap Stochastic + RSI → Keep only RSI
   - Swap MACD + RSI → Keep only MACD
   - Add Bollinger Band width for volatility
   - Results in 5-6 indicators instead of 8

3. **Expected:** 10-20% improvement in Sharpe

### Option B: Regime Detection (1 week) - COMPREHENSIVE
Implement market regime detection:

1. **Add regime filter:**
   - VIX > 30 = High volatility mode (stricter entry)
   - VIX < 15 = Low volatility mode (looser entry)
   - Different thresholds per regime

2. **Results:** Asset-dependent optimization
   - MSFT might operate at threshold 1.5 (works well)
   - TSLA needs threshold 3+ (too volatile otherwise)

3. **Expected:** 30-50% improvement

### Option C: Machine Learning Reweight (2 weeks) - ADVANCED
Use per-asset indicator optimization:

1. **Train XGBoost on 2023 data** to rank indicator importance
2. **Auto-adjust weights** per asset based on historical performance
3. **Backtest on 2024-2025**

4. **Expected:** 50%+ improvement

---

## Current State Assessment

| Aspect | Status | Comments |
|--------|--------|----------|
| Implementation | ✅ COMPLETE | All 8 indicators working, voting mechanism verified |
| Architecture | ✅ SOUND | Multi-indicator consensus approach valid |
| Performance | ❌ FAILED | Sharpe -0.40 avg (need 0.5+), win rate 35-40% (need 50%+) |
| Deployment Ready | ❌ NO | Must improve performance before live trading |
| Parameter Tuning | ✅ DONE | Threshold tested, 2.5 optimal but insufficient |
| Next Action | 🔄 CLEAR | Must change strategy, not just parameters |

---

## Decision Point

**Current voting strategy with 8 indicators is NOT profitable in current form.**

You must choose:
1. **Option A** - Try quick indicator modifications (risk: 50/50 success rate)
2. **Option B** - Implement regime detection (risk: medium complexity, 70% success)
3. **Option C** - Full ML approach (risk: high complexity, 80% success)
4. **Option D** - Abandon voting strategy, return to hybrid Phase 3

---

## Data Files Generated

All validation and tuning results saved:
- `VALIDATION_REPORT_MARCH11.md` - Full validation analysis
- `data/search_results/direct_validation.json` - Per-symbol results
- `data/search_results/threshold_tuning.json` - Threshold optimization  
- `scripts/tune_threshold.py` - Tuning script for future

---

## Files Ready for Next Phase

If you choose **Option A (Quick Fix)**:
```
1. Edit: src/financial_algorithms/strategies/multi_indicator_voting.py
   - Modify exit_threshold logic (line ~420)
   - Modify indicator_weights (8 → 5 indicators)

2. Run: scripts/tune_threshold.py
   - Retest with new configuration
   - Check if Sharpe improves to 0.1+

3. If improved: Run full validation
4. If not: Escalate to Option B or C
```

---

## My Assessment

The voting strategy architecture is solid but **needs surgery, not tweaking**. The consistent -0.40 Sharpe across optimal parameters suggests a systemic issue beyond just threshold tuning.

**Recommendation:** Start with **Option A** (quick exit logic + indicator reduction). If that doesn't get to +0.3 Sharpe minimum, escalate to **Option B** (regime detection) immediately.

Timeline: **Current path unviable for live trading without modifications.**

---

## Next Steps (Pick One)

[ ] **A** - Continue with voting strategy (require exit logic + indicator changes)  
[ ] **B** - Implement regime detection for voting strategy  
[ ] **C** - Full ML reweighting of indicators  
[ ] **D** - Abandon voting, go back to Phase 3 hybrid system  

**Current Status:** Awaiting direction to proceed...
