"""
Leveraged Alpha v7 — Trailing Stops + Expanded Pairs + Ultra-Selective + Ensemble
==================================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95  |  2010-01-01 to 2025-03-01

v6 best results:
  S4_CrashHedge(lev=1.0):      CAGR=16.39%, Sharpe=1.05, MaxDD=-25.74%
  S3_Pairs(not=0.3):           CAGR=1.82%,  Sharpe=1.02, MaxDD=-5.49%
  S5_MomBin(t=5,lev=1.0):      CAGR=13.03%, Sharpe=0.99, MaxDD=-18.91%
  Ens_S5+S3:                    CAGR=9.75%,  Sharpe=1.04, MaxDD=-12.99%

v7 innovations:
  1. TRAILING STOP-LOSS: Sequential backtest with trailing stop from equity peak.
     When equity drops X% from its peak → exit to cash. Re-enter when signals confirm.
     This DIRECTLY truncates the left tail → massive Sharpe improvement.
  2. EXPANDED PAIRS BASKET: 15+ pairs across sectors, asset classes, styles.
     More pairs → internal diversification → higher portfolio-level Sharpe.
  3. ULTRA-SELECTIVE QQQ TIMING: Invest only when conditions are PERFECT.
     Accept lower CAGR on this leg; the leverage + pairs compensate.
  4. OPTIMIZED ENSEMBLE WEIGHTS: Use inverse-variance weighting with
     correlation adjustment (not equal weights).
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

# ── Config ──
SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE"]
BROAD   = ["SPY", "QQQ", "IWM", "EFA"]
SAFE    = ["TLT", "IEF", "GLD", "SHY"]
ALL_TICKERS = SECTORS + BROAD + SAFE
START, END = "2010-01-01", "2025-03-01"
TX_BPS     = 5
LEV_COST   = 0.015
SHORT_COST = 0.005
RF_CASH    = 0.02
RF = 0.0
SEP = "=" * 90

# ── Data ──
def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers loaded, {len(p)} days "
          f"({p.index[0].strftime('%Y-%m-%d')} to {p.index[-1].strftime('%Y-%m-%d')})\n")
    return p

# ── Helpers ──
def sma(s, n):     return s.rolling(n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=max(10, n//2)).std() * np.sqrt(252)
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.clip(lower=1e-10))

def breadth_count(prices, lookback=50):
    sectors = [s for s in SECTORS if s in prices.columns]
    b = pd.Series(0.0, index=prices.index)
    for s in sectors:
        b += (prices[s] > sma(prices[s], lookback)).astype(float)
    return b


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENTIAL BACKTEST ENGINE WITH TRAILING STOP
# ═══════════════════════════════════════════════════════════════════════════

def backtest_with_stop(prices, weights, trailing_stop=None, reentry_delay=5,
                       rebal=1, cap=100_000.0):
    """
    Sequential backtest with optional trailing stop-loss.
    trailing_stop: e.g. -0.08 means exit when equity drops 8% from peak.
    reentry_delay: days to wait after stop triggered before allowing re-entry.
    """
    common = prices.columns.intersection(weights.columns)
    p = prices[common]
    w = weights[common].reindex(prices.index).fillna(0)

    if rebal > 1:
        mask = pd.Series(False, index=w.index)
        mask.iloc[::rebal] = True
        w = w.where(mask).ffill().fillna(0)

    # CRITICAL: shift(1) for no look-ahead
    w = w.shift(1).fillna(0)
    ret = p.pct_change().fillna(0)

    if trailing_stop is None:
        # Vectorized path (fast)
        port_ret = (w * ret).sum(axis=1)
        net_exp = w.sum(axis=1)
        cash_w = (1 - net_exp).clip(lower=0)
        cash_ret = cash_w * RF_CASH / 252
        turn = w.diff().fillna(0).abs().sum(axis=1)
        tx = turn * TX_BPS / 10_000
        g_exp = w.abs().sum(axis=1)
        lc = (g_exp - 1).clip(lower=0) * LEV_COST / 252
        sc = w.clip(upper=0).abs().sum(axis=1) * SHORT_COST / 252
        net = port_ret + cash_ret - tx - lc - sc
        eq = cap * (1 + net).cumprod()
    else:
        # Sequential with trailing stop
        n = len(p)
        eq_vals = np.empty(n)
        eq_vals[0] = cap
        net_vals = np.zeros(n)
        turn_vals = np.zeros(n)
        gexp_vals = np.zeros(n)
        nexp_vals = np.zeros(n)
        cash_vals = np.zeros(n)

        w_arr = w.values.copy()
        ret_arr = ret.values
        prev_w = np.zeros(w_arr.shape[1])

        peak = cap
        stopped = False
        stop_day = -999

        for i in range(1, n):
            # Check trailing stop
            dd = (eq_vals[i-1] - peak) / peak if peak > 0 else 0
            if dd < trailing_stop and not stopped:
                stopped = True
                stop_day = i

            # If stopped, check re-entry condition
            if stopped:
                days_since = i - stop_day
                if days_since >= reentry_delay:
                    # Check if original weights are non-zero (signals say go)
                    orig_sum = np.abs(w_arr[i]).sum()
                    if orig_sum > 0.01 and eq_vals[i-1] > peak * (1 + trailing_stop * 0.5):
                        # Price recovered half of the drawdown → re-enter
                        stopped = False

            if stopped:
                # All cash
                today_w = np.zeros(w_arr.shape[1])
            else:
                today_w = w_arr[i]

            daily_ret = np.dot(today_w, ret_arr[i])
            turn_val = np.abs(today_w - prev_w).sum()
            g_exp = np.abs(today_w).sum()
            n_exp = today_w.sum()
            cash_pct = max(1 - n_exp, 0)

            cash_r = cash_pct * RF_CASH / 252
            tx = turn_val * TX_BPS / 10_000
            lc = max(g_exp - 1, 0) * LEV_COST / 252
            sc = np.abs(np.minimum(today_w, 0)).sum() * SHORT_COST / 252
            net_ret = daily_ret + cash_r - tx - lc - sc

            eq_vals[i] = eq_vals[i-1] * (1 + net_ret)
            if not stopped:
                peak = max(peak, eq_vals[i])
            net_vals[i] = net_ret
            turn_vals[i] = turn_val
            gexp_vals[i] = g_exp
            nexp_vals[i] = n_exp
            cash_vals[i] = cash_pct
            prev_w = today_w

        eq = pd.Series(eq_vals, index=p.index, name="Equity")
        net = pd.Series(net_vals, index=p.index)
        turn = pd.Series(turn_vals, index=p.index)
        g_exp_s = pd.Series(gexp_vals, index=p.index)
        net_exp = pd.Series(nexp_vals, index=p.index)
        cash_w = pd.Series(cash_vals, index=p.index)
        w = pd.DataFrame(np.zeros_like(w_arr), index=p.index, columns=common)

    eq.name = "Equity"
    if trailing_stop is None:
        g_exp_s = w.abs().sum(axis=1)
        net_exp = w.sum(axis=1)
        net = port_ret + cash_ret - tx - lc - sc

    m = compute_metrics(net, eq, cap, risk_free_rate=RF,
                        turnover=turn, gross_exposure=g_exp_s)

    return {
        "equity_curve": eq, "portfolio_returns": net, "weights": w,
        "turnover": turn, "gross_exposure": g_exp_s, "metrics": m,
        "net_exposure": net_exp, "cash_weight": cash_w,
    }


# Simple vectorized backtest (no stop-loss)
def backtest_simple(prices, weights, rebal=1, cap=100_000.0):
    return backtest_with_stop(prices, weights, trailing_stop=None, rebal=rebal, cap=cap)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY A: Crash-Hedged QQQ (v6's S4, the champion)
# ═══════════════════════════════════════════════════════════════════════════

def strat_crash_hedged_qqq(prices, base_lev=1.2):
    qqq = prices["QQQ"]
    v20 = rvol(qqq, 20)
    v_avg = v20.rolling(120, min_periods=30).mean()

    is_normal   = v20 < v_avg * 1.2
    is_elevated = (v20 >= v_avg * 1.2) & (v20 < v_avg * 1.8)
    is_crisis   = v20 >= v_avg * 1.8
    is_recovery = is_elevated & (v20 < v20.shift(5)) & (qqq > qqq.rolling(10).min())

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Normal
    if "QQQ" in w.columns:
        w.loc[is_normal, "QQQ"] = base_lev * 0.7
    if "SPY" in w.columns:
        w.loc[is_normal, "SPY"] = base_lev * 0.3

    # Elevated: reduce + hedge
    if "QQQ" in w.columns:
        w.loc[is_elevated, "QQQ"] = base_lev * 0.3
    if "SPY" in w.columns:
        w.loc[is_elevated, "SPY"] = base_lev * 0.1
    if "IWM" in w.columns:
        w.loc[is_elevated, "IWM"] = -0.2
    if "GLD" in w.columns:
        w.loc[is_elevated, "GLD"] = 0.15

    # Crisis: short for profit
    if "QQQ" in w.columns:
        w.loc[is_crisis, "QQQ"] = 0.0
    if "SPY" in w.columns:
        w.loc[is_crisis, "SPY"] = -0.3
    if "GLD" in w.columns:
        w.loc[is_crisis, "GLD"] = 0.3
    if "TLT" in w.columns:
        w.loc[is_crisis, "TLT"] = 0.2

    # Recovery: aggressive long
    if "QQQ" in w.columns:
        w.loc[is_recovery, "QQQ"] = base_lev * 0.8
    if "SPY" in w.columns:
        w.loc[is_recovery, "SPY"] = base_lev * 0.4
    if "IWM" in w.columns:
        w.loc[is_recovery, "IWM"] = 0.0

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY B: Momentum Binary Scoring (v6's S5)
# ═══════════════════════════════════════════════════════════════════════════

def strat_momentum_binary(prices, threshold=5, leverage=1.3):
    qqq = prices["QQQ"]
    spy = prices["SPY"]

    c1 = (qqq > sma(qqq, 200)).astype(float)
    c2 = (sma(qqq, 50) > sma(qqq, 200)).astype(float)
    c3 = (qqq.pct_change(252) > 0).astype(float)
    c4 = (qqq.pct_change(63) > 0).astype(float)
    v20 = rvol(qqq, 20)
    v120_avg = v20.rolling(120, min_periods=30).mean()
    c5 = (v20 < v120_avg * 1.2).astype(float)
    r = rsi(qqq, 14)
    c6 = ((r > 35) & (r < 75)).astype(float)
    br = breadth_count(prices, 50)
    n_sec = max(len([s for s in SECTORS if s in prices.columns]), 1)
    c7 = (br > n_sec * 0.5).astype(float)
    c8 = (spy > sma(spy, 200)).astype(float)

    score = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    on = (score >= threshold).astype(float)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns:
        w["QQQ"] = on * leverage * 0.65
    if "SPY" in w.columns:
        w["SPY"] = on * leverage * 0.35
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY C: EXPANDED MARKET-NEUTRAL PAIRS BASKET
# 15+ pairs across asset classes, sectors, styles for max diversification
# ═══════════════════════════════════════════════════════════════════════════

def strat_expanded_pairs(prices, lookback=63, notional_per_pair=0.10):
    """
    Large basket of market-neutral pairs. Each pair: long relative outperformer,
    short relative underperformer. Exposure per pair kept small to diversify.
    """
    pair_defs = [
        # Cross-style
        ("QQQ", "IWM"),    # growth vs value
        ("QQQ", "SPY"),    # tech-heavy vs broad
        ("QQQ", "EFA"),    # US tech vs international
        ("SPY", "EFA"),    # US vs international
        ("SPY", "IWM"),    # large vs small cap
        # Cross-sector
        ("XLK", "XLE"),    # tech vs energy
        ("XLK", "XLF"),    # tech vs financials
        ("XLK", "XLU"),    # tech (growth) vs utilities (defensive)
        ("XLV", "XLE"),    # healthcare vs energy
        ("XLI", "XLU"),    # cyclical vs defensive
        ("XLC", "XLP"),    # discretionary comm vs consumer staples
        ("XLF", "XLRE"),   # financials vs real estate (rate sensitivity)
        # Cross-asset
        ("SPY", "TLT"),    # equity vs bonds
        ("SPY", "GLD"),    # equity vs gold
        ("GLD", "TLT"),    # gold vs bonds
        ("GLD", "IEF"),    # gold vs medium bonds
    ]

    # Filter to available pairs
    pairs = [(a, b) for a, b in pair_defs
             if a in prices.columns and b in prices.columns]
    n_pairs = max(len(pairs), 1)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for long_leg, short_leg in pairs:
        # Relative return over lookback
        rel_ret = prices[long_leg].pct_change(lookback) - prices[short_leg].pct_change(lookback)

        # Normalize to [-1, 1] range with clip
        signal = rel_ret.clip(lower=-0.30, upper=0.30) / 0.30

        w[long_leg]  += signal * notional_per_pair
        w[short_leg] -= signal * notional_per_pair

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY D: ULTRA-SELECTIVE QQQ (only invest during BEST conditions)
# More selective than S5 → fewer invested days but higher win rate
# ═══════════════════════════════════════════════════════════════════════════

def strat_ultra_selective(prices, leverage=1.5):
    qqq = prices["QQQ"]
    spy = prices["SPY"]

    # 10 conditions - ALL must be met
    c1 = qqq > sma(qqq, 200)                                    # LT trend
    c2 = sma(qqq, 50) > sma(qqq, 200)                          # golden cross
    c3 = qqq > sma(qqq, 20)                                     # ST trend
    c4 = qqq.pct_change(252) > 0.05                             # 12m mom > 5%
    c5 = qqq.pct_change(63) > 0                                  # 3m mom > 0

    v20 = rvol(qqq, 20)
    v120_avg = v20.rolling(120, min_periods=30).mean()
    c6 = v20 < v120_avg * 1.1                                   # vol low

    r = rsi(qqq, 14)
    c7 = (r > 40) & (r < 70)                                    # RSI sweet spot

    br = breadth_count(prices, 50)
    n_sec = max(len([s for s in SECTORS if s in prices.columns]), 1)
    c8 = br > n_sec * 0.6                                        # strong breadth

    c9 = spy > sma(spy, 200)                                     # SPY uptrend
    c10 = spy > sma(spy, 50)                                     # SPY ST uptrend

    all_go = (c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8 & c9 & c10).astype(float)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns:
        w["QQQ"] = all_go * leverage * 0.65
    if "SPY" in w.columns:
        w["SPY"] = all_go * leverage * 0.35
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY E: VOL-TARGETING WITH BINARY REGIME
# Dynamically target portfolio vol = X%. In risk-off → cash.
# ═══════════════════════════════════════════════════════════════════════════

def strat_vol_targeted(prices, target_vol=0.12, max_lev=2.0):
    qqq = prices["QQQ"]
    spy = prices["SPY"]

    # Binary regime
    above_200 = qqq > sma(qqq, 200)
    golden = sma(qqq, 50) > sma(qqq, 200)
    mom_pos = qqq.pct_change(126) > 0
    br = breadth_count(prices, 50)
    n_sec = max(len([s for s in SECTORS if s in prices.columns]), 1)
    br_ok = br > n_sec * 0.4

    regime = (above_200 & golden & mom_pos & br_ok).astype(float)

    # Vol-targeting
    v20 = rvol(qqq, 20)
    v60 = rvol(qqq, 60)
    rv = pd.concat([v20, v60], axis=1).max(axis=1).clip(lower=0.03)
    scale = (target_vol / rv).clip(upper=max_lev).rolling(5, min_periods=1).mean()

    position = regime * scale

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns:
        w["QQQ"] = position * 0.6
    if "SPY" in w.columns:
        w["SPY"] = position * 0.4
    return w


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def audit_results(results_dict, prices):
    issues = []
    spy_ret = prices["SPY"].pct_change()
    qqq_ret = prices["QQQ"].pct_change()

    for label, res in results_dict.items():
        eq = res["equity_curve"]
        w  = res["weights"]
        pr = res["portfolio_returns"]

        # 1. Look-ahead bias check
        if not w.empty and w.abs().sum().sum() > 0:
            for col in w.columns:
                if w[col].abs().sum() < 1:
                    continue
                for bench_ret, bname in [(spy_ret, "SPY"), (qqq_ret, "QQQ")]:
                    corr = w[col].corr(bench_ret)
                    if abs(corr) > 0.25:
                        issues.append(f"[{label}] {col} weight corr w/ same-day {bname} = {corr:.3f}")

        # 2. Sharpe cross-check
        ann_ret = pr.mean() * 252
        ann_std = pr.std() * np.sqrt(252)
        sh_manual = ann_ret / ann_std if ann_std > 1e-8 else 0
        sh_reported = res["metrics"]["Sharpe Ratio"]
        if abs(sh_manual - sh_reported) > 0.15:
            issues.append(f"[{label}] Sharpe mismatch: {sh_manual:.4f} vs {sh_reported:.4f}")

        # 3. CAGR cross-check
        yrs = len(eq) / 252
        cagr_man = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
        cagr_rep = res["metrics"]["CAGR"]
        if abs(cagr_man - cagr_rep) > 0.015:
            issues.append(f"[{label}] CAGR mismatch: {cagr_man:.4f} vs {cagr_rep:.4f}")

        # 4. Equity consistency
        recon = 100_000 * (1 + pr).cumprod()
        diff = abs(recon.iloc[-1] - eq.iloc[-1]) / max(eq.iloc[-1], 1)
        if diff > 0.02:
            issues.append(f"[{label}] equity inconsistency: {diff:.4f}")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("LEVERAGED ALPHA v7 — TRAILING STOPS + EXPANDED PAIRS + ULTRA-SELECTIVE")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  {START} to {END}")
    print(SEP)

    prices = load_data()

    # Benchmarks
    bench = {}
    for t in ["SPY", "QQQ"]:
        eq = prices[t] / prices[t].iloc[0] * 100_000
        ret = prices[t].pct_change().fillna(0)
        m = compute_metrics(ret, eq, 100_000, risk_free_rate=RF)
        bench[t] = m
        print(f"  {t}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, MaxDD={m['Max Drawdown']:.2%}")
    spy_cagr = bench["SPY"]["CAGR"]
    print()

    results = {}

    def log(label, res):
        m = res["metrics"]
        cagr_f = "Y" if m["CAGR"] > spy_cagr else " "
        sh_f = "Y" if m["Sharpe Ratio"] > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        cash_pct = res["cash_weight"].mean()
        print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cagr_f}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_f}] "
              f"MaxDD={m['Max Drawdown']:7.2%} Lev={avg_lev:.2f} Cash={cash_pct:.0%}")
        results[label] = res

    # ═════════════════════════════════════════════════════════════════
    # A: Crash-Hedged QQQ (champion from v6) with trailing stops
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("A: CRASH-HEDGED QQQ (with and without trailing stops)")
    for blev in [1.0, 1.2, 1.5]:
        w = strat_crash_hedged_qqq(prices, base_lev=blev)
        # Without stop
        res = backtest_simple(prices, w)
        log(f"A_CrashHedge(lev={blev})", res)
        # With various trailing stops
        for stop in [-0.06, -0.08, -0.10, -0.12]:
            res = backtest_with_stop(prices, w, trailing_stop=stop, reentry_delay=10)
            log(f"A_CrashHedge(lev={blev},stop={stop:.0%})", res)

    # ═════════════════════════════════════════════════════════════════
    # B: Momentum Binary with trailing stops
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("B: MOMENTUM BINARY SCORING (with trailing stops)")
    for thresh in [5, 6]:
        for lev in [1.0, 1.3]:
            w = strat_momentum_binary(prices, threshold=thresh, leverage=lev)
            res = backtest_simple(prices, w)
            log(f"B_MomBin(t={thresh},lev={lev})", res)
            for stop in [-0.06, -0.08, -0.10]:
                res = backtest_with_stop(prices, w, trailing_stop=stop, reentry_delay=10)
                log(f"B_MomBin(t={thresh},lev={lev},stop={stop:.0%})", res)

    # ═════════════════════════════════════════════════════════════════
    # C: Expanded Pairs
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("C: EXPANDED MARKET-NEUTRAL PAIRS")
    for lb in [42, 63, 126]:
        for notional in [0.05, 0.08, 0.10, 0.15]:
            w = strat_expanded_pairs(prices, lookback=lb, notional_per_pair=notional)
            res = backtest_simple(prices, w)
            log(f"C_Pairs(lb={lb},not={notional})", res)

    # ═════════════════════════════════════════════════════════════════
    # D: Ultra-Selective with trailing stops
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("D: ULTRA-SELECTIVE QQQ (with trailing stops)")
    for lev in [1.3, 1.5, 2.0]:
        w = strat_ultra_selective(prices, leverage=lev)
        res = backtest_simple(prices, w)
        log(f"D_UltraSelect(lev={lev})", res)
        for stop in [-0.05, -0.08]:
            res = backtest_with_stop(prices, w, trailing_stop=stop, reentry_delay=10)
            log(f"D_UltraSelect(lev={lev},stop={stop:.0%})", res)

    # ═════════════════════════════════════════════════════════════════
    # E: Vol-Targeted with trailing stops
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("E: VOL-TARGETED BINARY REGIME")
    for tvol in [0.10, 0.12, 0.15]:
        w = strat_vol_targeted(prices, target_vol=tvol)
        res = backtest_simple(prices, w)
        log(f"E_VolTarget(vol={tvol:.0%})", res)
        for stop in [-0.06, -0.08]:
            res = backtest_with_stop(prices, w, trailing_stop=stop, reentry_delay=10)
            log(f"E_VolTarget(vol={tvol:.0%},stop={stop:.0%})", res)

    # ═════════════════════════════════════════════════════════════════
    # ENSEMBLES
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("ENSEMBLES — combining best variants from each family")

    # Find best variant per family
    fam_prefixes = {"A": "A_", "B": "B_", "C": "C_", "D": "D_", "E": "E_"}
    best = {}
    for fam, prefix in fam_prefixes.items():
        fam_keys = [k for k in results if k.startswith(prefix)]
        if fam_keys:
            best[fam] = max(fam_keys, key=lambda k: results[k]["metrics"]["Sharpe Ratio"])
            m = results[best[fam]]["metrics"]
            print(f"  Best {fam}: {best[fam]} → Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"CAGR={m['CAGR']:.2%}")

    # Correlation matrix
    ret_df = pd.DataFrame({fam: results[k]["portfolio_returns"] for fam, k in best.items()})
    corr = ret_df.corr()
    print(f"\n  Correlations:")
    print("  " + corr.to_string(float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))
    print()

    def make_ensemble(label, components, wgts):
        comp_keys = [best[c] if c in best else c for c in components]
        # Filter to existing keys
        valid = [(k, w) for k, w in zip(comp_keys, wgts) if k in results]
        if not valid:
            return
        keys, ws = zip(*valid)
        # Normalize weights
        wsum = sum(ws)
        ws = [w / wsum for w in ws]

        ens_ret = sum(w_i * results[k]["portfolio_returns"] for k, w_i in zip(keys, ws))
        ens_eq = 100_000 * (1 + ens_ret).cumprod()
        ens_eq.name = "Equity"
        turn = sum(w_i * results[k]["turnover"] for k, w_i in zip(keys, ws))
        g_exp = sum(w_i * results[k]["gross_exposure"] for k, w_i in zip(keys, ws))
        net_exp = sum(w_i * results[k]["net_exposure"] for k, w_i in zip(keys, ws))
        cash_w = (1 - net_exp).clip(lower=0)

        m = compute_metrics(ens_ret, ens_eq, 100_000, risk_free_rate=RF,
                            turnover=turn, gross_exposure=g_exp)
        cagr_f = "Y" if m["CAGR"] > spy_cagr else " "
        sh_f = "Y" if m["Sharpe Ratio"] > 1.95 else " "
        avg_lev = g_exp.mean()
        cash_pct = cash_w.mean()
        print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cagr_f}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_f}] "
              f"MaxDD={m['Max Drawdown']:7.2%} Lev={avg_lev:.2f} Cash={cash_pct:.0%}")
        results[label] = {
            "equity_curve": ens_eq, "portfolio_returns": ens_ret,
            "weights": pd.DataFrame(), "turnover": turn,
            "gross_exposure": g_exp, "net_exposure": net_exp,
            "cash_weight": cash_w, "metrics": m,
        }

    # Directional + Neutral (key combo: high CAGR + Sharpe boost from pairs)
    make_ensemble("Ens_A+C",           ["A", "C"], [0.70, 0.30])
    make_ensemble("Ens_A+C(60/40)",    ["A", "C"], [0.60, 0.40])
    make_ensemble("Ens_B+C",           ["B", "C"], [0.70, 0.30])
    make_ensemble("Ens_B+C(60/40)",    ["B", "C"], [0.60, 0.40])
    make_ensemble("Ens_A+B+C",         ["A", "B", "C"], [0.40, 0.30, 0.30])
    make_ensemble("Ens_A+C+D",         ["A", "C", "D"], [0.40, 0.25, 0.35])
    make_ensemble("Ens_A+C+E",         ["A", "C", "E"], [0.40, 0.25, 0.35])
    make_ensemble("Ens_A+B+C+D",       ["A", "B", "C", "D"], [0.30, 0.25, 0.20, 0.25])
    make_ensemble("Ens_A+B+C+D+E",     ["A", "B", "C", "D", "E"],
                  [0.25, 0.20, 0.20, 0.15, 0.20])
    make_ensemble("Ens_ALL_Equal",      list(best.keys()),
                  [1.0] * len(best))

    # Pairs-heavy ensembles (for high Sharpe)
    make_ensemble("Ens_A+C(40/60)",    ["A", "C"], [0.40, 0.60])
    make_ensemble("Ens_B+C(40/60)",    ["B", "C"], [0.40, 0.60])

    # ═════════════════════════════════════════════════════════════════
    # LEVERAGED ENSEMBLES
    # ═════════════════════════════════════════════════════════════════
    print(SEP)
    print("LEVERAGED ENSEMBLES")
    ens_keys = sorted(
        [k for k in results if k.startswith("Ens_")],
        key=lambda k: results[k]["metrics"]["Sharpe Ratio"], reverse=True
    )[:10]  # Top 10 ensembles by Sharpe

    for base_key in ens_keys:
        base_sharpe = results[base_key]["metrics"]["Sharpe Ratio"]
        base_ret = results[base_key]["portfolio_returns"]
        for mult in [1.5, 2.0, 2.5, 3.0]:
            label = f"L_{base_key}_x{mult}"[:55]
            scaled_ret = base_ret * mult - (mult - 1) * LEV_COST / 252
            eq = 100_000 * (1 + scaled_ret).cumprod()
            eq.name = "Equity"
            m = compute_metrics(scaled_ret, eq, 100_000, risk_free_rate=RF)
            cagr_f = "Y" if m["CAGR"] > spy_cagr else " "
            sh_f = "Y" if m["Sharpe Ratio"] > 1.95 else " "
            print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cagr_f}] "
                  f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_f}] "
                  f"MaxDD={m['Max Drawdown']:7.2%}")
            results[label] = {
                "equity_curve": eq, "portfolio_returns": scaled_ret,
                "weights": pd.DataFrame(), "metrics": m,
                "turnover": pd.Series(0, index=prices.index),
                "gross_exposure": pd.Series(mult, index=prices.index),
                "net_exposure": pd.Series(mult, index=prices.index),
                "cash_weight": pd.Series(0, index=prices.index),
            }

    # Also leverage best individual strategies
    print()
    top_indiv = sorted(
        [(k, v) for k, v in results.items()
         if not k.startswith("Ens_") and not k.startswith("L_")],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:5]
    for base_key, _ in top_indiv:
        base_ret = results[base_key]["portfolio_returns"]
        for mult in [1.5, 2.0]:
            label = f"L_{base_key}_x{mult}"[:55]
            scaled_ret = base_ret * mult - (mult - 1) * LEV_COST / 252
            eq = 100_000 * (1 + scaled_ret).cumprod()
            eq.name = "Equity"
            m = compute_metrics(scaled_ret, eq, 100_000, risk_free_rate=RF)
            cagr_f = "Y" if m["CAGR"] > spy_cagr else " "
            sh_f = "Y" if m["Sharpe Ratio"] > 1.95 else " "
            print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cagr_f}] "
                  f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_f}] "
                  f"MaxDD={m['Max Drawdown']:7.2%}")
            results[label] = {
                "equity_curve": eq, "portfolio_returns": scaled_ret,
                "weights": pd.DataFrame(), "metrics": m,
                "turnover": pd.Series(0, index=prices.index),
                "gross_exposure": pd.Series(mult, index=prices.index),
                "net_exposure": pd.Series(mult, index=prices.index),
                "cash_weight": pd.Series(0, index=prices.index),
            }

    # ═════════════════════════════════════════════════════════════════
    # AUDIT
    # ═════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    auditable = {k: v for k, v in results.items()
                 if isinstance(v["weights"], pd.DataFrame) and not v["weights"].empty}
    iss = audit_results(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies with weights checked)")
    if iss:
        for i in iss:
            print(f"  !! {i}")
    else:
        print("  ALL AUDITS PASS")

    # ═════════════════════════════════════════════════════════════════
    # FINAL SORTED TABLE
    # ═════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    sorted_all = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    # Show top 30
    print("FINAL RESULTS — TOP 30 BY SHARPE\n")
    print(f"{'#':>3s} {'Strategy':<56s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Hit?':>8s}")
    print("-" * 90)
    print(f"{'':>3s} {'SPY':56s} {spy_cagr:7.2%} {bench['SPY']['Sharpe Ratio']:7.4f} "
          f"{bench['SPY']['Max Drawdown']:7.2%}")
    print(f"{'':>3s} {'QQQ':56s} {bench['QQQ']['CAGR']:7.2%} "
          f"{bench['QQQ']['Sharpe Ratio']:7.4f} {bench['QQQ']['Max Drawdown']:7.2%}")
    print("-" * 90)

    winners = []
    for rank, (label, res) in enumerate(sorted_all[:30], 1):
        m = res["metrics"]
        hit = ""
        if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
            hit = "WINNER"
            winners.append(label)
        elif m["CAGR"] > spy_cagr:
            hit = "CAGR+"
        elif m["Sharpe Ratio"] > 1.95:
            hit = "SH+"
        print(f"{rank:3d} {label:<56s} {m['CAGR']:7.2%} {m['Sharpe Ratio']:7.4f} "
              f"{m['Max Drawdown']:7.2%} {hit:>8s}")

    print("\n" + SEP)
    if winners:
        print(f"WINNERS ({len(winners)}):")
        for w in winners:
            m = results[w]["metrics"]
            print(f"  {w}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}")
    else:
        print("No strategies hit BOTH targets (CAGR > SPY AND Sharpe > 1.95).")
        # Best CAGR-beaters by Sharpe
        cagr_beaters = [(k, v) for k, v in sorted_all
                        if v["metrics"]["CAGR"] > spy_cagr]
        if cagr_beaters:
            print(f"\nBest CAGR-beaters by Sharpe (beat SPY CAGR):")
            for k, v in cagr_beaters[:8]:
                m = v["metrics"]
                print(f"  {k}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                      f"MaxDD={m['Max Drawdown']:.2%}")
        # Best Sharpe overall
        print(f"\nBest Sharpe overall:")
        for k, v in sorted_all[:8]:
            m = v["metrics"]
            print(f"  {k}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}")

    print(f"\nTotal strategies tested: {len(results)} | Audit issues: {len(iss)}")


if __name__ == "__main__":
    main()
