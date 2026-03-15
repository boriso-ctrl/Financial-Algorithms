"""
Leveraged Alpha v6 — QQQ-Centric + Asymmetric Binary Timing + Pairs + Leverage
================================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95  |  2010-01-01 to 2025-03-01

Key changes from v5 (which achieved best Sharpe = 0.84):
  v5 failures:
    - Gradual positioning = always partially invested during both good AND bad days
    - Vol-management capped upside as much as downside
    - Bond/gold hedges dragged returns in 2010-2025 bull market
    - DrawDown controls reduced CAGR too much for the Sharpe improvement

  v6 philosophy:
    1. QQQ as primary (CAGR ~18-19%, higher than SPY ~13.6%)
    2. BINARY regime switching: 100% invested or 100% cash (no in-between)
       - Mathematically: eliminates ALL left-tail days → massively lowers vol
       - While keeping ALL right-tail days when invested
    3. ASYMMETRIC timing: fast exit (ANY 1 of N signals), slow entry (ALL M of M)
       - Market crashes are fast → need quick exits
       - Recoveries are slow → can afford slow re-entry, catch sustained rallys
    4. Market-neutral pairs overlay for uncorrelated alpha
    5. Leverage ONLY during highest-conviction regimes (score = max)
    6. Daily rebalancing (not weekly) for faster reaction

  Expected improvements:
    - Binary timing should push Sharpe from 0.84 to 1.2-1.5 range
    - QQQ base should push CAGR above SPY even with timing misses
    - Pairs add uncorrelated return stream
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
TX_BPS     = 5           # transaction cost per trade (one-way)
LEV_COST   = 0.015       # annual cost for leverage > 1x
SHORT_COST = 0.005       # annual short borrow cost
RF_CASH    = 0.02        # annual return on cash (approximate)
RF = 0.0
SEP = "─" * 80

# ── Data ──
def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers loaded: {list(p.columns)}")
    print(f"  {len(p)} days ({p.index[0].strftime('%Y-%m-%d')} to "
          f"{p.index[-1].strftime('%Y-%m-%d')}, {len(p)/252:.1f}yr)\n")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def sma(s, n):    return s.rolling(n, min_periods=n).mean()
def ema(s, n):    return s.ewm(span=n, min_periods=n).mean()

def realized_vol(s, n=21):
    return s.pct_change().rolling(n, min_periods=max(10, n // 2)).std() * np.sqrt(252)

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.clip(lower=1e-10))

def breadth_count(prices, lookback=50):
    """Count sector ETFs above N-day MA. Returns 0..N."""
    sectors = [s for s in SECTORS if s in prices.columns]
    b = pd.Series(0.0, index=prices.index)
    for s in sectors:
        b += (prices[s] > sma(prices[s], lookback)).astype(float)
    return b

def max_drawdown_rolling(equity, window=252):
    """Rolling max drawdown over window."""
    peak = equity.rolling(window, min_periods=1).max()
    return (equity - peak) / peak


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (handles leverage, shorts, transaction costs, cash return)
# ═══════════════════════════════════════════════════════════════════════════

def backtest(prices, weights, rebal=1, cap=100_000.0):
    """
    Daily backtest with proper cost modeling.
    weights: DataFrame aligned to prices.index.
    Positive = long, negative = short. Can sum > 1 (leverage).
    Cash earns RF_CASH. shift(1) applied to prevent look-ahead.
    """
    common = prices.columns.intersection(weights.columns)
    p = prices[common]
    w = weights[common].reindex(prices.index).fillna(0)

    # Apply rebalance frequency
    if rebal > 1:
        mask = pd.Series(False, index=w.index)
        mask.iloc[::rebal] = True
        w = w.where(mask).ffill().fillna(0)

    # CRITICAL: shift(1) to prevent look-ahead bias
    w = w.shift(1).fillna(0)

    ret = p.pct_change().fillna(0)

    # Daily portfolio return
    port_ret = (w * ret).sum(axis=1)

    # Cash return: uninvested capital earns risk-free
    net_exposure = w.sum(axis=1)  # can be < 1 (partial cash) or > 1 (leverage)
    cash_weight  = (1 - net_exposure).clip(lower=0)
    cash_ret     = cash_weight * RF_CASH / 252

    # Costs
    turnover = w.diff().fillna(0).abs().sum(axis=1)
    tx_cost  = turnover * TX_BPS / 10_000

    gross_exp = w.abs().sum(axis=1)
    lev_cost  = (gross_exp - 1).clip(lower=0) * LEV_COST / 252
    short_exp = w.clip(upper=0).abs().sum(axis=1)
    short_cost = short_exp * SHORT_COST / 252

    net_ret = port_ret + cash_ret - tx_cost - lev_cost - short_cost
    eq = cap * (1 + net_ret).cumprod()
    eq.name = "Equity"

    m = compute_metrics(net_ret, eq, cap, risk_free_rate=RF,
                        turnover=turnover, gross_exposure=gross_exp)

    return {
        "equity_curve": eq, "portfolio_returns": net_ret, "weights": w,
        "turnover": turnover, "gross_exposure": gross_exp, "metrics": m,
        "net_exposure": net_exposure, "cash_weight": cash_weight,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: QQQ ASYMMETRIC BINARY TIMING
# Philosophy: When bullish → all-in QQQ. When any danger signal → 100% cash.
# Fast exit (ANY 1 bad signal), slow entry (ALL signals must be green).
# ═══════════════════════════════════════════════════════════════════════════

def strat1_qqq_binary(prices, leverage=1.0, spy_weight=0.0):
    """
    Binary QQQ timing with asymmetric entry/exit.
    leverage: multiplier when invested (1.0 = no leverage)
    spy_weight: fraction of equity allocation to SPY (rest goes to QQQ)
    """
    qqq = prices["QQQ"]
    spy = prices["SPY"]

    # ────── EXIT signals (ANY one triggers full exit) ──────
    # E1: QQQ below 200-day MA
    e1 = qqq < sma(qqq, 200)
    # E2: QQQ 50-day MA crossed below 200-day MA (death cross)
    e2 = sma(qqq, 50) < sma(qqq, 200)
    # E3: Short-term momentum collapse (10-day return < -5%)
    e3 = qqq.pct_change(10) < -0.05
    # E4: Breadth collapse (< 30% of sectors above 50d MA)
    br = breadth_count(prices, 50)
    n_sec = max(len([s for s in SECTORS if s in prices.columns]), 1)
    e4 = br < n_sec * 0.3
    # E5: Volatility spike (20d vol > 1.5x its 120d avg)
    v20 = realized_vol(qqq, 20)
    v120_avg = v20.rolling(120, min_periods=30).mean()
    e5 = v20 > v120_avg * 1.5

    # Any exit signal → risk-off
    exit_signal = (e1 | e2 | e3 | e4 | e5).astype(float)

    # ────── ENTRY signals (ALL must be green to go risk-on) ──────
    # N1: QQQ above 200-day MA
    n1 = qqq > sma(qqq, 200)
    # N2: QQQ 50-day MA above 200-day MA (golden cross)
    n2 = sma(qqq, 50) > sma(qqq, 200)
    # N3: QQQ 12-month momentum > 0
    n3 = qqq.pct_change(252) > 0
    # N4: Breadth > 50%
    n4 = br > n_sec * 0.5
    # N5: RSI(14) > 40 (not deeply oversold still falling)
    n5 = rsi(qqq, 14) > 40
    # N6: 20-day vol not extreme (< 1.3x 120d avg)
    n6 = v20 < v120_avg * 1.3

    # All entry signals must confirm
    entry_signal = (n1 & n2 & n3 & n4 & n5 & n6).astype(float)

    # ────── REGIME ──────
    # regime = 1 (risk-on) or 0 (risk-off)
    # Start with exit signal having priority (conservative)
    regime = pd.Series(0.0, index=prices.index)
    for i in range(1, len(regime)):
        if exit_signal.iloc[i] > 0:
            regime.iloc[i] = 0  # exit overrides everything
        elif entry_signal.iloc[i] > 0:
            regime.iloc[i] = 1  # all entry signals green
        else:
            regime.iloc[i] = regime.iloc[i - 1]  # hold previous state

    # ────── WEIGHTS ──────
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    qqq_frac = 1.0 - spy_weight
    if "QQQ" in w.columns:
        w["QQQ"] = regime * leverage * qqq_frac
    if "SPY" in w.columns:
        w["SPY"] = regime * leverage * spy_weight
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: MULTI-ASSET BINARY TREND
# Each asset independently: if above 200MA AND positive 6m mom → long, else cash
# Weight by inverse-vol. Apply leverage to hit target vol.
# ═══════════════════════════════════════════════════════════════════════════

def strat2_multi_binary_trend(prices, target_vol=0.15, max_lev=2.0):
    assets = [c for c in prices.columns if c in BROAD + ["GLD", "TLT"]]
    ret = prices[assets].pct_change()
    vol = ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    vol = vol.clip(lower=0.01)

    # Binary trend per asset
    signals = pd.DataFrame(0.0, index=prices.index, columns=assets)
    for c in assets:
        above_ma = prices[c] > sma(prices[c], 200)
        mom_pos  = prices[c].pct_change(126) > 0
        signals[c] = (above_ma & mom_pos).astype(float)

    # Inverse-vol weighting for active assets
    inv_vol = 1.0 / vol
    raw_w = (signals * inv_vol)
    wsum = raw_w.sum(axis=1).clip(lower=0.01)
    raw_w = raw_w.div(wsum, axis=0)

    # Replace NaN with 0 (warmup period)
    raw_w = raw_w.fillna(0)

    # Scale to target vol
    port_ret_approx = (raw_w.shift(1).fillna(0) * ret).sum(axis=1)
    pvol = port_ret_approx.rolling(60, min_periods=20).std() * np.sqrt(252)
    pvol = pvol.clip(lower=0.02)
    scale = (target_vol / pvol).clip(upper=max_lev).rolling(5, min_periods=1).mean()
    raw_w = raw_w.mul(scale, axis=0)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in assets:
        w[c] = raw_w[c]
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: MARKET-NEUTRAL PAIRS
# Long outperformer / short underperformer within related asset groups.
# Provides genuinely uncorrelated returns.
# ═══════════════════════════════════════════════════════════════════════════

def strat3_pairs(prices, lookback=63, target_notional=0.5):
    """
    Market-neutral pairs: long relative strength leader, short laggard.
    Pairs: QQQ/IWM (growth/value), XLK/XLE (tech/energy), SPY/EFA (US/intl)
    """
    pair_defs = [
        ("QQQ", "IWM"),   # growth vs value
        ("XLK", "XLE"),   # tech vs energy
        ("SPY", "EFA"),   # US vs international
    ]

    # Filter to pairs where both legs exist
    pairs = [(a, b) for a, b in pair_defs
             if a in prices.columns and b in prices.columns]

    n_pairs = max(len(pairs), 1)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for long_leg, short_leg in pairs:
        # Relative momentum: which leg outperformed over lookback?
        rel_ret = prices[long_leg].pct_change(lookback) - prices[short_leg].pct_change(lookback)

        # Long the outperformer, short the underperformer
        # Scale position by strength of relative momentum
        signal = rel_ret.clip(lower=-0.30, upper=0.30) / 0.30  # normalized to [-1, 1]

        pos_size = target_notional / n_pairs

        # When signal > 0: long long_leg, short short_leg
        # When signal < 0: long short_leg, short long_leg
        w[long_leg]  += signal * pos_size
        w[short_leg] -= signal * pos_size

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: CRASH HEDGED QQQ
# Base: long QQQ. Hedge: short SPY (or reduce position) when vol spikes.
# The short provides positive returns during crashes.
# ═══════════════════════════════════════════════════════════════════════════

def strat4_crash_hedged(prices, base_lev=1.2):
    qqq = prices["QQQ"]
    spy = prices["SPY"]

    # Volatility regime
    v20 = realized_vol(qqq, 20)
    v60 = realized_vol(qqq, 60)
    v_avg = v20.rolling(120, min_periods=30).mean()

    # Regime detection
    # Normal: v20 < v_avg * 1.2 → full long
    # Elevated: v_avg * 1.2 < v20 < v_avg * 1.8 → reduced, add hedge
    # Crisis: v20 > v_avg * 1.8 → minimal long, full hedge
    # Recovery: v20 declining from crisis AND price > 10d low → aggressive long

    is_normal   = v20 < v_avg * 1.2
    is_elevated = (v20 >= v_avg * 1.2) & (v20 < v_avg * 1.8)
    is_crisis   = v20 >= v_avg * 1.8
    is_recovery = is_elevated & (v20 < v20.shift(5)) & (qqq > qqq.rolling(10).min())

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Normal: full long QQQ
    if "QQQ" in w.columns:
        w.loc[is_normal, "QQQ"] = base_lev * 0.7
    if "SPY" in w.columns:
        w.loc[is_normal, "SPY"] = base_lev * 0.3

    # Elevated: reduce long, add short hedge
    if "QQQ" in w.columns:
        w.loc[is_elevated, "QQQ"] = base_lev * 0.3
    if "SPY" in w.columns:
        w.loc[is_elevated, "SPY"] = base_lev * 0.1
    if "IWM" in w.columns:
        w.loc[is_elevated, "IWM"] = -0.2  # short small caps as hedge
    if "GLD" in w.columns:
        w.loc[is_elevated, "GLD"] = 0.15

    # Crisis: minimal long, short for profit
    if "QQQ" in w.columns:
        w.loc[is_crisis, "QQQ"] = 0.0
    if "SPY" in w.columns:
        w.loc[is_crisis, "SPY"] = -0.3  # short SPY in crisis
    if "GLD" in w.columns:
        w.loc[is_crisis, "GLD"] = 0.3
    if "TLT" in w.columns:
        w.loc[is_crisis, "TLT"] = 0.2

    # Recovery override: aggressive long
    if "QQQ" in w.columns:
        w.loc[is_recovery, "QQQ"] = base_lev * 0.8
    if "SPY" in w.columns:
        w.loc[is_recovery, "SPY"] = base_lev * 0.4
    # Clear any shorts in recovery
    if "IWM" in w.columns:
        w.loc[is_recovery, "IWM"] = 0.0

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: ADAPTIVE MOMENTUM SCORING (improved v3 S2_Scoring)
# Key difference from v5: NO gradual positioning. Score → binary threshold.
# Uses QQQ instead of SPY, and adds breadth + vol conditions.
# ═══════════════════════════════════════════════════════════════════════════

def strat5_momentum_binary(prices, threshold=5, leverage=1.3):
    """
    8-condition scoring on QQQ. If score >= threshold → all-in at leverage.
    If score < threshold → 100% cash. Binary switch.
    """
    qqq = prices["QQQ"]
    spy = prices["SPY"]

    c1 = (qqq > sma(qqq, 200)).astype(float)           # LT trend
    c2 = (sma(qqq, 50) > sma(qqq, 200)).astype(float)  # golden cross
    c3 = (qqq.pct_change(252) > 0).astype(float)        # 12m momentum > 0
    c4 = (qqq.pct_change(63) > 0).astype(float)         # 3m momentum > 0

    v20 = realized_vol(qqq, 20)
    v120_avg = v20.rolling(120, min_periods=30).mean()
    c5 = (v20 < v120_avg * 1.2).astype(float)           # vol not elevated

    r = rsi(qqq, 14)
    c6 = ((r > 35) & (r < 75)).astype(float)            # RSI healthy range

    br = breadth_count(prices, 50)
    n_sec = max(len([s for s in SECTORS if s in prices.columns]), 1)
    c7 = (br > n_sec * 0.5).astype(float)               # breadth > 50%

    c8 = (spy > sma(spy, 200)).astype(float)             # SPY also in uptrend

    score = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8  # 0-8

    # Binary: all-in or all-out
    on = (score >= threshold).astype(float)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns:
        w["QQQ"] = on * leverage * 0.65
    if "SPY" in w.columns:
        w["SPY"] = on * leverage * 0.35
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 6: SECTOR ROTATION + BINARY MARKET REGIME
# When market is risk-on: long top-3 sectors by momentum.
# When risk-off: 100% cash.
# ═══════════════════════════════════════════════════════════════════════════

def strat6_sector_rotation(prices, n_top=3, leverage=1.2):
    spy = prices["SPY"]
    sectors = [s for s in SECTORS if s in prices.columns]

    # Market regime: binary (same as Strategy 1 logic)
    above_200 = spy > sma(spy, 200)
    golden    = sma(spy, 50) > sma(spy, 200)
    mom_12    = spy.pct_change(252) > 0
    regime = (above_200 & golden & mom_12).astype(float)

    # Sector momentum ranking (3-month)
    mom = prices[sectors].pct_change(63)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        if regime.iloc[i] < 0.5:
            continue  # risk-off → all cash
        row = mom.iloc[i].dropna()
        if len(row) < n_top:
            continue
        top = row.sort_values(ascending=False).head(n_top)
        for sec in top.index:
            if top[sec] > 0:  # only long positive momentum sectors
                w.loc[prices.index[i], sec] = leverage / n_top

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 7: DUAL MOMENTUM (absolute + relative)
# Classic Antonacci dual momentum adapted for ETFs.
# Long the best-performing of QQQ/SPY/EFA if it has positive absolute momentum.
# Otherwise: GLD/cash.
# ═══════════════════════════════════════════════════════════════════════════

def strat7_dual_momentum(prices, lookback=252, leverage=1.3):
    candidates = [c for c in ["QQQ", "SPY", "EFA"] if c in prices.columns]
    mom = prices[candidates].pct_change(lookback)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(lookback, len(prices)):
        row = mom.iloc[i].dropna()
        if len(row) == 0:
            continue
        best = row.idxmax()
        if row[best] > 0:
            # Best has positive absolute momentum → long it
            w.loc[prices.index[i], best] = leverage
        else:
            # All negative → go to GLD or cash
            if "GLD" in w.columns:
                w.loc[prices.index[i], "GLD"] = 0.5

    return w


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def audit(results_dict, prices):
    issues = []
    spy_ret = prices["SPY"].pct_change()
    qqq_ret = prices["QQQ"].pct_change()

    for label, res in results_dict.items():
        eq = res["equity_curve"]
        w  = res["weights"]
        pr = res["portfolio_returns"]

        # 1. Look-ahead: weights must not correlate with SAME-DAY returns
        for col in w.columns:
            if w[col].abs().sum() < 1:
                continue  # skip zero-weight columns
            corr_spy = w[col].corr(spy_ret)
            corr_qqq = w[col].corr(qqq_ret)
            if abs(corr_spy) > 0.25:
                issues.append(f"[{label}] {col} weight corr with same-day SPY = {corr_spy:.3f}")
            if abs(corr_qqq) > 0.25:
                issues.append(f"[{label}] {col} weight corr with same-day QQQ = {corr_qqq:.3f}")

        # 2. Sharpe cross-check
        ann_ret = pr.mean() * 252
        ann_std = pr.std() * np.sqrt(252)
        sharpe_manual = ann_ret / ann_std if ann_std > 1e-8 else 0
        sharpe_reported = res["metrics"].get("Sharpe Ratio", 0)
        if abs(sharpe_manual - sharpe_reported) > 0.15:
            issues.append(f"[{label}] Sharpe mismatch: manual={sharpe_manual:.4f} vs "
                          f"reported={sharpe_reported:.4f}")

        # 3. CAGR cross-check
        yrs = len(eq) / 252
        cagr_manual = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
        cagr_reported = res["metrics"].get("CAGR", 0)
        if abs(cagr_manual - cagr_reported) > 0.015:
            issues.append(f"[{label}] CAGR mismatch: manual={cagr_manual:.4f} vs "
                          f"reported={cagr_reported:.4f}")

        # 4. Equity consistency
        reconstructed = 100_000 * (1 + pr).cumprod()
        final_diff = abs(reconstructed.iloc[-1] - eq.iloc[-1]) / max(eq.iloc[-1], 1)
        if final_diff > 0.02:
            issues.append(f"[{label}] equity inconsistency: {final_diff:.4f}")

        # 5. Check for future data leakage in weights (weights should be constant
        #    when there's no rebalance)
        w_changes = w.diff().abs().sum(axis=1)
        # After shift(1) in backtest, row 0 should be 0
        if w.iloc[0].abs().sum() > 1e-10:
            issues.append(f"[{label}] Non-zero weights on day 0 (look-ahead risk)")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("LEVERAGED ALPHA v6 — QQQ-CENTRIC + ASYMMETRIC BINARY TIMING + PAIRS + LEVERAGE")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  {START} to {END}")
    print("=" * 90)

    prices = load_data()

    # ── Benchmarks ──
    benchmarks = {}
    for ticker in ["SPY", "QQQ"]:
        if ticker in prices.columns:
            eq = prices[ticker] / prices[ticker].iloc[0] * 100_000
            ret = prices[ticker].pct_change().fillna(0)
            m = compute_metrics(ret, eq, 100_000, risk_free_rate=RF)
            benchmarks[ticker] = m
            print(f"  Bench {ticker}: CAGR={m['CAGR']:.2%}, "
                  f"Sharpe={m['Sharpe Ratio']:.4f}, MaxDD={m['Max Drawdown']:.2%}")
    spy_cagr   = benchmarks["SPY"]["CAGR"]
    spy_sharpe = benchmarks["SPY"]["Sharpe Ratio"]
    print()

    results = {}

    def run_and_log(label, weights_df, rebal=1):
        res = backtest(prices, weights_df, rebal=rebal)
        m = res["metrics"]
        cagr_ok = "Y" if m["CAGR"] > spy_cagr else " "
        sh_ok   = "Y" if m["Sharpe Ratio"] > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        cash_pct = res["cash_weight"].mean()
        print(f"  {label:52s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} Lev={avg_lev:.2f} Cash={cash_pct:.0%}")
        results[label] = res
        return res

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 1: QQQ Binary Timing
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 1: QQQ Asymmetric Binary Timing")
    for lev in [1.0, 1.2, 1.5, 1.8]:
        for spy_w in [0.0, 0.3]:
            label = f"S1_QQQ(lev={lev},spy={spy_w:.0%})"
            w = strat1_qqq_binary(prices, leverage=lev, spy_weight=spy_w)
            run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 2: Multi-Asset Binary Trend
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 2: Multi-Asset Binary Trend")
    for tvol in [0.12, 0.15, 0.20]:
        label = f"S2_Trend(vol={tvol:.0%})"
        w = strat2_multi_binary_trend(prices, target_vol=tvol)
        run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 3: Market-Neutral Pairs
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 3: Market-Neutral Pairs")
    for notional in [0.3, 0.5, 0.8]:
        label = f"S3_Pairs(not={notional})"
        w = strat3_pairs(prices, target_notional=notional)
        run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 4: Crash-Hedged QQQ
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 4: Crash-Hedged QQQ")
    for blev in [1.0, 1.2, 1.5]:
        label = f"S4_CrashHedge(lev={blev})"
        w = strat4_crash_hedged(prices, base_lev=blev)
        run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 5: Momentum Binary Scoring
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 5: Momentum Binary Scoring")
    for thresh in [4, 5, 6, 7]:
        for lev in [1.0, 1.3, 1.5]:
            label = f"S5_MomBin(t={thresh},lev={lev})"
            w = strat5_momentum_binary(prices, threshold=thresh, leverage=lev)
            run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 6: Sector Rotation + Binary
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 6: Sector Rotation + Binary Market Regime")
    for n in [2, 3, 4]:
        for lev in [1.0, 1.3]:
            label = f"S6_SectorRot(n={n},lev={lev})"
            w = strat6_sector_rotation(prices, n_top=n, leverage=lev)
            run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 7: Dual Momentum
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("STRATEGY 7: Dual Momentum")
    for lb in [126, 252]:
        for lev in [1.0, 1.3, 1.5]:
            label = f"S7_DualMom(lb={lb},lev={lev})"
            w = strat7_dual_momentum(prices, lookback=lb, leverage=lev)
            run_and_log(label, w)

    # ═══════════════════════════════════════════════════════════════════
    # ENSEMBLES: combine best variants from each family
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("ENSEMBLES")

    # Find best variant per family by Sharpe
    families = {
        "S1": [k for k in results if k.startswith("S1_")],
        "S2": [k for k in results if k.startswith("S2_")],
        "S3": [k for k in results if k.startswith("S3_")],
        "S4": [k for k in results if k.startswith("S4_")],
        "S5": [k for k in results if k.startswith("S5_")],
        "S6": [k for k in results if k.startswith("S6_")],
        "S7": [k for k in results if k.startswith("S7_")],
    }

    best_per_family = {}
    for fam, keys in families.items():
        if keys:
            best_key = max(keys, key=lambda k: results[k]["metrics"]["Sharpe Ratio"])
            best_per_family[fam] = best_key
            m = results[best_key]["metrics"]
            print(f"  Best {fam}: {best_key} → Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"CAGR={m['CAGR']:.2%}")

    # Compute pairwise correlations between best variants
    ret_best = pd.DataFrame({fam: results[k]["portfolio_returns"]
                             for fam, k in best_per_family.items()})
    print(f"\n  Correlations (best per family):")
    corr = ret_best.corr()
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))
    print()

    # Build ensembles
    def build_ensemble(label, components, weights_list):
        """components: list of result keys, weights_list: list of floats."""
        ens_ret = sum(w_i * results[k]["portfolio_returns"]
                      for k, w_i in zip(components, weights_list))
        ens_eq = 100_000 * (1 + ens_ret).cumprod()
        ens_eq.name = "Equity"
        turn = sum(w_i * results[k]["turnover"]
                   for k, w_i in zip(components, weights_list))
        g_exp = sum(w_i * results[k]["gross_exposure"]
                    for k, w_i in zip(components, weights_list))
        net_exp = sum(w_i * results[k]["net_exposure"]
                      for k, w_i in zip(components, weights_list))
        cash_w = (1 - net_exp).clip(lower=0)

        m = compute_metrics(ens_ret, ens_eq, 100_000, risk_free_rate=RF,
                            turnover=turn, gross_exposure=g_exp)

        cagr_ok = "Y" if m["CAGR"] > spy_cagr else " "
        sh_ok   = "Y" if m["Sharpe Ratio"] > 1.95 else " "
        avg_lev = g_exp.mean()
        cash_pct = cash_w.mean()
        print(f"  {label:52s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} Lev={avg_lev:.2f} Cash={cash_pct:.0%}")

        results[label] = {
            "equity_curve": ens_eq, "portfolio_returns": ens_ret,
            "weights": pd.DataFrame(), "turnover": turn,
            "gross_exposure": g_exp, "net_exposure": net_exp,
            "cash_weight": cash_w, "metrics": m,
        }

    # Various ensemble combos
    bpf = best_per_family
    if len(bpf) >= 2:
        combos = []

        # Directional + Neutral
        if "S1" in bpf and "S3" in bpf:
            combos.append(("Ens_S1+S3",       [bpf["S1"], bpf["S3"]], [0.7, 0.3]))
        if "S5" in bpf and "S3" in bpf:
            combos.append(("Ens_S5+S3",       [bpf["S5"], bpf["S3"]], [0.7, 0.3]))

        # Timing combos
        if "S1" in bpf and "S5" in bpf:
            combos.append(("Ens_S1+S5",       [bpf["S1"], bpf["S5"]], [0.5, 0.5]))
        if "S1" in bpf and "S7" in bpf:
            combos.append(("Ens_S1+S7",       [bpf["S1"], bpf["S7"]], [0.5, 0.5]))

        # 3-strategy combos
        if all(f in bpf for f in ["S1", "S3", "S5"]):
            combos.append(("Ens_S1+S3+S5",    [bpf["S1"], bpf["S3"], bpf["S5"]],
                           [0.45, 0.15, 0.40]))
        if all(f in bpf for f in ["S1", "S3", "S7"]):
            combos.append(("Ens_S1+S3+S7",    [bpf["S1"], bpf["S3"], bpf["S7"]],
                           [0.40, 0.15, 0.45]))

        # 4+ strategy combos
        if all(f in bpf for f in ["S1", "S2", "S3", "S5"]):
            combos.append(("Ens_S1+S2+S3+S5", [bpf["S1"], bpf["S2"], bpf["S3"], bpf["S5"]],
                           [0.35, 0.15, 0.10, 0.40]))

        # All families
        if len(bpf) >= 5:
            all_keys = list(bpf.values())
            eq_w = [1.0 / len(all_keys)] * len(all_keys)
            combos.append(("Ens_ALL_Equal", all_keys, eq_w))

        for label, comp, wgts in combos:
            build_ensemble(label, comp, wgts)

    # ═══════════════════════════════════════════════════════════════════
    # LEVERAGED ENSEMBLES
    # ═══════════════════════════════════════════════════════════════════
    print(SEP)
    print("LEVERAGED ENSEMBLES (scale up best ensembles)")
    ens_keys = [k for k in results if k.startswith("Ens_")]
    # Also include best individual strategies
    top_indiv = sorted(
        [(k, v) for k, v in results.items() if not k.startswith("Ens_")],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:5]
    lev_bases = ens_keys + [k for k, _ in top_indiv]

    for base_key in lev_bases:
        base_ret = results[base_key]["portfolio_returns"]
        base_sharpe = results[base_key]["metrics"]["Sharpe Ratio"]
        if base_sharpe < 0.5:
            continue  # skip poor base strategies
        for mult in [1.3, 1.5, 2.0]:
            label = f"LEV_{base_key}_x{mult}"
            if len(label) > 55:
                label = label[:55]
            scaled_ret = base_ret * mult
            extra_cost = (mult - 1) * LEV_COST / 252
            scaled_ret = scaled_ret - extra_cost
            eq = 100_000 * (1 + scaled_ret).cumprod()
            eq.name = "Equity"
            m = compute_metrics(scaled_ret, eq, 100_000, risk_free_rate=RF)
            cagr_ok = "Y" if m["CAGR"] > spy_cagr else " "
            sh_ok   = "Y" if m["Sharpe Ratio"] > 1.95 else " "
            print(f"  {label:52s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
                  f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
                  f"MaxDD={m['Max Drawdown']:7.2%}")
            results[label] = {
                "equity_curve": eq, "portfolio_returns": scaled_ret,
                "weights": pd.DataFrame(), "metrics": m,
                "turnover": pd.Series(0, index=prices.index),
                "gross_exposure": pd.Series(mult, index=prices.index),
                "net_exposure": pd.Series(mult, index=prices.index),
                "cash_weight": pd.Series(0, index=prices.index),
            }

    # ═══════════════════════════════════════════════════════════════════
    # AUDIT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    auditable = {k: v for k, v in results.items()
                 if not v["weights"].empty if isinstance(v["weights"], pd.DataFrame)}
    iss = audit(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies checked)")
    print(f"  Issues: {len(iss)}")
    if iss:
        for i in iss:
            print(f"  !! {i}")
    else:
        print("  ALL AUDITS PASS")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("FINAL RESULTS (sorted by Sharpe Ratio)\n")
    sorted_res = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    print(f"{'Strategy':<56s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Target?':>8s}")
    print("-" * 86)
    print(f"{'SPY (benchmark)':<56s} {spy_cagr:7.2%} {spy_sharpe:7.4f} "
          f"{benchmarks['SPY']['Max Drawdown']:7.2%}")
    qqq_m = benchmarks.get("QQQ", {})
    if qqq_m:
        print(f"{'QQQ (benchmark)':<56s} {qqq_m['CAGR']:7.2%} "
              f"{qqq_m['Sharpe Ratio']:7.4f} {qqq_m['Max Drawdown']:7.2%}")
    print("-" * 86)

    winners = []
    for label, res in sorted_res:
        m = res["metrics"]
        cagr = m["CAGR"]
        sharpe = m["Sharpe Ratio"]
        dd = m["Max Drawdown"]
        hit = ""
        if cagr > spy_cagr and sharpe > 1.95:
            hit = "  WINNER"
            winners.append(label)
        elif cagr > spy_cagr:
            hit = "  CAGR+"
        elif sharpe > 1.95:
            hit = "  SH+"
        print(f"{label:<56s} {cagr:7.2%} {sharpe:7.4f} {dd:7.2%}{hit}")

    print("\n" + "=" * 90)
    if winners:
        print(f"WINNERS ({len(winners)} strategies meeting BOTH targets):")
        for label in winners:
            m = results[label]["metrics"]
            print(f"  {label}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}")
    else:
        print("No full-period winners yet. Closest to BOTH targets:")
        # Show strategies that beat SPY CAGR, sorted by Sharpe
        cagr_beaters = [(k, v) for k, v in sorted_res
                        if v["metrics"]["CAGR"] > spy_cagr]
        if cagr_beaters:
            for label, res in cagr_beaters[:8]:
                m = res["metrics"]
                print(f"  {label}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")
        else:
            print("  (No strategies beat SPY CAGR either)")
            # Show top by Sharpe anyway
            for label, res in sorted_res[:8]:
                m = res["metrics"]
                print(f"  {label}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")

    print(f"\nAudit: {len(iss)} issues | Strategies tested: {len(results)}")


if __name__ == "__main__":
    main()
