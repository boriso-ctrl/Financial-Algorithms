"""
Leveraged Alpha v3 — Conditional Leverage + Dynamic Hedging
============================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95
Period: 2010-2025

Key insight from v2: F_HedgedLev achieved Sharpe=1.10 (best so far).
To reach 1.95, I need to cut portfolio vol by ~40% while maintaining/increasing returns.

v3 Design:
  - 4-state tactical allocation (trend × vol regime)
  - Conditional leverage: 2x in optimal, 0x in crisis
  - Dynamic hedging: actively manage GLD/IEF size
  - Multi-condition scoring overlay
  - Very aggressive crash avoidance
  - Drawdown-based circuit breaker

The math: need ~15% CAGR at ~7-8% annual vol to get Sharpe ~1.95
  → This means being OUT of equity during all high-vol periods
  → And LEVERAGED during calm uptrends to compensate
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
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD"]
START = "2010-01-01"
END = "2025-03-01"
TX_BPS = 5
LEV_COST = 0.015
SHORT_COST = 0.005
RF = 0.0


def load_data():
    raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return p.dropna(how="all").ffill().bfill()


def run_bt(prices, weights, rebal=5, cap=100_000.0):
    """Vectorized backtest with leverage costs. Shift-1 applied here."""
    common = prices.columns.intersection(weights.columns)
    p, w = prices[common], weights[common].reindex(prices.index).fillna(0)
    mask = pd.Series(False, index=w.index)
    mask.iloc[::rebal] = True
    w = w.where(mask).ffill().fillna(0)
    w = w.shift(1).fillna(0)  # NO look-ahead
    ret = p.pct_change().fillna(0)
    gross_ret = (w * ret).sum(axis=1)
    turn = w.diff().fillna(0).abs().sum(axis=1)
    tx = turn * TX_BPS / 10_000
    g_exp = w.abs().sum(axis=1)
    lc = (g_exp - 1).clip(lower=0) * LEV_COST / 252
    sc = w.clip(upper=0).abs().sum(axis=1) * SHORT_COST / 252
    net = gross_ret - tx - lc - sc
    eq = cap * (1 + net).cumprod()
    eq.name = "Equity"
    m = compute_metrics(net, eq, cap, risk_free_rate=RF, turnover=turn, gross_exposure=g_exp)
    return {"equity_curve": eq, "portfolio_returns": net, "weights": w,
            "turnover": turn, "gross_exposure": g_exp, "metrics": m}


# ── Signal primitives ──
def sma(s, n): return s.rolling(n, min_periods=n).mean()
def ema(s, n): return s.ewm(span=n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=10).std() * np.sqrt(252)

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.clip(lower=1e-10))

def drawdown_pct(equity: pd.Series) -> pd.Series:
    """Rolling drawdown from peak."""
    peak = equity.cummax()
    return (equity - peak) / peak


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: 4-State Tactical Allocation
#   States based on (trend, vol):
#     OPTIMAL  : uptrend + low vol   → 2x leveraged equity
#     NORMAL   : uptrend + high vol  → 0.5x equity + hedge
#     DEFENSIVE: downtrend + low vol → 0x equity + moderate hedge
#     CRISIS   : downtrend + high vol→ 0x equity + max hedge
# ═══════════════════════════════════════════════════════════════════════════

def strategy_4state(prices, eq_lev=2.0, vol_threshold_pctile=60,
                    hedge_crisis=0.5, hedge_def=0.4):
    spy = prices["SPY"]
    ma200 = sma(spy, 200)
    ma50 = sma(spy, 50)

    # Trend: uptrend if SPY > MA200 AND MA50 > MA200
    uptrend = ((spy > ma200) & (ma50 > ma200)).astype(float)

    # Vol regime: low vol if current vol < percentile of trailing vol
    vol = rvol(spy, 21)
    vol_med = vol.rolling(252, min_periods=63).quantile(vol_threshold_pctile / 100.0)
    low_vol = (vol < vol_med).astype(float)

    # 4 states
    optimal = uptrend * low_vol
    normal = uptrend * (1 - low_vol)
    defensive = (1 - uptrend) * low_vol
    crisis = (1 - uptrend) * (1 - low_vol)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    eq_cols = [c for c in ["SPY", "QQQ"] if c in prices.columns]
    per_eq = eq_lev / max(len(eq_cols), 1)

    for c in eq_cols:
        w[c] = optimal * per_eq + normal * per_eq * 0.3

    if "GLD" in w.columns:
        w["GLD"] = (crisis * hedge_crisis * 0.6 + defensive * hedge_def * 0.6
                    + normal * 0.1 + optimal * 0.05)
    if "IEF" in w.columns:
        w["IEF"] = (crisis * hedge_crisis * 0.4 + defensive * hedge_def * 0.4
                    + normal * 0.1 + optimal * 0.05)
    if "TLT" in w.columns:
        w["TLT"] = crisis * 0.1 + defensive * 0.1

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Multi-Condition Scoring with Conditional Leverage
#   Each condition scores +1. Deploy capital proportional to score.
#   Conditions: trend200, trend50, momentum12m, low_vol, rsi_healthy,
#               price_above_20dma
#   Score 5-6: 2x leverage
#   Score 3-4: 1x
#   Score 1-2: 0.3x + hedge
#   Score 0:   0x, max hedge
# ═══════════════════════════════════════════════════════════════════════════

def strategy_scoring(prices, max_lev=2.0):
    spy = prices["SPY"]

    # Condition scores (each 0 or 1)
    c1 = (spy > sma(spy, 200)).astype(float)       # long-term trend
    c2 = (sma(spy, 50) > sma(spy, 200)).astype(float)  # golden cross
    c3 = (spy.pct_change(252) > 0).astype(float)     # 12m momentum positive
    vol = rvol(spy, 21)
    vol_avg = vol.rolling(126, min_periods=30).mean()
    c4 = (vol < vol_avg * 1.1).astype(float)         # vol not elevated
    r = rsi(spy, 14)
    c5 = ((r > 30) & (r < 75)).astype(float)         # RSI not extreme
    c6 = (spy > sma(spy, 20)).astype(float)           # short-term healthy

    score = c1 + c2 + c3 + c4 + c5 + c6  # 0-6

    # Leverage mapping
    lev = pd.Series(0.0, index=prices.index)
    lev[score >= 5] = max_lev
    lev[(score >= 3) & (score < 5)] = max_lev * 0.5
    lev[(score >= 1) & (score < 3)] = max_lev * 0.15
    # score == 0 → 0 leverage

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    eq_cols = [c for c in ["SPY", "QQQ"] if c in prices.columns]
    per_eq = 1.0 / max(len(eq_cols), 1)
    for c in eq_cols:
        w[c] = lev * per_eq

    # Hedge inversely proportional to score
    hedge_size = ((6 - score) / 6).clip(lower=0.05, upper=0.6)
    if "GLD" in w.columns:
        w["GLD"] = hedge_size * 0.6
    if "IEF" in w.columns:
        w["IEF"] = hedge_size * 0.4

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Enhanced Hedged Leveraged (build on v2's F_HedgedLev)
#   Improvements:
#     - Vol overlay: scale equity by inverse vol
#     - Dynamic hedge: increase in high-vol
#     - RSI boost: overweight on oversold dips in uptrend
#     - Drawdown circuit breaker: reduce to 0 if DD > threshold
# ═══════════════════════════════════════════════════════════════════════════

def strategy_enhanced_hedged(prices, base_lev=1.5, hedge_pct=0.25,
                             dd_halt=-0.12, vol_target=0.12):
    spy = prices["SPY"]
    qqq = prices.get("QQQ", spy)
    ma200 = sma(spy, 200)
    ma50 = sma(spy, 50)

    # Trend
    uptrend = (spy > ma200).astype(float)
    strong = uptrend * (ma50 > ma200).astype(float)

    # Vol-managed sizing
    port_ret = 0.55 * spy.pct_change() + 0.45 * qqq.pct_change()
    pvol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_scale = (vol_target / pvol.clip(lower=0.03)).clip(upper=2.5)
    vol_scale = vol_scale.rolling(5, min_periods=1).mean()

    # RSI boost
    r = rsi(spy, 14)
    rsi_mult = np.where((r < 35) & (strong > 0), 1.3,
               np.where(r > 78, 0.6, 1.0))

    # Equity weights
    eq_base = base_lev * (1 - hedge_pct)
    eq_weight = eq_base * strong.clip(lower=0.15) * vol_scale * rsi_mult

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in w.columns:
        w["SPY"] = eq_weight * 0.55
    if "QQQ" in w.columns:
        w["QQQ"] = eq_weight * 0.45

    # Dynamic hedge
    vol_ratio = (pvol / pvol.rolling(126, min_periods=30).mean()).fillna(1)
    hedge_mult = vol_ratio.clip(lower=0.5, upper=2.5)
    base_hedge = base_lev * hedge_pct
    if "GLD" in w.columns:
        w["GLD"] = base_hedge * 0.5 * hedge_mult
    if "IEF" in w.columns:
        w["IEF"] = base_hedge * 0.5 * hedge_mult

    # Drawdown circuit breaker
    # Compute rolling equity proxy to detect drawdowns
    total_w = w.abs().sum(axis=1)
    port_proxy = (1 + (w * prices.pct_change().fillna(0)).sum(axis=1)).cumprod()
    dd = drawdown_pct(port_proxy)
    in_dd = (dd < dd_halt).astype(float)
    # When in drawdown, reduce everything to 20%
    dd_scale = 1.0 - in_dd * 0.8
    w = w.mul(dd_scale, axis=0)

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Crisis Alpha Rotation
#   When equity is good: leveraged equity
#   When equity is bad: actively long GLD + short-duration bonds
#   Key: GLD and bonds often APPRECIATE during equity crashes
#        → This turns defensive periods into POSITIVE return periods
# ═══════════════════════════════════════════════════════════════════════════

def strategy_crisis_alpha(prices, eq_lev=1.8, alt_lev=0.8):
    spy = prices["SPY"]
    ma200 = sma(spy, 200)
    ma50 = sma(spy, 50)

    # States
    bull = ((spy > ma200) & (ma50 > ma200)).astype(float)
    pullback = ((spy > ma200) & (ma50 <= ma200)).astype(float)  # weakening
    bear = (1 - bull - pullback).clip(lower=0)

    # Vol regime
    vol = rvol(spy, 21)
    vol_high = (vol > vol.rolling(126, min_periods=30).quantile(0.7)).astype(float)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    eq_cols = [c for c in ["SPY", "QQQ"] if c in prices.columns]

    # Bull: leveraged equity, small hedge
    for c in eq_cols:
        per = eq_lev / max(len(eq_cols), 1)
        w[c] = bull * per * (1 - vol_high * 0.4)  # reduce in high vol
        w[c] += pullback * per * 0.3  # small position during pullback

    # Bear / crisis: actively long alternatives
    if "GLD" in w.columns:
        # GLD tends to rise during equity crashes
        w["GLD"] = bear * alt_lev * 0.5 + pullback * 0.3
        w["GLD"] += vol_high * bear * 0.2  # extra in high vol crisis
    if "TLT" in w.columns:
        # TLT for rate-cut expectations during crises
        w["TLT"] = bear * alt_lev * 0.3 + pullback * 0.2
    if "IEF" in w.columns:
        w["IEF"] = bear * alt_lev * 0.2 + pullback * 0.1
        # Always hold small IEF position
        w["IEF"] += 0.05

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: Aggressive Conditional + Vol Filter (kitchen sink approach)
#   Combines ALL insights:
#   - Multi-condition score for leverage sizing
#   - Vol-managed within each state
#   - Dynamic hedge proportional to risk
#   - Drawdown circuit breaker
#   - RSI timing for entries
# ═══════════════════════════════════════════════════════════════════════════

def strategy_kitchen_sink(prices, max_lev=2.0, vol_target=0.10,
                          dd_halt=-0.10):
    spy = prices["SPY"]
    qqq = prices.get("QQQ", spy)

    # ── Condition Scoring ──
    c_trend200 = (spy > sma(spy, 200)).astype(float)
    c_golden = (sma(spy, 50) > sma(spy, 200)).astype(float)
    c_mom12 = (spy.pct_change(252) > 0).astype(float)
    c_mom3 = (spy.pct_change(63) > 0).astype(float)
    vol = rvol(spy, 21)
    vol_avg = vol.rolling(126, min_periods=30).mean()
    c_lowvol = (vol < vol_avg * 1.0).astype(float)
    r = rsi(spy, 14)
    c_rsi_healthy = ((r > 25) & (r < 75)).astype(float)
    c_above_20ma = (spy > sma(spy, 20)).astype(float)
    c_above_50ma = (spy > sma(spy, 50)).astype(float)

    score = (c_trend200 + c_golden + c_mom12 + c_mom3 +
             c_lowvol + c_rsi_healthy + c_above_20ma + c_above_50ma)  # 0-8

    # ── Leverage from score ──
    lev = pd.Series(0.0, index=prices.index)
    lev[score >= 7] = max_lev
    lev[(score >= 5) & (score < 7)] = max_lev * 0.6
    lev[(score >= 3) & (score < 5)] = max_lev * 0.25
    lev[(score >= 1) & (score < 3)] = 0.0
    # score == 0 → 0

    # ── Vol-managed adjustment ──
    port_ret = 0.55 * spy.pct_change() + 0.45 * qqq.pct_change()
    pvol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_adj = (vol_target / pvol.clip(lower=0.03)).clip(upper=2.0)
    vol_adj = vol_adj.rolling(5, min_periods=1).mean()

    # Combined equity sizing
    eq_size = lev * vol_adj

    # Cap at max leverage
    eq_size = eq_size.clip(upper=max_lev)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in w.columns:
        w["SPY"] = eq_size * 0.55
    if "QQQ" in w.columns:
        w["QQQ"] = eq_size * 0.45

    # ── Dynamic hedge (inverse of score) ──
    hedge_intensity = ((8 - score) / 8).clip(lower=0.05)
    if "GLD" in w.columns:
        w["GLD"] = hedge_intensity * 0.35
        # Boost GLD in crisis
        crisis = (score <= 2).astype(float)
        w["GLD"] += crisis * 0.2
    if "IEF" in w.columns:
        w["IEF"] = hedge_intensity * 0.20
    if "TLT" in w.columns:
        w["TLT"] = (score <= 2).astype(float) * 0.15

    # ── Drawdown circuit breaker ──
    port_proxy = (1 + (w * prices.pct_change().fillna(0)).sum(axis=1)).cumprod()
    dd = drawdown_pct(port_proxy)
    in_dd = (dd < dd_halt).astype(float)
    dd_scale = 1.0 - in_dd * 0.85  # reduce to 15% of normal
    w = w.mul(dd_scale, axis=0)

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 6: Ultra-Selective Leveraged Deployment
#   Only deploy capital when ALL conditions are perfect.
#   Use very high leverage (2.5-3x) during these periods.
#   Rest of time: 100% in hedges.
#   Targets very low vol (by being in market <50% of the time) with
#   high returns (via leverage during good periods).
# ═══════════════════════════════════════════════════════════════════════════

def strategy_ultra_selective(prices, deploy_lev=2.5, min_score=6):
    spy = prices["SPY"]
    qqq = prices.get("QQQ", spy)

    # Strict conditions
    c1 = (spy > sma(spy, 200)).astype(float)
    c2 = (sma(spy, 50) > sma(spy, 200)).astype(float)
    c3 = (spy > sma(spy, 50)).astype(float)
    c4 = (spy > sma(spy, 20)).astype(float)
    vol = rvol(spy, 21)
    c5 = (vol < vol.rolling(126, min_periods=30).quantile(0.5)).astype(float)
    r = rsi(spy, 14)
    c6 = ((r > 30) & (r < 70)).astype(float)
    c7 = (spy.pct_change(63) > 0).astype(float)
    c8 = (spy.pct_change(21) > -0.03).astype(float)  # not crashed recently

    score = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

    deploy = (score >= min_score).astype(float)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    eq_cols = [c for c in ["SPY", "QQQ"] if c in prices.columns]
    per = deploy_lev / max(len(eq_cols), 1)
    for c in eq_cols:
        w[c] = deploy * per

    # When NOT deployed: hold defensive
    not_deployed = 1 - deploy
    if "GLD" in w.columns:
        w["GLD"] = not_deployed * 0.35
    if "IEF" in w.columns:
        w["IEF"] = not_deployed * 0.35
    if "TLT" in w.columns:
        w["TLT"] = not_deployed * 0.15

    return w


# ═══════════════════════════════════════════════════════════════════════════
# META-ENSEMBLE: Blend of uncorrelated strategies
# ═══════════════════════════════════════════════════════════════════════════

def strategy_meta_ensemble(prices, strats_and_weights=None):
    if strats_and_weights is None:
        strats_and_weights = [
            (strategy_4state, {"eq_lev": 2.0, "vol_threshold_pctile": 55}, 0.25),
            (strategy_scoring, {"max_lev": 2.0}, 0.25),
            (strategy_enhanced_hedged, {"base_lev": 1.5}, 0.25),
            (strategy_crisis_alpha, {"eq_lev": 1.8}, 0.25),
        ]
    parts = []
    for fn, kwargs, w in strats_and_weights:
        parts.append((fn(prices, **kwargs), w))

    # Align columns
    all_cols = set()
    for wdf, _ in parts:
        all_cols.update(wdf.columns)
    common = sorted(all_cols.intersection(prices.columns))

    result = pd.DataFrame(0.0, index=prices.index, columns=common)
    for wdf, weight in parts:
        for c in common:
            if c in wdf.columns:
                result[c] += wdf[c].reindex(prices.index, fill_value=0) * weight

    return result


# ── Audit Functions ──

def audit_all(name, weights, prices, result):
    issues = 0
    # Look-ahead check
    ret = prices.pct_change()
    for col in weights.columns:
        if col in ret.columns and weights[col].abs().sum() > 0:
            corr = weights[col].corr(ret[col])
            if abs(corr) > 0.25:
                print(f"    ⚠️  {name}/{col}: lookahead corr={corr:.4f}")
                issues += 1

    # Sharpe check
    r = result["portfolio_returns"]
    excess = r - RF / 252
    sc = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    if abs(sc - result["metrics"]["Sharpe Ratio"]) > 0.05:
        print(f"    ⚠️  {name}: Sharpe mismatch ({sc:.4f} vs {result['metrics']['Sharpe Ratio']:.4f})")
        issues += 1

    # CAGR check
    eq = result["equity_curve"]
    yrs = len(r) / 252
    tr = eq.iloc[-1] / 100_000 - 1
    cc = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
    if abs(cc - result["metrics"]["CAGR"]) > 0.005:
        print(f"    ⚠️  {name}: CAGR mismatch ({cc*100:.2f}% vs {result['metrics']['CAGR']*100:.2f}%)")
        issues += 1

    # Sanity
    m = result["metrics"]
    if m["CAGR"] > 0.80:
        print(f"    ⚠️  {name}: CAGR={m['CAGR']*100:.1f}% unrealistic")
        issues += 1
    if m["Sharpe Ratio"] > 4.5:
        print(f"    ⚠️  {name}: Sharpe={m['Sharpe Ratio']:.2f} unrealistic")
        issues += 1

    return issues


# ── Main ──

def main():
    print("=" * 90)
    print("LEVERAGED ALPHA v3 — CONDITIONAL LEVERAGE + DYNAMIC HEDGING")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  {START} to {END}")
    print("=" * 90)

    prices = load_data()
    print(f"\n📊 {len(prices.columns)} tickers, {len(prices)} days, "
          f"{len(prices)/252:.1f} years")

    spy_r = prices["SPY"].pct_change().dropna()
    spy_eq = 100_000 * (1 + spy_r).cumprod()
    spy_m = compute_metrics(spy_r, spy_eq, 100_000, risk_free_rate=RF)
    print(f"📈 SPY: CAGR={spy_m['CAGR']*100:.2f}%, Sharpe={spy_m['Sharpe Ratio']:.4f}, "
          f"MaxDD={spy_m['Max Drawdown']*100:.2f}%")

    # ── Strategy definitions with params to sweep ──
    strats = {
        "S1_4State": (strategy_4state, [
            {"eq_lev": 1.5, "vol_threshold_pctile": 50},
            {"eq_lev": 2.0, "vol_threshold_pctile": 50},
            {"eq_lev": 2.0, "vol_threshold_pctile": 55},
            {"eq_lev": 2.0, "vol_threshold_pctile": 60},
            {"eq_lev": 2.5, "vol_threshold_pctile": 55},
            {"eq_lev": 2.5, "vol_threshold_pctile": 60},
            {"eq_lev": 3.0, "vol_threshold_pctile": 60},
        ]),
        "S2_Scoring": (strategy_scoring, [
            {"max_lev": 1.5},
            {"max_lev": 2.0},
            {"max_lev": 2.5},
            {"max_lev": 3.0},
        ]),
        "S3_EnhancedHedge": (strategy_enhanced_hedged, [
            {"base_lev": 1.3, "hedge_pct": 0.25, "vol_target": 0.10},
            {"base_lev": 1.5, "hedge_pct": 0.25, "vol_target": 0.10},
            {"base_lev": 1.5, "hedge_pct": 0.30, "vol_target": 0.10},
            {"base_lev": 1.5, "hedge_pct": 0.25, "vol_target": 0.12},
            {"base_lev": 1.8, "hedge_pct": 0.30, "vol_target": 0.10},
            {"base_lev": 2.0, "hedge_pct": 0.30, "vol_target": 0.10},
            {"base_lev": 2.0, "hedge_pct": 0.30, "vol_target": 0.12},
        ]),
        "S4_CrisisAlpha": (strategy_crisis_alpha, [
            {"eq_lev": 1.5, "alt_lev": 0.6},
            {"eq_lev": 1.8, "alt_lev": 0.8},
            {"eq_lev": 2.0, "alt_lev": 0.8},
            {"eq_lev": 2.0, "alt_lev": 1.0},
            {"eq_lev": 2.5, "alt_lev": 1.0},
        ]),
        "S5_KitchenSink": (strategy_kitchen_sink, [
            {"max_lev": 2.0, "vol_target": 0.08, "dd_halt": -0.10},
            {"max_lev": 2.0, "vol_target": 0.10, "dd_halt": -0.10},
            {"max_lev": 2.0, "vol_target": 0.10, "dd_halt": -0.12},
            {"max_lev": 2.5, "vol_target": 0.10, "dd_halt": -0.10},
            {"max_lev": 2.5, "vol_target": 0.12, "dd_halt": -0.12},
            {"max_lev": 3.0, "vol_target": 0.10, "dd_halt": -0.10},
            {"max_lev": 3.0, "vol_target": 0.12, "dd_halt": -0.12},
        ]),
        "S6_UltraSelective": (strategy_ultra_selective, [
            {"deploy_lev": 2.0, "min_score": 6},
            {"deploy_lev": 2.5, "min_score": 6},
            {"deploy_lev": 2.5, "min_score": 7},
            {"deploy_lev": 3.0, "min_score": 6},
            {"deploy_lev": 3.0, "min_score": 7},
            {"deploy_lev": 3.5, "min_score": 7},
        ]),
    }

    all_results = {}
    best_per = {}

    for sname, (fn, params_list) in strats.items():
        print(f"\n{'─'*70}\n🔄 {sname}")
        best_sharpe = -999
        for params in params_list:
            w = fn(prices, **params)
            res = run_bt(prices, w)
            m = res["metrics"]
            label = f"{sname}({', '.join(f'{k}={v}' for k,v in params.items())})"
            all_results[label] = (w, res)
            bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
            bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
            star = " ⭐" if bc == "✓" and bs == "✓" else ""
            ps = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"   {ps:<45} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
                  f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] "
                  f"MaxDD={m['Max Drawdown']*100:>7.2f}% "
                  f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")
            if m["Sharpe Ratio"] > best_sharpe:
                best_sharpe = m["Sharpe Ratio"]
                best_per[sname] = (params, w, res)
        bp, _, br = best_per[sname]
        print(f"   ► Best: {bp} → Sharpe={br['metrics']['Sharpe Ratio']:.4f}, "
              f"CAGR={br['metrics']['CAGR']*100:.2f}%")

    # ── Ensemble ──
    print(f"\n{'─'*70}\n🔄 Ensemble strategies")
    ensemble_configs = [
        ("Ens_equal", None),
    ]
    for ens_name, custom in ensemble_configs:
        w = strategy_meta_ensemble(prices, custom)
        res = run_bt(prices, w)
        m = res["metrics"]
        all_results[ens_name] = (w, res)
        bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
        bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
        star = " ⭐" if bc == "✓" and bs == "✓" else ""
        print(f"   {ens_name:<45} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
              f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] "
              f"MaxDD={m['Max Drawdown']*100:>7.2f}% "
              f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

    # Weight optimized ensemble: overweight best strategies
    best_strats_for_ensemble = []
    for sname, (params, w_fn_result, result) in best_per.items():
        fn = strats[sname][0]
        best_strats_for_ensemble.append(
            (fn, params, result["metrics"]["Sharpe Ratio"])
        )
    # Weight by Sharpe
    total_sharpe = sum(s for _, _, s in best_strats_for_ensemble)
    weighted_ensemble = []
    for fn, params, s in best_strats_for_ensemble:
        weighted_ensemble.append((fn, params, s / total_sharpe))
    w = strategy_meta_ensemble(prices, weighted_ensemble)
    res = run_bt(prices, w)
    m = res["metrics"]
    all_results["Ens_sharpe_weighted"] = (w, res)
    bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
    bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
    star = " ⭐" if bc == "✓" and bs == "✓" else ""
    print(f"   {'Ens_sharpe_weighted':<45} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
          f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] "
          f"MaxDD={m['Max Drawdown']*100:>7.2f}% "
          f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

    # ── Audit ──
    print(f"\n{'='*90}\n🔍 AUDIT")
    total_issues = 0
    for sname, (params, w, res) in best_per.items():
        total_issues += audit_all(sname, w, prices, res)
    for ens_name in ["Ens_equal", "Ens_sharpe_weighted"]:
        if ens_name in all_results:
            ww, rr = all_results[ens_name]
            total_issues += audit_all(ens_name, ww, prices, rr)
    print(f"   Total audit issues: {total_issues}")
    if total_issues == 0:
        print("   ✅ All audits pass")

    # ── Correlation of best strategies ──
    print(f"\n{'='*90}\nStrategy correlations (best configs):")
    corr_data = {s: r["portfolio_returns"] for s, (_, _, r) in best_per.items()}
    corr = pd.DataFrame(corr_data).corr()
    print(corr.round(3).to_string())

    # ── Sub-period analysis ──
    print(f"\n{'='*90}\n📅 SUB-PERIOD ANALYSIS (best configs)")
    periods = [("2010-2025", None, None), ("2015-2025", "2015-01-01", None),
               ("2018-2025", "2018-01-01", None), ("2020-2025", "2020-01-01", None)]
    for pname, s, e in periods:
        sp = prices.copy()
        if s: sp = sp[sp.index >= s]
        if e: sp = sp[sp.index <= e]
        sr = sp["SPY"].pct_change().dropna()
        se = 100_000 * (1 + sr).cumprod()
        sm = compute_metrics(sr, se, 100_000, risk_free_rate=RF)

        print(f"\n  {pname}:")
        print(f"  SPY: CAGR={sm['CAGR']*100:.2f}%, Sharpe={sm['Sharpe Ratio']:.4f}")
        for sname, (params, _, _) in best_per.items():
            fn = strats[sname][0]
            w = fn(sp, **params)
            r = run_bt(sp, w)
            m = r["metrics"]
            flags = []
            if m["CAGR"] > sm["CAGR"]: flags.append("CAGR✓")
            if m["Sharpe Ratio"] > 1.95: flags.append("Sharpe✓")
            if len(flags) == 2: flags.append("⭐")
            print(f"  {sname:<25} CAGR={m['CAGR']*100:>7.2f}% "
                  f"Sharpe={m['Sharpe Ratio']:>7.4f} MaxDD={m['Max Drawdown']*100:>7.2f}% "
                  f"{' '.join(flags)}")

    # ── Final sorted table ──
    print(f"\n{'='*90}\n📊 FINAL RESULTS (sorted by Sharpe)")
    print(f"{'Label':<55} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'AvgLev':>6}")
    print("─" * 82)
    print(f"{'SPY':<55} {spy_m['CAGR']*100:>6.2f}% {spy_m['Sharpe Ratio']:>7.4f} "
          f"{spy_m['Max Drawdown']*100:>6.2f}% {'1.00x':>6}")
    print("─" * 82)

    sorted_r = sorted(all_results.items(),
                       key=lambda x: x[1][1]["metrics"]["Sharpe Ratio"], reverse=True)
    winners = []
    for label, (_, res) in sorted_r:
        m = res["metrics"]
        star = " ⭐" if m["CAGR"] > spy_m["CAGR"] and m["Sharpe Ratio"] > 1.95 else ""
        if star: winners.append((label, m))
        print(f"{label:<55} {m['CAGR']*100:>6.2f}% {m['Sharpe Ratio']:>7.4f} "
              f"{m['Max Drawdown']*100:>6.2f}% {m['Avg Gross Leverage']:>5.2f}x{star}")

    if winners:
        print(f"\n🏆 WINNERS:")
        for l, m in winners:
            print(f"   ⭐ {l}: CAGR={m['CAGR']*100:.2f}%, Sharpe={m['Sharpe Ratio']:.4f}")
    else:
        print(f"\n⚠️  No full-period winners yet. Closest strategies above.")

    print(f"\nAudit: {total_issues} issues")
    print("=" * 90)
    return all_results, best_per, spy_m, prices


if __name__ == "__main__":
    main()
