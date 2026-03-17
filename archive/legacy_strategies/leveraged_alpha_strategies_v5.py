"""
Leveraged Alpha v5 — Vol-Management + Drawdown Control + Breadth + Multi-Source
================================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95  |  2010-01-01 to 2025-03-01

Key innovations vs v1-v4:
  1. PROPER VOL-MANAGEMENT: scale positions by target_vol / max(realized_vol_lookbacks)
     - Academic evidence: improves Sharpe 30-50% (Moreira & Muir 2017)
  2. GRADUAL DRAWDOWN CONTROL: reduce positions proportionally in drawdowns
     - Not a binary halt (which sells at bottoms), but smooth scaling
     - Computed sequentially to avoid look-ahead
  3. BREADTH REGIME DETECTION: count sector ETFs above 50-day MA (0-10)
     - Leading indicator: breadth deteriorates BEFORE price drops
     - Adds information independent of SPY MA/momentum
  4. UNCORRELATED SUB-STRATEGIES:
     - A: Vol-managed equity scoring (equity-centric, trend/momentum)
     - B: Risk parity with momentum filter (multi-asset, low vol)
     - C: Mean-reversion overlay (counter-trend on dips)
     - D: Cross-asset rotational (relative ranking, low corr)
  5. ENSEMBLE with inverse-correlation weighting

Maximum expected Sharpe (theoretical):
  - Single best strategy: ~1.2-1.5 (from vol-management + drawdown control)
  - Ensemble of 4 uncorrelated strategies at Sharpe ~0.8-1.0 each, ρ~0.3:
    Sharpe ≈ 0.9 * sqrt(4 / (1+3*0.3)) = 0.9 * 1.45 = 1.31

Note: Sharpe > 1.95 over 15 years with daily ETF data is at the frontier of what
passive/systematic allocation can achieve. Only high-frequency or multi-asset
class hedge-fund-grade approaches reliably sustain Sharpe > 2.0 over such periods.
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
SAFE    = ["TLT", "IEF", "GLD"]
UNIVERSE = SECTORS + BROAD + SAFE
START, END = "2010-01-01", "2025-03-01"
TX_BPS     = 5
LEV_COST   = 0.015   # annual cost for leverage > 1.0x
SHORT_COST = 0.005   # annual short borrow cost
RF = 0.0
SEP = "─" * 70

# ── Data ──
def load_data():
    raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n📊 {len(p.columns)} tickers loaded: {list(p.columns)}")
    print(f"   {len(p)} days ({p.index[0].strftime('%Y-%m-%d')} to "
          f"{p.index[-1].strftime('%Y-%m-%d')}, {len(p)/252:.1f}yr)\n")
    return p


# ── Signal helpers ──
def sma(s, n):   return s.rolling(n, min_periods=n).mean()
def ema(s, n):   return s.ewm(span=n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=max(10, n//2)).std() * np.sqrt(252)

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.clip(lower=1e-10))

def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


# ── Breadth indicator (SECTOR-BASED) ──
def compute_breadth(prices, sectors=None, lookback=50):
    """Count how many sector ETFs are above their N-day MA. Returns 0-10."""
    if sectors is None:
        sectors = [s for s in SECTORS if s in prices.columns]
    breadth = pd.Series(0.0, index=prices.index)
    for s in sectors:
        if s in prices.columns:
            breadth += (prices[s] > sma(prices[s], lookback)).astype(float)
    return breadth  # 0 .. len(sectors)


# ── Vol-management ──
def vol_scale_factor(prices, target_vol=0.12, max_scale=2.5, smooth=5):
    """
    Conservative vol-scale factor: target_vol / max(vol_20, vol_60, vol_120).
    Uses SPY as the vol proxy. Smoothed to avoid whipsaw.
    """
    spy_ret = prices["SPY"].pct_change()
    v20  = spy_ret.rolling(20,  min_periods=10).std() * np.sqrt(252)
    v60  = spy_ret.rolling(60,  min_periods=20).std() * np.sqrt(252)
    v120 = spy_ret.rolling(120, min_periods=40).std() * np.sqrt(252)
    realized = pd.concat([v20, v60, v120], axis=1).max(axis=1).clip(lower=0.03)
    scale = (target_vol / realized).clip(upper=max_scale)
    return scale.rolling(smooth, min_periods=1).mean()


# ══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE with Drawdown Control
# ══════════════════════════════════════════════════════════════════════════

def run_bt(prices, weights, rebal=5, cap=100_000.0, dd_schedule=None):
    """
    Vectorized/sequential backtest. shift(1) for no look-ahead.
    If dd_schedule is provided, runs sequentially with drawdown-based scaling.
    dd_schedule: list of (threshold, scale), e.g. [(-0.05, 0.75), (-0.10, 0.50)]
    """
    common = prices.columns.intersection(weights.columns)
    p = prices[common]
    w = weights[common].reindex(prices.index).fillna(0)

    # Rebalance mask
    mask = pd.Series(False, index=w.index)
    mask.iloc[::rebal] = True
    w = w.where(mask).ffill().fillna(0)
    w = w.shift(1).fillna(0)  # NO look-ahead

    ret = p.pct_change().fillna(0)

    if dd_schedule is None or len(dd_schedule) == 0:
        # Pure vectorized (fast)
        gross_ret = (w * ret).sum(axis=1)
        turn = w.diff().fillna(0).abs().sum(axis=1)
        tx = turn * TX_BPS / 10_000
        g_exp = w.abs().sum(axis=1)
        lc = (g_exp - 1).clip(lower=0) * LEV_COST / 252
        sc = w.clip(upper=0).abs().sum(axis=1) * SHORT_COST / 252
        net = gross_ret - tx - lc - sc
        eq = cap * (1 + net).cumprod()
    else:
        # Sequential with drawdown control
        n = len(p)
        eq_arr = np.empty(n)
        eq_arr[0] = cap
        peak = cap
        net_arr = np.zeros(n)
        turn_arr = np.zeros(n)
        g_exp_arr = np.zeros(n)
        w_arr = w.values.copy()
        ret_arr = ret.values
        prev_w = np.zeros(w_arr.shape[1])

        # Sort dd_schedule descending by threshold for correct lookup
        dd_sorted = sorted(dd_schedule, key=lambda x: x[0], reverse=True)

        for i in range(1, n):
            dd = (eq_arr[i-1] - peak) / peak if peak > 0 else 0
            scale = 1.0
            for thresh, sf in dd_sorted:
                if dd < thresh:
                    scale = sf
            today_w = w_arr[i] * scale
            w_arr[i] = today_w

            daily_ret = np.dot(today_w, ret_arr[i])
            turn_val = np.abs(today_w - prev_w).sum()
            g_exp_val = np.abs(today_w).sum()

            tx = turn_val * TX_BPS / 10_000
            lc = max(g_exp_val - 1, 0) * LEV_COST / 252
            sc = np.abs(np.minimum(today_w, 0)).sum() * SHORT_COST / 252
            net_ret = daily_ret - tx - lc - sc

            eq_arr[i] = eq_arr[i-1] * (1 + net_ret)
            peak = max(peak, eq_arr[i])
            net_arr[i] = net_ret
            turn_arr[i] = turn_val
            g_exp_arr[i] = g_exp_val
            prev_w = today_w

        eq = pd.Series(eq_arr, index=p.index, name="Equity")
        net = pd.Series(net_arr, index=p.index)
        turn = pd.Series(turn_arr, index=p.index)
        g_exp = pd.Series(g_exp_arr, index=p.index)
        w = pd.DataFrame(w_arr, index=p.index, columns=common)

    eq.name = "Equity"
    if dd_schedule is None:
        turn_s = w.diff().fillna(0).abs().sum(axis=1)
        g_exp_s = w.abs().sum(axis=1)
    else:
        turn_s = turn
        g_exp_s = g_exp

    m = compute_metrics(net, eq, cap, risk_free_rate=RF,
                        turnover=turn_s, gross_exposure=g_exp_s)
    return {"equity_curve": eq, "portfolio_returns": net, "weights": w,
            "turnover": turn_s, "gross_exposure": g_exp_s, "metrics": m}


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY A: Vol-Managed Scoring + Drawdown Control
#   Build on v3's S2_Scoring but add:
#   - breadth as 7th condition
#   - vol-managed sizing
#   - gradual drawdown control (in backtest engine)
# ══════════════════════════════════════════════════════════════════════════

def strat_a_vol_scoring(prices, max_lev=1.5, target_vol=0.12):
    spy = prices["SPY"]

    # 7 scoring conditions
    c1 = (spy > sma(spy, 200)).astype(float)              # LT trend
    c2 = (sma(spy, 50) > sma(spy, 200)).astype(float)     # golden cross
    c3 = (spy.pct_change(252) > 0).astype(float)           # 12m momentum
    vol = rvol(spy, 21)
    vol_avg = vol.rolling(126, min_periods=30).mean()
    c4 = (vol < vol_avg * 1.1).astype(float)               # vol not elevated
    r = rsi(spy, 14)
    c5 = ((r > 30) & (r < 75)).astype(float)               # RSI healthy
    c6 = (spy > sma(spy, 20)).astype(float)                 # ST trend
    breadth = compute_breadth(prices, lookback=50)
    n_sectors = max(len([s for s in SECTORS if s in prices.columns]), 1)
    c7 = (breadth > n_sectors * 0.5).astype(float)          # breadth > 50%

    score = c1 + c2 + c3 + c4 + c5 + c6 + c7  # 0-7

    # Leverage mapping: smooth step function
    lev = pd.Series(0.0, index=prices.index)
    lev[score >= 6] = max_lev
    lev[(score >= 4) & (score < 6)] = max_lev * 0.55
    lev[(score >= 2) & (score < 4)] = max_lev * 0.15
    # score 0-1 → 0

    # Vol-management
    vscale = vol_scale_factor(prices, target_vol=target_vol)
    lev = lev * vscale

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in ["SPY", "QQQ"]:
        if c in w.columns:
            w[c] = lev * 0.5

    # Hedge: inverse of score
    hedge = ((7 - score) / 7).clip(lower=0.05, upper=0.5)
    if "GLD" in w.columns:
        w["GLD"] = hedge * 0.5
    if "IEF" in w.columns:
        w["IEF"] = hedge * 0.5
    return w


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY B: Risk Parity + Momentum Filter
#   Equal-risk weighting across assets with positive momentum.
#   Fundamentally different from equity-centric scoring.
# ══════════════════════════════════════════════════════════════════════════

def strat_b_risk_parity_mom(prices, target_vol=0.10, mom_lookback=126):
    assets = [c for c in prices.columns if c in BROAD + SAFE]
    ret = prices[assets].pct_change()
    vol = ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    vol = vol.clip(lower=0.01)

    # Inverse-vol weights (before momentum filter)
    inv_vol = 1.0 / vol
    raw_w = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Momentum filter: zero weight if 6m return < 0
    for c in assets:
        mom_pos = (prices[c].pct_change(mom_lookback) > 0).astype(float)
        raw_w[c] = raw_w[c] * mom_pos

    # Re-normalize surviving weights
    wsum = raw_w.sum(axis=1).clip(lower=0.01)
    raw_w = raw_w.div(wsum, axis=0)

    # Vol-target the portfolio
    port_ret = (raw_w.shift(1).fillna(0) * ret).sum(axis=1)
    pvol_20 = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    pvol_60 = port_ret.rolling(60, min_periods=20).std() * np.sqrt(252)
    pvol = pd.concat([pvol_20, pvol_60], axis=1).max(axis=1).clip(lower=0.02)
    scale = (target_vol / pvol).clip(upper=3.0).rolling(5, min_periods=1).mean()
    raw_w = raw_w.mul(scale, axis=0)

    # Align to full universe
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in assets:
        if c in w.columns:
            w[c] = raw_w[c]
    return w


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY C: Mean-Reversion Overlay
#   Buy oversold dips in uptrending assets. Short-term (3-5 day) holding.
#   Low correlation with trend-following strategies.
# ══════════════════════════════════════════════════════════════════════════

def strat_c_mean_reversion(prices, target_vol=0.10):
    spy = prices["SPY"]
    qqq = prices.get("QQQ", spy)

    # Only trade in uptrend (200-day MA filter)
    up_spy = (spy > sma(spy, 200)).astype(float)
    up_qqq = (qqq > sma(qqq, 200)).astype(float)

    # RSI-based mean-reversion signal
    rsi_spy = rsi(spy, 5)  # Short-term 5-day RSI for mean-reversion
    rsi_qqq = rsi(qqq, 5)

    # Buy when RSI < 25 in uptrend (oversold bounce)
    # Signal persists for 5 days after trigger
    sig_spy = ((rsi_spy < 25) & (up_spy > 0)).astype(float)
    sig_qqq = ((rsi_qqq < 25) & (up_qqq > 0)).astype(float)

    # Hold signal for 5 days
    sig_spy = sig_spy.rolling(5, min_periods=1).max()
    sig_qqq = sig_qqq.rolling(5, min_periods=1).max()

    # Also: buy dip if 3-day return < -3% in uptrend
    dip_spy = ((spy.pct_change(3) < -0.03) & (up_spy > 0)).astype(float)
    dip_qqq = ((qqq.pct_change(3) < -0.03) & (up_qqq > 0)).astype(float)
    dip_spy = dip_spy.rolling(5, min_periods=1).max()
    dip_qqq = dip_qqq.rolling(5, min_periods=1).max()

    # Combine
    mr_spy = (sig_spy + dip_spy).clip(upper=1)
    mr_qqq = (sig_qqq + dip_qqq).clip(upper=1)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in w.columns:
        w["SPY"] = mr_spy * 0.5
    if "QQQ" in w.columns:
        w["QQQ"] = mr_qqq * 0.5

    # Vol-management
    vscale = vol_scale_factor(prices, target_vol=target_vol)
    w = w.mul(vscale, axis=0)
    return w


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY D: Cross-Asset Rotational
#   Rank assets by 3m momentum. Long top-3, avoid bottom-3.
#   Different alpha source than single-asset timing.
# ══════════════════════════════════════════════════════════════════════════

def strat_d_cross_rotation(prices, n_long=3, target_vol=0.12, mom_period=63):
    assets = [c for c in prices.columns if c in BROAD + SAFE]
    mom = prices[assets].pct_change(mom_period)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        row = mom.iloc[i].dropna()
        if len(row) < n_long + 1:
            continue
        ranked = row.sort_values(ascending=False)
        top = ranked.head(n_long).index.tolist()
        for c in top:
            if ranked[c] > 0:  # Only long positive momentum
                w.loc[prices.index[i], c] = 1.0 / n_long

    # Vol-target
    vscale = vol_scale_factor(prices, target_vol=target_vol)
    w = w.mul(vscale, axis=0)
    return w


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY E: Breadth-Based Tactical
#   Pure breadth signal (sector-based) → equity/hedge allocation.
#   Partially independent from SPY-based signals.
# ══════════════════════════════════════════════════════════════════════════

def strat_e_breadth_tactical(prices, max_lev=1.5, target_vol=0.12):
    breadth = compute_breadth(prices, lookback=50)
    n_sectors = max(len([s for s in SECTORS if s in prices.columns]), 1)
    br_pct = breadth / n_sectors  # 0.0 to 1.0

    # Also use breadth momentum (is breadth improving or deteriorating?)
    br_momentum = breadth - breadth.shift(10)  # 10-day change in breadth

    # Equity allocation: proportional to breadth
    eq_lev = pd.Series(0.0, index=prices.index)
    eq_lev[br_pct >= 0.8] = max_lev            # strong breadth: full leverage
    eq_lev[(br_pct >= 0.5) & (br_pct < 0.8)] = max_lev * 0.6  # moderate
    eq_lev[(br_pct >= 0.3) & (br_pct < 0.5)] = max_lev * 0.2  # weak
    # br_pct < 0.3 → 0 equity

    # Breadth momentum overlay: boost/reduce based on direction
    br_mom_adj = pd.Series(1.0, index=prices.index)
    br_mom_adj[br_momentum > 2]  = 1.15   # breadth improving
    br_mom_adj[br_momentum < -2] = 0.70   # breadth deteriorating
    eq_lev = eq_lev * br_mom_adj

    # Vol-management
    vscale = vol_scale_factor(prices, target_vol=target_vol)
    eq_lev = eq_lev * vscale

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in ["SPY", "QQQ"]:
        if c in w.columns:
            w[c] = eq_lev * 0.5

    # Hedge inversely proportional to breadth
    hedge = ((1 - br_pct) * 0.4).clip(lower=0.05, upper=0.5)
    if "GLD" in w.columns:
        w["GLD"] = hedge * 0.5
    if "IEF" in w.columns:
        w["IEF"] = hedge * 0.5
    return w


# ══════════════════════════════════════════════════════════════════════════
# AUDIT
# ══════════════════════════════════════════════════════════════════════════

def audit(results_dict, prices):
    issues = []
    spy_ret = prices["SPY"].pct_change()
    for label, res in results_dict.items():
        eq = res["equity_curve"]
        w = res["weights"]
        pr = res["portfolio_returns"]

        # 1. Look-ahead: weights must not use today's return
        for col in w.columns:
            corr = w[col].corr(spy_ret)
            if abs(corr) > 0.30:
                issues.append(f"[{label}] weight {col} corr with same-day SPY ret = {corr:.3f}")

        # 2. Sharpe sanity
        ann_ret = pr.mean() * 252
        ann_std = pr.std() * np.sqrt(252)
        sharpe_manual = ann_ret / ann_std if ann_std > 1e-8 else 0
        sharpe_reported = res["metrics"].get("Sharpe Ratio", 0)
        if abs(sharpe_manual - sharpe_reported) > 0.15:
            issues.append(f"[{label}] Sharpe mismatch: manual={sharpe_manual:.4f} vs reported={sharpe_reported:.4f}")

        # 3. CAGR sanity
        yrs = len(eq) / 252
        cagr_manual = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
        cagr_reported = res["metrics"].get("CAGR", 0)
        if abs(cagr_manual - cagr_reported) > 0.015:
            issues.append(f"[{label}] CAGR mismatch: manual={cagr_manual:.4f} vs reported={cagr_reported:.4f}")

        # 4. Equity curve consistency
        reconstructed = 100_000 * (1 + pr).cumprod()
        final_diff = abs(reconstructed.iloc[-1] - eq.iloc[-1]) / eq.iloc[-1]
        if final_diff > 0.02:
            issues.append(f"[{label}] equity curve inconsistency: {final_diff:.4f}")

    return issues


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("LEVERAGED ALPHA v5 — VOL-MANAGED + DRAWDOWN CONTROL + BREADTH + MULTI-SOURCE")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  {START} to {END}")
    print("=" * 80)

    prices = load_data()
    spy_eq = prices["SPY"] / prices["SPY"].iloc[0] * 100_000
    spy_ret = prices["SPY"].pct_change().fillna(0)
    spy_m = compute_metrics(spy_ret, spy_eq, 100_000, risk_free_rate=RF)
    spy_cagr = spy_m.get("CAGR", 0)
    spy_sharpe = spy_m.get("Sharpe Ratio", 0)
    spy_dd = spy_m.get("Max Drawdown", 0)
    print(f"📈 SPY: CAGR={spy_cagr:.2%}, Sharpe={spy_sharpe:.4f}, MaxDD={spy_dd:.2%}\n")

    results = {}
    all_eq = {"SPY": spy_eq}

    # ── Drawdown schedules ──
    dd_mild    = [(-0.05, 0.80), (-0.10, 0.55), (-0.15, 0.30), (-0.20, 0.15)]
    dd_moderate = [(-0.04, 0.75), (-0.08, 0.50), (-0.12, 0.30), (-0.18, 0.10)]
    dd_aggressive = [(-0.03, 0.70), (-0.06, 0.45), (-0.10, 0.25), (-0.15, 0.10)]

    # ══════════════════════════════════════════════════════════════════════
    # Strategy A: Vol-Managed Scoring
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 Strategy A: Vol-Managed Scoring (with drawdown control)")
    configs_a = [
        ("A_NoDD(lev=1.5,vol=12%)", dict(max_lev=1.5, target_vol=0.12), None),
        ("A_MildDD(lev=1.5,vol=12%)", dict(max_lev=1.5, target_vol=0.12), dd_mild),
        ("A_ModDD(lev=1.5,vol=12%)", dict(max_lev=1.5, target_vol=0.12), dd_moderate),
        ("A_AggDD(lev=1.5,vol=12%)", dict(max_lev=1.5, target_vol=0.12), dd_aggressive),
        ("A_ModDD(lev=1.8,vol=12%)", dict(max_lev=1.8, target_vol=0.12), dd_moderate),
        ("A_ModDD(lev=1.5,vol=10%)", dict(max_lev=1.5, target_vol=0.10), dd_moderate),
        ("A_ModDD(lev=2.0,vol=15%)", dict(max_lev=2.0, target_vol=0.15), dd_moderate),
    ]
    for label, kwargs, dd in configs_a:
        w = strat_a_vol_scoring(prices, **kwargs)
        res = run_bt(prices, w, rebal=5, dd_schedule=dd)
        m = res["metrics"]
        cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
        sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} AvgLev={avg_lev:.2f}x")
        results[label] = res
        all_eq[label] = res["equity_curve"]

    # ══════════════════════════════════════════════════════════════════════
    # Strategy B: Risk Parity + Momentum
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 Strategy B: Risk Parity + Momentum Filter")
    configs_b = [
        ("B_RP(vol=10%)", dict(target_vol=0.10, mom_lookback=126), None),
        ("B_RP(vol=12%)", dict(target_vol=0.12, mom_lookback=126), None),
        ("B_RP(vol=10%)+ModDD", dict(target_vol=0.10, mom_lookback=126), dd_moderate),
        ("B_RP(vol=12%)+ModDD", dict(target_vol=0.12, mom_lookback=126), dd_moderate),
    ]
    for label, kwargs, dd in configs_b:
        w = strat_b_risk_parity_mom(prices, **kwargs)
        res = run_bt(prices, w, rebal=5, dd_schedule=dd)
        m = res["metrics"]
        cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
        sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} AvgLev={avg_lev:.2f}x")
        results[label] = res
        all_eq[label] = res["equity_curve"]

    # ══════════════════════════════════════════════════════════════════════
    # Strategy C: Mean-Reversion Overlay
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 Strategy C: Mean-Reversion Overlay")
    configs_c = [
        ("C_MR(vol=10%)", dict(target_vol=0.10), None),
        ("C_MR(vol=12%)", dict(target_vol=0.12), None),
        ("C_MR(vol=10%)+ModDD", dict(target_vol=0.10), dd_moderate),
    ]
    for label, kwargs, dd in configs_c:
        w = strat_c_mean_reversion(prices, **kwargs)
        res = run_bt(prices, w, rebal=5, dd_schedule=dd)
        m = res["metrics"]
        cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
        sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} AvgLev={avg_lev:.2f}x")
        results[label] = res
        all_eq[label] = res["equity_curve"]

    # ══════════════════════════════════════════════════════════════════════
    # Strategy D: Cross-Asset Rotational
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 Strategy D: Cross-Asset Rotational")
    configs_d = [
        ("D_Rot(n=3,vol=12%)", dict(n_long=3, target_vol=0.12), None),
        ("D_Rot(n=3,vol=12%)+ModDD", dict(n_long=3, target_vol=0.12), dd_moderate),
        ("D_Rot(n=2,vol=10%)+ModDD", dict(n_long=2, target_vol=0.10), dd_moderate),
    ]
    for label, kwargs, dd in configs_d:
        w = strat_d_cross_rotation(prices, **kwargs)
        res = run_bt(prices, w, rebal=5, dd_schedule=dd)
        m = res["metrics"]
        cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
        sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} AvgLev={avg_lev:.2f}x")
        results[label] = res
        all_eq[label] = res["equity_curve"]

    # ══════════════════════════════════════════════════════════════════════
    # Strategy E: Breadth Tactical
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 Strategy E: Breadth-Based Tactical")
    configs_e = [
        ("E_Breadth(lev=1.5,vol=12%)", dict(max_lev=1.5, target_vol=0.12), None),
        ("E_Breadth(lev=1.5,vol=12%)+ModDD", dict(max_lev=1.5, target_vol=0.12), dd_moderate),
    ]
    for label, kwargs, dd in configs_e:
        w = strat_e_breadth_tactical(prices, **kwargs)
        res = run_bt(prices, w, rebal=5, dd_schedule=dd)
        m = res["metrics"]
        cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
        sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
        avg_lev = res["gross_exposure"].mean()
        print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} AvgLev={avg_lev:.2f}x")
        results[label] = res
        all_eq[label] = res["equity_curve"]

    # ══════════════════════════════════════════════════════════════════════
    # ENSEMBLES: Combine sub-strategies for diversification
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 ENSEMBLES: Combining sub-strategies for diversification benefit")

    # Pick best variant from each strategy family
    best_a = "A_ModDD(lev=1.5,vol=12%)"
    best_b = "B_RP(vol=10%)+ModDD"
    best_c = "C_MR(vol=10%)+ModDD"
    best_d = "D_Rot(n=3,vol=12%)+ModDD"
    best_e = "E_Breadth(lev=1.5,vol=12%)+ModDD"

    combos = [
        ("Ens_A+B",       [best_a, best_b],                     [0.60, 0.40]),
        ("Ens_A+B+C",     [best_a, best_b, best_c],             [0.50, 0.30, 0.20]),
        ("Ens_A+B+D",     [best_a, best_b, best_d],             [0.50, 0.30, 0.20]),
        ("Ens_A+B+C+D",   [best_a, best_b, best_c, best_d],    [0.40, 0.25, 0.15, 0.20]),
        ("Ens_A+E",        [best_a, best_e],                    [0.55, 0.45]),
        ("Ens_A+B+E",      [best_a, best_b, best_e],           [0.45, 0.25, 0.30]),
        ("Ens_ALL",        [best_a, best_b, best_c, best_d, best_e],
                                                                 [0.30, 0.20, 0.15, 0.15, 0.20]),
    ]

    for label, keys, wgts in combos:
        # Ensemble by weighting daily returns
        ens_ret = sum(w_i * results[k]["portfolio_returns"] for k, w_i in zip(keys, wgts))
        ens_eq = 100_000 * (1 + ens_ret).cumprod()
        ens_eq.name = "Equity"
        turn = sum(w_i * results[k]["turnover"] for k, w_i in zip(keys, wgts))
        g_exp = sum(w_i * results[k]["gross_exposure"] for k, w_i in zip(keys, wgts))
        m = compute_metrics(ens_ret, ens_eq, 100_000, risk_free_rate=RF,
                            turnover=turn, gross_exposure=g_exp)
        cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
        sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
        avg_lev = g_exp.mean()
        print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
              f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
              f"MaxDD={m['Max Drawdown']:7.2%} AvgLev={avg_lev:.2f}x")
        results[label] = {"equity_curve": ens_eq, "portfolio_returns": ens_ret,
                          "weights": pd.DataFrame(), "turnover": turn,
                          "gross_exposure": g_exp, "metrics": m}
        all_eq[label] = ens_eq

    # ══════════════════════════════════════════════════════════════════════
    # Leveraged ensembles: scale up the best ensembles
    # ══════════════════════════════════════════════════════════════════════
    print(SEP)
    print("🔄 Leveraged ensembles (scale up to boost CAGR)")
    for base_label in ["Ens_A+B", "Ens_A+B+C+D", "Ens_ALL"]:
        base_ret = results[base_label]["portfolio_returns"]
        for lev_mult in [1.3, 1.5]:
            label = f"{base_label}_x{lev_mult}"
            scaled_ret = base_ret * lev_mult
            # Extra leverage cost
            extra_cost = (lev_mult - 1) * LEV_COST / 252
            scaled_ret = scaled_ret - extra_cost
            eq = 100_000 * (1 + scaled_ret).cumprod()
            eq.name = "Equity"
            m = compute_metrics(scaled_ret, eq, 100_000, risk_free_rate=RF)
            cagr_ok = "✓" if m.get("CAGR", 0) > spy_cagr else " "
            sh_ok   = "✓" if m.get("Sharpe Ratio", 0) > 1.95 else " "
            print(f"   {label:46s} CAGR={m['CAGR']:7.2%}[{cagr_ok}] "
                  f"Sharpe={m['Sharpe Ratio']:7.4f}[{sh_ok}] "
                  f"MaxDD={m['Max Drawdown']:7.2%}")
            results[label] = {"equity_curve": eq, "portfolio_returns": scaled_ret,
                              "weights": pd.DataFrame(), "metrics": m,
                              "turnover": pd.Series(0, index=prices.index),
                              "gross_exposure": pd.Series(lev_mult, index=prices.index)}

    # ══════════════════════════════════════════════════════════════════════
    # AUDIT
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    auditable = {k: v for k, v in results.items() if not v["weights"].empty}
    iss = audit(auditable, prices)
    print(f"🔍 AUDIT")
    print(f"   Total issues: {len(iss)}")
    if iss:
        for i in iss:
            print(f"   ⚠️  {i}")
    else:
        print(f"   ✅ All audits pass")

    # ══════════════════════════════════════════════════════════════════════
    # CORRELATION MATRIX
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 Strategy CORRELATIONS (daily returns):")
    ret_df = pd.DataFrame({k: v["portfolio_returns"] for k, v in results.items()})
    corr = ret_df.corr()
    # Show correlation between the sub-strategy families
    families = [best_a, best_b, best_c, best_d, best_e]
    families = [f for f in families if f in corr.columns]
    if len(families) > 1:
        print(corr.loc[families, families].to_string(float_format=lambda x: f"{x:.3f}"))

    # ══════════════════════════════════════════════════════════════════════
    # SUB-PERIOD ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    # Sort all results by Sharpe
    sorted_res = sorted(results.items(), key=lambda x: x[1]["metrics"].get("Sharpe Ratio", 0),
                        reverse=True)
    top_labels = [k for k, _ in sorted_res[:8]]

    periods = [
        ("Full",    START, END),
        ("2015-25", "2015-01-01", END),
        ("2018-25", "2018-01-01", END),
        ("2020-25", "2020-01-01", END),
    ]
    print("📅 SUB-PERIOD ANALYSIS (top 8 by full-period Sharpe)\n")
    for pname, ps, pe in periods:
        mask = (prices.index >= ps) & (prices.index <= pe)
        spy_sub = prices["SPY"][mask]
        if len(spy_sub) < 100:
            continue
        spy_eq_sub = spy_sub / spy_sub.iloc[0] * 100_000
        spy_ret_sub = spy_sub.pct_change().fillna(0)
        spy_m_sub = compute_metrics(spy_ret_sub, spy_eq_sub, 100_000, risk_free_rate=RF)
        print(f"  {pname}: SPY CAGR={spy_m_sub['CAGR']:.2%}, Sharpe={spy_m_sub['Sharpe Ratio']:.4f}")

        for label in top_labels:
            res = results[label]
            pr = res["portfolio_returns"][mask]
            if len(pr) < 100:
                continue
            eq_sub = 100_000 * (1 + pr).cumprod()
            eq_sub.name = "Equity"
            m_sub = compute_metrics(pr, eq_sub, 100_000, risk_free_rate=RF)
            cagr_flag = " CAGR✓" if m_sub.get("CAGR", 0) > spy_m_sub.get("CAGR", 0) else ""
            sharpe_flag = " SHARPE✓" if m_sub.get("Sharpe Ratio", 0) > 1.95 else ""
            print(f"  {label:46s} CAGR={m_sub['CAGR']:7.2%} "
                  f"Sharpe={m_sub['Sharpe Ratio']:7.4f} "
                  f"MaxDD={m_sub['Max Drawdown']:7.2%}{cagr_flag}{sharpe_flag}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("📊 FINAL RESULTS (sorted by Sharpe)")
    print(f"{'Label':56s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'AvgLev':>7s}")
    print("─" * 84)
    print(f"{'SPY':56s} {spy_cagr:7.2%} {spy_sharpe:7.4f} {spy_dd:7.2%} {'1.00x':>7s}")
    print("─" * 84)

    winners = []
    for label, res in sorted_res:
        m = res["metrics"]
        avg_lev = res["gross_exposure"].mean() if hasattr(res["gross_exposure"], "mean") else 1.0
        cagr = m.get("CAGR", 0)
        sharpe = m.get("Sharpe Ratio", 0)
        dd = m.get("Max Drawdown", 0)
        print(f"{label:56s} {cagr:7.2%} {sharpe:7.4f} {dd:7.2%} {avg_lev:6.2f}x")
        if cagr > spy_cagr and sharpe > 1.95:
            winners.append(label)

    print()
    if winners:
        print(f"🏆 WINNER(S) meeting BOTH targets (CAGR > SPY & Sharpe > 1.95):")
        for w in winners:
            m = results[w]["metrics"]
            print(f"   {w}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")
    else:
        print("⚠️  No full-period winners yet.")
        # Show closest to both targets
        print("   Closest to both targets:")
        candidates = [(k, v) for k, v in results.items() if v["metrics"].get("CAGR", 0) > spy_cagr]
        candidates.sort(key=lambda x: x[1]["metrics"].get("Sharpe Ratio", 0), reverse=True)
        for label, res in candidates[:5]:
            m = res["metrics"]
            print(f"   {label}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")

    print(f"\nAudit: {len(iss)} issues")


if __name__ == "__main__":
    main()
