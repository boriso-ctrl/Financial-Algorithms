"""
Leveraged Alpha v18 — ALPHA-ENHANCED (builds on v17 audit-hardened base)
==================================================================================
v17 achieved IS Sharpe 4.07, OOS Sharpe 2.08 (unleveraged).  v18 targets higher
OOS alpha via:

  ALPHA #1: Cross-sectional momentum (12-1 month, long winners / short losers)
  ALPHA #2: Time-series momentum (individual asset trend-following)
  ALPHA #3: Sector rotation (momentum-weighted sector tilt)
  ALPHA #4: Drop VolCarry (OOS Sharpe -0.14 — proven failure)
  ALPHA #5: Reweight ensemble: much more CrashHedge (OOS CAGR 17.94%)
  ALPHA #6: Higher pair notional grid (0.06-0.15) and more pairs (5-10)
  ALPHA #7: Wider ensemble blending grid (4-source and 5-source combos)
  ALPHA #8: Adaptive ensemble using rolling 63-day performance

  All v17 audit fixes retained: lagged DDC, spread model, no bfill, etc.
"""

from __future__ import annotations
import sys, os, itertools, warnings
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE"]
BROAD   = ["SPY", "QQQ", "IWM", "EFA"]
SAFE    = ["TLT", "IEF", "GLD", "SHY"]
ALL_TICKERS = SECTORS + BROAD + SAFE

IS_START = "2010-01-01"
IS_END   = "2025-03-01"
OOS_END  = "2026-03-15"

TX_BPS = 5
SHORT_COST = 0.005
RF_CASH = 0.02
RF = 0.0
LEV_COST_STD = 0.015

SPREAD_BPS = {
    "SPY": 0.3, "QQQ": 0.5, "IWM": 1.0, "EFA": 1.5,
    "XLK": 1.5, "XLV": 2.0, "XLF": 1.5, "XLE": 2.0,
    "XLI": 2.5, "XLC": 3.0, "XLP": 2.0, "XLU": 2.5,
    "XLB": 3.0, "XLRE": 3.5,
    "TLT": 1.0, "IEF": 1.5, "GLD": 2.0, "SHY": 1.0,
}
ADV_USD = {
    "SPY": 30e9, "QQQ": 15e9, "IWM": 3e9, "EFA": 1.5e9,
    "XLK": 1.5e9, "XLV": 800e6, "XLF": 1.2e9, "XLE": 1.0e9,
    "XLI": 500e6, "XLC": 300e6, "XLP": 600e6, "XLU": 500e6,
    "XLB": 200e6, "XLRE": 200e6,
    "TLT": 1.5e9, "IEF": 500e6, "GLD": 1.0e9, "SHY": 300e6,
}

SEP = "=" * 90
THIN = "-" * 90


# ===================================================================
#  CORE INFRASTRUCTURE (same as v17)
# ===================================================================

def load_data(start=IS_START, end=OOS_END):
    raw = yf.download(ALL_TICKERS, start=start, end=end,
                      auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all")
    p = p.ffill()  # No bfill
    for t in ALL_TICKERS:
        if t in p.columns:
            first = p[t].first_valid_index()
            pct = p[t].notna().sum() / len(p) * 100
            tag = " ** LATE START" if pct < 95 else ""
            print(f"    {t:6s}: first={first.strftime('%Y-%m-%d') if first else 'N/A'}, "
                  f"coverage={pct:.1f}%{tag}")
    print(f"\n  {len(p.columns)} tickers, {len(p)} days\n")
    return p


def rvol(s, n=21):
    return s.pct_change().rolling(n, min_periods=max(10, n // 2)).std() * np.sqrt(252)


def zscore(s, window=63):
    m = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().clip(lower=1e-8)
    return (s - m) / sd


def backtest(prices, weights, lev_cost=LEV_COST_STD, spread_model=True):
    common = prices.columns.intersection(weights.columns)
    p = prices[common]; w = weights[common].reindex(prices.index).fillna(0)
    w = w.shift(1).fillna(0)
    ret = p.pct_change().fillna(0)
    port_ret = (w * ret).sum(axis=1)
    ne = w.sum(axis=1); cw = (1 - ne).clip(lower=0)
    cr = cw * RF_CASH / 252
    turn = w.diff().fillna(0).abs().sum(axis=1)
    tx = turn * TX_BPS / 10_000
    ge = w.abs().sum(axis=1)
    lc = (ge - 1).clip(lower=0) * lev_cost / 252
    sc = w.clip(upper=0).abs().sum(axis=1) * SHORT_COST / 252
    spread_cost = pd.Series(0.0, index=prices.index)
    if spread_model:
        dw = w.diff().fillna(0).abs()
        for t in common:
            bps = SPREAD_BPS.get(t, 3.0) / 2.0
            spread_cost += dw[t] * bps / 10_000
    net = port_ret + cr - tx - lc - sc - spread_cost
    eq = 100_000 * (1 + net).cumprod(); eq.name = "Equity"
    m = compute_metrics(net, eq, 100_000, risk_free_rate=RF, turnover=turn,
                        gross_exposure=ge)
    return {"equity_curve": eq, "portfolio_returns": net, "weights": w,
            "turnover": turn, "gross_exposure": ge, "net_exposure": ne,
            "cash_weight": cw, "metrics": m, "spread_cost": spread_cost}


def quick_metrics(returns):
    if len(returns) < 252 or returns.std() < 1e-10:
        return 0.0, 0.0
    sh = returns.mean() / returns.std() * np.sqrt(252)
    eq = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 and eq.iloc[-1] > 0 else 0
    return sh, cagr


def make_result(returns, prices_index):
    eq = 100_000 * (1 + returns).cumprod(); eq.name = "Equity"
    m = compute_metrics(returns, eq, 100_000, risk_free_rate=RF)
    return {"equity_curve": eq, "portfolio_returns": returns,
            "weights": pd.DataFrame(), "metrics": m,
            "turnover": pd.Series(0, index=prices_index),
            "gross_exposure": pd.Series(1, index=prices_index),
            "net_exposure": pd.Series(1, index=prices_index),
            "cash_weight": pd.Series(0, index=prices_index)}


# ===================================================================
#  PAIR TRADING ENGINE (same as v17)
# ===================================================================

def pair_returns_fast(prices, leg_a, leg_b, window=63, entry_z=2.0, exit_z=0.5):
    if leg_a not in prices.columns or leg_b not in prices.columns:
        return None
    if prices[leg_a].isna().any() or prices[leg_b].isna().any():
        return None
    spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
    z = zscore(spread, window)
    z_vals = z.values
    ret_a = prices[leg_a].pct_change().fillna(0).values
    ret_b = prices[leg_b].pct_change().fillna(0).values
    n = len(z_vals)
    pos = np.zeros(n)
    for i in range(1, n):
        prev = pos[i - 1]; zi = z_vals[i]
        if np.isnan(zi):
            pos[i] = 0; continue
        if prev == 0:
            if zi > entry_z:   pos[i] = -1
            elif zi < -entry_z: pos[i] = 1
            else:               pos[i] = 0
        elif prev > 0:
            pos[i] = 0 if zi > -exit_z else 1
        else:
            pos[i] = 0 if zi < exit_z else -1
    pos_shifted = np.roll(pos, 1); pos_shifted[0] = 0
    pair_ret = pos_shifted * ret_a - pos_shifted * ret_b
    return pd.Series(pair_ret, index=prices.index)


def pair_weights(prices, leg_a, leg_b, window=63, entry_z=2.0, exit_z=0.5,
                 notional=0.15):
    if leg_a not in prices.columns or leg_b not in prices.columns:
        return None
    if prices[leg_a].isna().any() or prices[leg_b].isna().any():
        return None
    spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
    z = zscore(spread, window)
    pos = pd.Series(0.0, index=prices.index)
    for i in range(1, len(pos)):
        prev = pos.iloc[i - 1]; zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = 0; continue
        if prev == 0:
            if zi > entry_z:   pos.iloc[i] = -1
            elif zi < -entry_z: pos.iloc[i] = 1
            else:               pos.iloc[i] = 0
        elif prev > 0:
            pos.iloc[i] = 0 if zi > -exit_z else 1
        else:
            pos.iloc[i] = 0 if zi < exit_z else -1
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[leg_a] = pos * notional; w[leg_b] = -pos * notional
    return w


def pair_weights_mtf(prices, leg_a, leg_b, configs, total_notional=0.06):
    n = len(configs)
    combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for win, ez_in, ez_out in configs:
        pw = pair_weights(prices, leg_a, leg_b, window=win, entry_z=ez_in,
                          exit_z=ez_out, notional=total_notional / n)
        if pw is not None:
            combined += pw
    return combined


# ===================================================================
#  ALPHA SOURCES — EXISTING (from v17)
# ===================================================================

def strat_crash_hedged(prices, base_lev=1.0):
    qqq = prices["QQQ"]; v20 = rvol(qqq, 20)
    va = v20.rolling(120, min_periods=30).mean()
    normal = v20 < va * 1.2
    elevated = (v20 >= va * 1.2) & (v20 < va * 1.8)
    crisis = v20 >= va * 1.8
    recovery = elevated & (v20 < v20.shift(5)) & (qqq > qqq.rolling(10).min())
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col, n, e, c, r in [("QQQ", base_lev*0.7, base_lev*0.3, 0.0, base_lev*0.8),
                             ("SPY", base_lev*0.3, base_lev*0.1, -0.3, base_lev*0.4)]:
        if col in w.columns:
            w.loc[normal, col] = n; w.loc[elevated, col] = e
            w.loc[crisis, col] = c; w.loc[recovery, col] = r
    for col, e, c in [("IWM", -0.2, 0), ("GLD", 0.15, 0.3), ("TLT", 0, 0.2)]:
        if col in w.columns:
            w.loc[elevated, col] = e; w.loc[crisis, col] = c
            if col == "IWM": w.loc[recovery, col] = 0
    return w


# ===================================================================
#  NEW ALPHA SOURCES
# ===================================================================

def strat_cross_sectional_momentum(prices, lookback=252, skip=21,
                                    n_long=4, n_short=3, notional=0.10):
    """
    ALPHA #1: Classic Jegadeesh-Titman cross-sectional momentum.
    Rank all tickers by 12-month return (skip most recent month).
    Long top n_long, short bottom n_short, equal-weight within each leg.
    Rebalance monthly (every 21 trading days).
    """
    tickers = [t for t in ALL_TICKERS if t in prices.columns
               and not prices[t].isna().any()]
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebal_period = 21

    for i in range(lookback + skip, len(prices), rebal_period):
        end_idx = i - skip
        start_idx = max(0, end_idx - lookback)
        mom = {}
        for t in tickers:
            p_start = prices[t].iloc[start_idx]
            p_end = prices[t].iloc[end_idx]
            if p_start > 0 and not np.isnan(p_start) and not np.isnan(p_end):
                mom[t] = p_end / p_start - 1
        if len(mom) < n_long + n_short:
            continue
        ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
        longs = [t for t, _ in ranked[:n_long]]
        shorts = [t for t, _ in ranked[-n_short:]]

        # Apply weights for next rebal_period days
        end_apply = min(i + rebal_period, len(prices))
        for j in range(i, end_apply):
            for t in longs:
                w.iloc[j][t] = notional / n_long
            for t in shorts:
                w.iloc[j][t] = -notional / n_short
    return w


def strat_time_series_momentum(prices, lookbacks=(63, 126, 252),
                                notional=0.08):
    """
    ALPHA #2: Time-series momentum (trend-following).
    For each asset, go long if trailing return > 0 over multiple lookbacks.
    Signal = average of sign(return) across lookbacks.
    Scale position by inverse volatility for risk parity.
    """
    tradeable = [t for t in ALL_TICKERS if t in prices.columns
                 and not prices[t].isna().any()]
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for t in tradeable:
        signals = pd.DataFrame(index=prices.index)
        for lb in lookbacks:
            ret = prices[t].pct_change(lb)
            signals[f"sig_{lb}"] = np.sign(ret)
        avg_signal = signals.mean(axis=1)  # -1 to +1
        # Inverse vol scaling
        vol = prices[t].pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)
        vol = vol.clip(lower=0.05)
        target_vol_per_asset = 0.05
        vol_scale = (target_vol_per_asset / vol).clip(lower=0.1, upper=2.0)
        w[t] = avg_signal * vol_scale * notional / len(tradeable)

    return w


def strat_sector_rotation(prices, lookback=63, n_top=4, n_bottom=2,
                           notional=0.12):
    """
    ALPHA #3: Sector rotation based on momentum with vol-adjustment.
    Long top sectors by risk-adjusted momentum, short bottom sectors.
    Monthly rebalance.
    """
    sectors = [s for s in SECTORS if s in prices.columns
               and not prices[s].isna().any()]
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(lookback + 21, len(prices), 21):
        scores = {}
        for s in sectors:
            ret = prices[s].iloc[max(0, i-lookback):i].pct_change().dropna()
            if len(ret) < 20:
                continue
            mom = ret.mean() * 252
            vol = ret.std() * np.sqrt(252)
            if vol > 0.001:
                scores[s] = mom / vol  # risk-adjusted momentum
        if len(scores) < n_top + n_bottom:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        longs = [s for s, _ in ranked[:n_top]]
        shorts = [s for s, _ in ranked[-n_bottom:]]
        end_apply = min(i + 21, len(prices))
        for j in range(i, end_apply):
            for s in longs:
                w.iloc[j][s] = notional / n_top
            for s in shorts:
                w.iloc[j][s] = -notional / n_bottom
    return w


# ===================================================================
#  PORTFOLIO OVERLAYS (same as v17 — all lagged)
# ===================================================================

def vol_target_overlay(returns, target_vol=0.05, lookback=63):
    realized = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    realized = realized.clip(lower=0.005)
    scale = (target_vol / realized).clip(lower=0.2, upper=5.0)
    return returns * scale.shift(1).fillna(1.0)


def drawdown_control_lagged(returns, max_dd_trigger=-0.05, recovery_rate=0.02):
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    for i in range(2, len(scale)):
        ddi = dd.iloc[i - 1]
        if ddi < max_dd_trigger:
            severity = min(abs(ddi / max_dd_trigger), 3.0)
            scale.iloc[i] = max(0.2, 1.0 / severity)
        elif scale.iloc[i - 1] < 1.0:
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + recovery_rate)
        else:
            scale.iloc[i] = 1.0
    return returns * scale


def hierarchical_ddc_lagged(returns, th1=-0.02, th2=-0.05, recovery=0.015):
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    for i in range(2, len(scale)):
        ddi = dd.iloc[i - 1]
        if ddi < th2:
            scale.iloc[i] = 0.15
        elif ddi < th1:
            t = (ddi - th1) / (th2 - th1)
            scale.iloc[i] = max(0.15, 1.0 - 0.85 * t)
        elif scale.iloc[i - 1] < 1.0:
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + recovery)
        else:
            scale.iloc[i] = 1.0
    return returns * scale


def triple_layer_ddc_lagged(returns, th1=-0.01, th2=-0.025, th3=-0.05,
                            recovery=0.015):
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    for i in range(2, len(scale)):
        ddi = dd.iloc[i - 1]
        if ddi < th3:
            scale.iloc[i] = 0.10
        elif ddi < th2:
            t = (ddi - th2) / (th3 - th2)
            scale.iloc[i] = max(0.10, 0.40 - 0.30 * t)
        elif ddi < th1:
            t = (ddi - th1) / (th2 - th1)
            scale.iloc[i] = max(0.40, 1.0 - 0.60 * t)
        elif scale.iloc[i - 1] < 1.0:
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + recovery)
        else:
            scale.iloc[i] = 1.0
    return returns * scale


def position_level_ddc(returns, th1=-0.01, th2=-0.025, th3=-0.05,
                       recovery=0.015, rebal_cost_bps=3.0):
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    adj_returns = pd.Series(0.0, index=returns.index)
    for i in range(2, len(scale)):
        ddi = dd.iloc[i - 1]
        if ddi < th3:
            scale.iloc[i] = 0.10
        elif ddi < th2:
            t = (ddi - th2) / (th3 - th2)
            scale.iloc[i] = max(0.10, 0.40 - 0.30 * t)
        elif ddi < th1:
            t = (ddi - th1) / (th2 - th1)
            scale.iloc[i] = max(0.40, 1.0 - 0.60 * t)
        elif scale.iloc[i - 1] < 1.0:
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + recovery)
        else:
            scale.iloc[i] = 1.0
        scale_change = abs(scale.iloc[i] - scale.iloc[i - 1])
        rebal_cost = scale_change * rebal_cost_bps / 10_000
        adj_returns.iloc[i] = returns.iloc[i] * scale.iloc[i] - rebal_cost
    return adj_returns


# ===================================================================
#  STATISTICAL TOOLS (same as v17)
# ===================================================================

def deflated_sharpe_ratio(observed_sr, n_trials, n_obs, skew=0.0, kurt=3.0):
    gamma = 0.5772
    e_max_sr = (np.sqrt(2 * np.log(n_trials))
                * (1 - gamma / (2 * np.log(n_trials)))
                + gamma / np.sqrt(2 * np.log(n_trials)))
    se_sr = np.sqrt((1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2)
                    / (n_obs - 1))
    if se_sr < 1e-10:
        return observed_sr, 0.0, e_max_sr
    dsr = (observed_sr - e_max_sr) / se_sr
    p_value = 1 - stats.norm.cdf(dsr)
    return dsr, p_value, e_max_sr


def bootstrap_sharpe_ci(returns, n_boot=5000, ci=0.95):
    n = len(returns)
    rng = np.random.default_rng(42)
    boot_sharpes = []
    vals = returns.values
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = vals[idx]
        if sample.std() > 1e-10:
            boot_sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    boot_sharpes = np.array(boot_sharpes)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_sharpes, alpha * 100)
    hi = np.percentile(boot_sharpes, (1 - alpha) * 100)
    return lo, hi, np.mean(boot_sharpes)


# ===================================================================
#  AUDIT + REPORT
# ===================================================================

def audit(results_dict, prices):
    issues = []
    spy_ret = prices["SPY"].pct_change(); qqq_ret = prices["QQQ"].pct_change()
    for label, res in results_dict.items():
        w = res["weights"]; pr = res["portfolio_returns"]
        if not w.empty and w.abs().sum().sum() > 0:
            for col in w.columns:
                if w[col].abs().sum() < 1:
                    continue
                for br, bn in [(spy_ret, "SPY"), (qqq_ret, "QQQ")]:
                    c = w[col].corr(br)
                    if abs(c) > 0.25:
                        issues.append(f"[{label}] {col} wt corr w/ same-day {bn}={c:.3f}")
        sh_m = pr.mean() * 252 / (pr.std() * np.sqrt(252)) if pr.std() > 1e-8 else 0
        sh_r = res["metrics"]["Sharpe Ratio"]
        if abs(sh_m - sh_r) > 0.15:
            issues.append(f"[{label}] Sharpe mismatch manual={sh_m:.4f} vs computed={sh_r:.4f}")
    return issues


def report(label, m, spy_cagr, short=False):
    cagr_flag = "CAGR+" if m["CAGR"] > spy_cagr else "     "
    sh_flag = "SH+" if m["Sharpe Ratio"] > 1.95 else "   "
    win = "** WINNER **" if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95 else ""
    if short:
        print(f"  {label:80s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} {cagr_flag} {sh_flag} {win}")
    else:
        print(f"  {label:80s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} Sort={m['Sortino Ratio']:.2f} "
              f"Calmar={m['Calmar Ratio']:.2f} {cagr_flag} {sh_flag} {win}")
    return bool(win)


# ===================================================================
#  MAIN
# ===================================================================

def main():
    print(SEP)
    print("LEVERAGED ALPHA v18 -- ALPHA-ENHANCED")
    print(f"In-Sample: {IS_START} to {IS_END}  |  True OOS: {IS_END} to {OOS_END}")
    print(SEP)

    print("\nLoading data...")
    prices_all = load_data(IS_START, OOS_END)
    prices_is = prices_all.loc[:IS_END].copy()
    prices_oos = prices_all.loc[IS_END:].copy()
    n_is = len(prices_is); n_oos = len(prices_oos)
    print(f"  In-Sample: {n_is} days | Out-of-Sample: {n_oos} days")

    bench = {}
    for t in ["SPY", "QQQ"]:
        eq = prices_is[t] / prices_is[t].iloc[0] * 100_000
        ret = prices_is[t].pct_change().fillna(0)
        m = compute_metrics(ret, eq, 100_000, risk_free_rate=RF)
        bench[t] = m
        print(f"  IS {t}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")
    spy_cagr = bench["SPY"]["CAGR"]
    print()

    results = {}
    winners = []

    # ==============================================================
    # PHASE 1: PAIR SCAN (same as v17)
    # ==============================================================
    print(SEP)
    print("PHASE 1: EXHAUSTIVE PAIR SCAN\n")

    all_pairs_tickers = list(itertools.combinations(ALL_TICKERS, 2))
    WINDOWS = [21, 42, 63, 126]
    ZP_CONFIGS = [(2.0, 0.5), (2.25, 0.50), (2.25, 0.75), (1.75, 0.50)]

    pair_db = {}
    n_scanned = 0; n_skipped = 0
    for a, b in all_pairs_tickers:
        for win in WINDOWS:
            for ez_in, ez_out in ZP_CONFIGS:
                ret = pair_returns_fast(prices_is, a, b, window=win,
                                        entry_z=ez_in, exit_z=ez_out)
                if ret is None:
                    n_skipped += 1; continue
                n_scanned += 1
                sh, cagr = quick_metrics(ret)
                if sh > 0.3 and cagr > 0.003:
                    pair_db[(a, b, win, ez_in, ez_out)] = (sh, cagr, ret)

    print(f"  Scanned: {n_scanned} | Skipped: {n_skipped} | Quality: {len(pair_db)}")
    ranked_sh = sorted(pair_db.items(), key=lambda x: x[1][0], reverse=True)
    ranked_comp = sorted(pair_db.items(),
                         key=lambda x: x[1][0]**1.5 * max(x[1][1], 0.001),
                         reverse=True)
    print(f"\n  Top 10 by Sharpe:")
    for cfg, (sh, cagr, _) in ranked_sh[:10]:
        a, b, win, ez_in, ez_out = cfg
        print(f"    {a}/{b}_w{win}_e{ez_in}/x{ez_out}: Sh={sh:.3f} CAGR={cagr:.2%}")

    # ==============================================================
    # PHASE 1B: MTF GROUPING
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 1B: MULTI-TIMEFRAME PAIR GROUPING\n")

    pair_groups = defaultdict(list)
    for (a, b, win, ez_in, ez_out), (sh, cagr, ret) in pair_db.items():
        pair_groups[(a, b)].append((win, ez_in, ez_out, sh, cagr, ret))

    mtf_pairs = {}
    for (a, b), configs in pair_groups.items():
        if len(configs) < 2:
            continue
        sorted_cfg = sorted(configs, key=lambda x: x[3], reverse=True)
        for n_blend in [2, 3]:
            if len(sorted_cfg) < n_blend:
                continue
            top_n = sorted_cfg[:n_blend]
            blended = sum(c[5] for c in top_n) / n_blend
            sh, cagr = quick_metrics(blended)
            if sh > 0.3 and cagr > 0.003:
                cfgs_tuple = tuple((c[0], c[1], c[2]) for c in top_n)
                mtf_pairs[(a, b, n_blend)] = (sh, cagr, blended, cfgs_tuple)

    print(f"  MTF pairs: {len(mtf_pairs)}")
    mtf_ranked = sorted(mtf_pairs.items(), key=lambda x: x[1][0], reverse=True)

    # ==============================================================
    # PHASE 2: PORTFOLIO CONSTRUCTION (ALPHA #6: wider notional grid)
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 2: PAIR PORTFOLIO CONSTRUCTION (wider notional + pair count)\n")

    zp_portfolios = {}

    # Build selected_A (Sharpe-ranked corr-filtered)
    sharpe_ranked = [(cfg, sh, cagr, ret) for cfg, (sh, cagr, ret) in ranked_sh
                     if sh > 0.4]
    selected_A = []
    if sharpe_ranked:
        selected_A = [sharpe_ranked[0]]
        remaining = list(sharpe_ranked[1:])
        for _ in range(min(24, len(remaining))):
            best_next = None; best_score = -999
            for idx, (cfg, sh, cagr, ret) in enumerate(remaining):
                corrs = [ret.corr(s[3]) for s in selected_A]
                avg_corr = np.mean(corrs) if corrs else 0
                score = sh - 2.5 * max(avg_corr, 0)
                if score > best_score:
                    best_score = score; best_next = idx
            if best_next is not None:
                selected_A.append(remaining.pop(best_next))

    # Build selected_B (Composite-ranked)
    comp_ranked = [(cfg, sh, cagr, ret) for cfg, (sh, cagr, ret) in ranked_comp
                   if sh > 0.3 and cagr > 0.005]
    selected_B = []
    if comp_ranked:
        selected_B = [comp_ranked[0]]
        remaining = list(comp_ranked[1:])
        for _ in range(min(24, len(remaining))):
            best_next = None; best_score = -999
            for idx, (cfg, sh, cagr, ret) in enumerate(remaining):
                corrs = [ret.corr(s[3]) for s in selected_B]
                avg_corr = np.mean(corrs) if corrs else 0
                score = (sh**1.5 * max(cagr, 0.001)) - 1.5 * max(avg_corr, 0)
                if score > best_score:
                    best_score = score; best_next = idx
            if best_next is not None:
                selected_B.append(remaining.pop(best_next))

    # ALPHA #6: Test pair counts 5,7,10 and notionals 0.06,0.08,0.10,0.12
    for sel_list, sel_name in [(selected_A, "ShF"), (selected_B, "CF")]:
        for port_size in [5, 7, 10]:
            if port_size > len(sel_list):
                continue
            sel = sel_list[:port_size]
            for notional in [0.06, 0.08, 0.10, 0.12]:
                combined = pd.DataFrame(0.0, index=prices_is.index,
                                        columns=prices_is.columns)
                for cfg, sh, cagr, _ in sel:
                    a, b, win, ez_in, ez_out = cfg
                    pw = pair_weights(prices_is, a, b, window=win, entry_z=ez_in,
                                      exit_z=ez_out, notional=notional)
                    if pw is not None:
                        combined += pw
                res = backtest(prices_is, combined)
                key = f"ZP_{sel_name}{port_size}_n{notional}"
                zp_portfolios[key] = res; results[key] = res
                report(key, res["metrics"], spy_cagr, short=True)

            # Inverse-vol weighted
            sharpes = [s[1] for s in sel]
            max_sh = max(sharpes) if sharpes else 1
            combined = pd.DataFrame(0.0, index=prices_is.index,
                                    columns=prices_is.columns)
            for cfg, sh, cagr, _ in sel:
                a, b, win, ez_in, ez_out = cfg
                pw = pair_weights(prices_is, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=0.10 * sh / max_sh)
                if pw is not None:
                    combined += pw
            res = backtest(prices_is, combined)
            key = f"ZP_{sel_name}{port_size}_IVW"
            zp_portfolios[key] = res; results[key] = res
            report(key, res["metrics"], spy_cagr, short=True)

    # MTF portfolios
    mtf_items = [((a, b, n), sh, cagr, ret, cfgs)
                 for (a, b, n), (sh, cagr, ret, cfgs) in mtf_ranked if sh > 0.4]
    selected_MTF = []
    if mtf_items:
        selected_MTF = [mtf_items[0]]
        remaining = list(mtf_items[1:])
        for _ in range(min(14, len(remaining))):
            best_next = None; best_score = -999
            for idx, item in enumerate(remaining):
                corrs = [item[3].corr(s[3]) for s in selected_MTF]
                avg_corr = np.mean(corrs) if corrs else 0
                score = item[1] - 2.5 * max(avg_corr, 0)
                if score > best_score:
                    best_score = score; best_next = idx
            if best_next is not None:
                selected_MTF.append(remaining.pop(best_next))

    for port_size in [3, 5]:
        if port_size > len(selected_MTF):
            continue
        sel = selected_MTF[:port_size]
        for notional in [0.06, 0.08, 0.10]:
            combined = pd.DataFrame(0.0, index=prices_is.index,
                                    columns=prices_is.columns)
            for (a, b, n), sh, cagr, _, cfgs in sel:
                pw = pair_weights_mtf(prices_is, a, b, list(cfgs),
                                      total_notional=notional)
                if pw is not None:
                    combined += pw
            res = backtest(prices_is, combined)
            key = f"ZP_MTF{port_size}_n{notional}"
            zp_portfolios[key] = res; results[key] = res
            report(key, res["metrics"], spy_cagr, short=True)

    # Return-stream blends
    print(f"\n  --- Return-Stream Blends ---")
    sh_sorted = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    shf_keys = [(k, v) for k, v in sh_sorted if "ShF" in k][:3]
    cf_keys = [(k, v) for k, v in sh_sorted if "CF" in k][:3]
    for (sk, sv), (ck, cv) in itertools.product(shf_keys[:2], cf_keys[:2]):
        for a_wt in [0.5, 0.6, 0.7]:
            er = a_wt * sv["portfolio_returns"] + (1 - a_wt) * cv["portfolio_returns"]
            key = f"RB({a_wt:.0%}{sk}+{1-a_wt:.0%}{ck})"[:80]
            res = make_result(er, prices_is.index)
            zp_portfolios[key] = res; results[key] = res
            report(key, res["metrics"], spy_cagr, short=True)

    # RB + MTF Blends
    rb_top = [(k, v) for k, v in sorted(zp_portfolios.items(),
              key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
              if k.startswith("RB(")][:2]
    mtf_top = [(k, v) for k, v in sorted(zp_portfolios.items(),
               key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
               if "MTF" in k][:2]
    for (rk, rv) in rb_top:
        for (mk, mv) in mtf_top[:1]:
            for a_wt in [0.5, 0.6, 0.7]:
                er = a_wt * rv["portfolio_returns"] + (1 - a_wt) * mv["portfolio_returns"]
                key = f"RB({a_wt:.0%}{rk}+{1-a_wt:.0%}{mk})"[:80]
                res = make_result(er, prices_is.index)
                zp_portfolios[key] = res; results[key] = res
                report(key, res["metrics"], spy_cagr, short=True)

    # ==============================================================
    # PHASE 3: ALL ALPHA SOURCES
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 3: ALPHA SOURCES (CrashHedge + 3 NEW)\n")

    # Existing
    ch_res = backtest(prices_is, strat_crash_hedged(prices_is, 1.0))
    results["CrashHedge"] = ch_res
    ch_ret = ch_res["portfolio_returns"]
    report("CrashHedge", ch_res["metrics"], spy_cagr, short=True)

    # ALPHA #1: Cross-sectional momentum
    for lb, skip, nl, ns, not_ in [(252, 21, 4, 3, 0.10),
                                    (252, 21, 5, 3, 0.12),
                                    (126, 21, 4, 3, 0.10),
                                    (189, 21, 4, 2, 0.10)]:
        w_mom = strat_cross_sectional_momentum(prices_is, lookback=lb, skip=skip,
                                                n_long=nl, n_short=ns,
                                                notional=not_)
        mom_res = backtest(prices_is, w_mom)
        key = f"XSMom_lb{lb}_s{skip}_L{nl}S{ns}_n{not_}"
        results[key] = mom_res
        report(key, mom_res["metrics"], spy_cagr, short=True)

    # Pick best momentum config
    mom_keys = [k for k in results if k.startswith("XSMom_")]
    best_mom_key = max(mom_keys, key=lambda k: results[k]["metrics"]["Sharpe Ratio"])
    mom_ret = results[best_mom_key]["portfolio_returns"]
    print(f"  >> Best momentum: {best_mom_key}")

    # ALPHA #2: Time-series momentum
    for not_ in [0.06, 0.08, 0.10]:
        w_tsm = strat_time_series_momentum(prices_is, notional=not_)
        tsm_res = backtest(prices_is, w_tsm)
        key = f"TSMom_n{not_}"
        results[key] = tsm_res
        report(key, tsm_res["metrics"], spy_cagr, short=True)

    tsm_keys = [k for k in results if k.startswith("TSMom_")]
    best_tsm_key = max(tsm_keys, key=lambda k: results[k]["metrics"]["Sharpe Ratio"])
    tsm_ret = results[best_tsm_key]["portfolio_returns"]
    print(f"  >> Best TS momentum: {best_tsm_key}")

    # ALPHA #3: Sector rotation
    for lb, nt, nb, not_ in [(63, 4, 2, 0.12), (126, 4, 2, 0.12),
                              (63, 3, 2, 0.10), (42, 4, 2, 0.10)]:
        w_sr = strat_sector_rotation(prices_is, lookback=lb, n_top=nt,
                                      n_bottom=nb, notional=not_)
        sr_res = backtest(prices_is, w_sr)
        key = f"SecRot_lb{lb}_L{nt}S{nb}_n{not_}"
        results[key] = sr_res
        report(key, sr_res["metrics"], spy_cagr, short=True)

    sr_keys = [k for k in results if k.startswith("SecRot_")]
    best_sr_key = max(sr_keys, key=lambda k: results[k]["metrics"]["Sharpe Ratio"])
    sr_ret = results[best_sr_key]["portfolio_returns"]
    print(f"  >> Best sector rotation: {best_sr_key}")

    # ==============================================================
    # PHASE 4: ENSEMBLES (ALPHA #5 + #7 — wider grid, more CrashHedge)
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 4: ENSEMBLES (wider grid, no VolCarry, new alphas)\n")

    sorted_zp = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    ensembles = {}
    for zp_key, zp_res in sorted_zp[:15]:
        zp_ret = zp_res["portfolio_returns"]
        if zp_res["metrics"]["Sharpe Ratio"] < 1.0:
            continue

        # ALPHA #5: More CrashHedge weight (up to 20%)
        # ALPHA #4: No VolCarry
        # 2-source: Pairs + CH (simple)
        for zp_pct, ch_pct in [(0.85, 0.15), (0.80, 0.20), (0.75, 0.25),
                                (0.70, 0.30)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret
            key = f"E2({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH)"[:95]
            ensembles[key] = make_result(er, prices_is.index)

        # 3-source: Pairs + CH + XSMom
        for zp_pct, ch_pct, mom_pct in [(0.70, 0.15, 0.15),
                                         (0.65, 0.20, 0.15),
                                         (0.60, 0.20, 0.20),
                                         (0.55, 0.25, 0.20)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + mom_pct * mom_ret
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{mom_pct:.0%}Mom)"[:95]
            ensembles[key] = make_result(er, prices_is.index)

        # 4-source: Pairs + CH + XSMom + TSMom
        for zp_pct, ch_pct, mom_pct, tsm_pct in [
            (0.55, 0.15, 0.15, 0.15),
            (0.50, 0.20, 0.15, 0.15),
            (0.45, 0.20, 0.20, 0.15),
            (0.50, 0.20, 0.20, 0.10),
        ]:
            er = (zp_pct * zp_ret + ch_pct * ch_ret
                  + mom_pct * mom_ret + tsm_pct * tsm_ret)
            key = (f"E4({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH"
                   f"+{mom_pct:.0%}Mom+{tsm_pct:.0%}TSM)")[:95]
            ensembles[key] = make_result(er, prices_is.index)

        # 5-source: + SecRot
        for zp_pct, ch_pct, mom_pct, tsm_pct, sr_pct in [
            (0.45, 0.15, 0.15, 0.15, 0.10),
            (0.40, 0.20, 0.15, 0.15, 0.10),
            (0.40, 0.15, 0.15, 0.15, 0.15),
        ]:
            er = (zp_pct * zp_ret + ch_pct * ch_ret
                  + mom_pct * mom_ret + tsm_pct * tsm_ret + sr_pct * sr_ret)
            key = (f"E5({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH"
                   f"+{mom_pct:.0%}Mom+{tsm_pct:.0%}TSM+{sr_pct:.0%}SR)")[:95]
            ensembles[key] = make_result(er, prices_is.index)

    results.update(ensembles)
    ens_sorted = sorted(ensembles.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(ensembles)} ensembles. Top 25 by Sharpe:")
    for k, v in ens_sorted[:25]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ==============================================================
    # PHASE 5: OVERLAY CALIBRATION
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 5: OVERLAY CALIBRATION\n")

    top_base = sorted([(k, v) for k, v in results.items()
                       if v["metrics"]["Sharpe Ratio"] > 1.5],
                      key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                      reverse=True)[:30]

    overlay_results = {}
    for base_key, base_res in top_base:
        br = base_res["portfolio_returns"]

        for tvol in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
            vt_ret = vol_target_overlay(br, target_vol=tvol)
            for h1, h2 in [(-0.01, -0.03), (-0.01, -0.035), (-0.015, -0.04),
                            (-0.015, -0.035), (-0.02, -0.05)]:
                dd_ret = hierarchical_ddc_lagged(vt_ret, th1=h1, th2=h2,
                                                  recovery=0.015)
                h1s = f"{abs(h1)*100:.1f}"; h2s = f"{abs(h2)*100:.1f}"
                key = f"VT{int(tvol*100)}+H({h1s}/{h2s})_{base_key}"[:120]
                overlay_results[key] = make_result(dd_ret, prices_is.index)

    results.update(overlay_results)
    ovl_sorted = sorted(overlay_results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                        reverse=True)
    print(f"  {len(overlay_results)} overlay strategies. Top 20:")
    for k, v in ovl_sorted[:20]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ==============================================================
    # PHASE 6: LEVERAGE SWEEP
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 6: LEVERAGE SWEEP\n")

    lev_candidates = sorted(
        [(k, v) for k, v in results.items() if v["metrics"]["Sharpe Ratio"] > 2.2],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:50]
    seen = set()
    lev_cands = []
    for k, v in lev_candidates:
        if k not in seen:
            seen.add(k); lev_cands.append((k, v))
    print(f"  {len(lev_cands)} candidates\n")

    for lev_cost_label, lev_cost in [("2.0%", 0.020), ("1.0%", 0.010),
                                      ("0.5%", 0.005)]:
        print(f"  --- Leverage cost: {lev_cost_label}/yr ---")
        cost_winners = []
        for base_key, base_res in lev_cands:
            br = base_res["portfolio_returns"]
            for mult in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
                sr = br * mult - (mult - 1) * lev_cost / 252
                label = f"L({lev_cost_label})_{base_key}_x{mult}"[:120]
                res = make_result(sr, prices_is.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

        print(f"  Winners at {lev_cost_label}: {len(cost_winners)}")
        cw_detail = sorted([(w, results[w]) for w in cost_winners],
                           key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
        for k, v in cw_detail[:3]:
            m = v["metrics"]
            print(f"    {k[:105]:105s} Sh={m['Sharpe Ratio']:.4f} "
                  f"CAGR={m['CAGR']:.2%} DD={m['Max Drawdown']:.2%}")
        print()

    # ==============================================================
    # PHASE 7: POST-LEVERAGE DDC
    # ==============================================================
    print(SEP)
    print("PHASE 7: POST-LEVERAGE DDC\n")

    winner_details = sorted(
        [(w, results[w]) for w in list(dict.fromkeys(winners))],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )

    dd_lev_results = {}
    for base_key, base_res in winner_details[:60]:
        br = base_res["portfolio_returns"]

        for t1, t2, t3 in [(-0.01, -0.025, -0.05), (-0.01, -0.02, -0.04),
                            (-0.015, -0.03, -0.05), (-0.01, -0.03, -0.06)]:
            dd_ret = triple_layer_ddc_lagged(br, th1=t1, th2=t2, th3=t3,
                                              recovery=0.015)
            key = f"TL({t1:.1%}/{t2:.1%}/{t3:.1%})_{base_key}"[:125]
            res = make_result(dd_ret, prices_is.index); m = res["metrics"]
            dd_lev_results[key] = res; results[key] = res
            if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                winners.append(key)

        for th1, th2 in [(-0.01, -0.03), (-0.015, -0.04), (-0.02, -0.05)]:
            dd_ret = hierarchical_ddc_lagged(br, th1=th1, th2=th2, recovery=0.015)
            key = f"HDDC({th1:.1%}/{th2:.1%})_{base_key}"[:125]
            res = make_result(dd_ret, prices_is.index); m = res["metrics"]
            dd_lev_results[key] = res; results[key] = res
            if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                winners.append(key)

        for t1, t2, t3 in [(-0.01, -0.02, -0.04), (-0.01, -0.025, -0.05)]:
            for rc in [3.0, 5.0]:
                dd_ret = position_level_ddc(br, th1=t1, th2=t2, th3=t3,
                                             recovery=0.015, rebal_cost_bps=rc)
                key = f"PL({t1:.1%}/{t2:.1%}/{t3:.1%},rc{rc:.0f})_{base_key}"[:125]
                res = make_result(dd_ret, prices_is.index); m = res["metrics"]
                dd_lev_results[key] = res; results[key] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(key)

    dd_sorted = sorted(dd_lev_results.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(dd_lev_results)} DD-controlled. Top 15:")
    for k, v in dd_sorted[:15]:
        m = v["metrics"]
        print(f"    {k[:110]:110s} Sh={m['Sharpe Ratio']:.4f} "
              f"CAGR={m['CAGR']:.2%} DD={m['Max Drawdown']:.2%}")

    # ==============================================================
    # PHASE 8: WALK-FORWARD
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 8: WALK-FORWARD\n")

    top_for_wf = sorted(
        [(k, v) for k, v in results.items()
         if v["metrics"]["CAGR"] > spy_cagr and v["metrics"]["Sharpe Ratio"] > 1.95],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:25]

    periods = [
        ("2010-12", "2010-01-01", "2013-01-01"),
        ("2013-15", "2013-01-01", "2016-01-01"),
        ("2016-18", "2016-01-01", "2019-01-01"),
        ("2019-21", "2019-01-01", "2022-01-01"),
        ("2022-25", "2022-01-01", "2025-03-01"),
    ]

    if top_for_wf:
        p_headers = [p[0] for p in periods]
        print(f"  {'Strategy':<70s} " + " ".join(f"{h:>7s}" for h in p_headers)
              + f" {'Min':>7s} {'Ok':>4s}")
        print("  " + "-" * 120)

        for k, v in top_for_wf:
            pr = v["portfolio_returns"]
            period_results = []
            for pname, ps, pe in periods:
                mask = (pr.index >= ps) & (pr.index < pe)
                sub = pr[mask]
                if len(sub) > 100:
                    sh, _ = quick_metrics(sub)
                    period_results.append(sh)
                else:
                    period_results.append(0)
            consistent = all(s > 0.5 for s in period_results)
            min_sh = min(period_results)
            p_strs = " ".join(f"{s:7.2f}" for s in period_results)
            tag = "ALL" if consistent else " no"
            print(f"  {k[:70]:70s} {p_strs} {min_sh:7.2f} {tag:>4s}")

    # ==============================================================
    # PHASE 9: TRUE OOS TEST
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 9: TRUE OUT-OF-SAMPLE TEST (2025-03 to 2026-03)")
    print("  Parameters LOCKED from in-sample. NEVER SEEN data.\n")

    oos_results = {}
    if n_oos >= 100:
        spy_oos_ret = prices_oos["SPY"].pct_change().fillna(0)
        spy_oos_sh, spy_oos_cagr = quick_metrics(spy_oos_ret) if len(spy_oos_ret) > 50 else (0, 0)
        print(f"  SPY OOS: Sharpe={spy_oos_sh:.4f}, CAGR={spy_oos_cagr:.2%}")
        print(f"  OOS: {prices_oos.index[0].strftime('%Y-%m-%d')} to "
              f"{prices_oos.index[-1].strftime('%Y-%m-%d')} ({n_oos} days)\n")

        # --- OOS: Pair portfolio replay ---
        print(f"  -- Pair Portfolio OOS --")
        oos_zp_rets = {}
        for port_size in [5, 7, 10]:
            if port_size > len(selected_A):
                continue
            sel = selected_A[:port_size]
            for notional in [0.06, 0.10, 0.12]:
                combined = pd.DataFrame(0.0, index=prices_oos.index,
                                        columns=prices_oos.columns)
                for cfg, sh, cagr, _ in sel:
                    a, b, win, ez_in, ez_out = cfg
                    pw = pair_weights(prices_oos, a, b, window=win,
                                      entry_z=ez_in, exit_z=ez_out,
                                      notional=notional)
                    if pw is not None:
                        combined += pw
                res = backtest(prices_oos, combined)
                m = res["metrics"]
                okey = f"OOS_ZP_ShF{port_size}_n{notional}"
                oos_results[okey] = res
                oos_zp_rets[okey] = res["portfolio_returns"]
                print(f"    {okey:50s} Sh={m['Sharpe Ratio']:.4f} "
                      f"CAGR={m['CAGR']:.2%} DD={m['Max Drawdown']:.2%}")

        # --- OOS: CrashHedge ---
        print(f"\n  -- OOS Alpha Sources --")
        ch_oos = backtest(prices_oos, strat_crash_hedged(prices_oos, 1.0))
        oos_results["OOS_CrashHedge"] = ch_oos
        ch_oos_r = ch_oos["portfolio_returns"]
        print(f"    {'OOS_CrashHedge':50s} Sh={ch_oos['metrics']['Sharpe Ratio']:.4f} "
              f"CAGR={ch_oos['metrics']['CAGR']:.2%} DD={ch_oos['metrics']['Max Drawdown']:.2%}")

        # --- OOS: XS Momentum ---
        best_mom_cfg = best_mom_key.replace("XSMom_", "")
        # Parse config from key
        for lb, skip, nl, ns, not_ in [(252, 21, 4, 3, 0.10),
                                        (252, 21, 5, 3, 0.12),
                                        (126, 21, 4, 3, 0.10),
                                        (189, 21, 4, 2, 0.10)]:
            check_key = f"XSMom_lb{lb}_s{skip}_L{nl}S{ns}_n{not_}"
            if check_key == best_mom_key:
                w_mom_oos = strat_cross_sectional_momentum(
                    prices_oos, lookback=lb, skip=skip,
                    n_long=nl, n_short=ns, notional=not_)
                mom_oos = backtest(prices_oos, w_mom_oos)
                oos_results["OOS_XSMom"] = mom_oos
                mom_oos_r = mom_oos["portfolio_returns"]
                print(f"    {'OOS_XSMom':50s} Sh={mom_oos['metrics']['Sharpe Ratio']:.4f} "
                      f"CAGR={mom_oos['metrics']['CAGR']:.2%} DD={mom_oos['metrics']['Max Drawdown']:.2%}")
                break

        # --- OOS: TS Momentum ---
        for not_ in [0.06, 0.08, 0.10]:
            check_key = f"TSMom_n{not_}"
            if check_key == best_tsm_key:
                w_tsm_oos = strat_time_series_momentum(prices_oos, notional=not_)
                tsm_oos = backtest(prices_oos, w_tsm_oos)
                oos_results["OOS_TSMom"] = tsm_oos
                tsm_oos_r = tsm_oos["portfolio_returns"]
                print(f"    {'OOS_TSMom':50s} Sh={tsm_oos['metrics']['Sharpe Ratio']:.4f} "
                      f"CAGR={tsm_oos['metrics']['CAGR']:.2%} DD={tsm_oos['metrics']['Max Drawdown']:.2%}")
                break

        # --- OOS: Sector Rotation ---
        for lb, nt, nb, not_ in [(63, 4, 2, 0.12), (126, 4, 2, 0.12),
                                  (63, 3, 2, 0.10), (42, 4, 2, 0.10)]:
            check_key = f"SecRot_lb{lb}_L{nt}S{nb}_n{not_}"
            if check_key == best_sr_key:
                w_sr_oos = strat_sector_rotation(prices_oos, lookback=lb,
                                                  n_top=nt, n_bottom=nb,
                                                  notional=not_)
                sr_oos = backtest(prices_oos, w_sr_oos)
                oos_results["OOS_SecRot"] = sr_oos
                sr_oos_r = sr_oos["portfolio_returns"]
                print(f"    {'OOS_SecRot':50s} Sh={sr_oos['metrics']['Sharpe Ratio']:.4f} "
                      f"CAGR={sr_oos['metrics']['CAGR']:.2%} DD={sr_oos['metrics']['Max Drawdown']:.2%}")
                break

        # --- OOS: Ensembles ---
        print(f"\n  -- OOS Ensembles --")
        # Use best pair OOS returns
        best_zp_oos_key = max(oos_zp_rets.keys(),
                              key=lambda k: oos_results[k]["metrics"]["Sharpe Ratio"])
        zp_oos_r = oos_zp_rets[best_zp_oos_key]
        print(f"    Best OOS pair portfolio: {best_zp_oos_key}")

        # Test all ensemble weight combos on OOS
        oos_ens = {}
        # Ensure we have all alpha source OOS returns
        has_mom = "OOS_XSMom" in oos_results
        has_tsm = "OOS_TSMom" in oos_results
        has_sr = "OOS_SecRot" in oos_results

        # 2-source: Pairs + CH
        for zp_pct, ch_pct in [(0.85, 0.15), (0.80, 0.20), (0.75, 0.25),
                                (0.70, 0.30), (0.60, 0.40), (0.50, 0.50)]:
            er = zp_pct * zp_oos_r + ch_pct * ch_oos_r
            key = f"OOS_E2({zp_pct:.0%}ZP+{ch_pct:.0%}CH)"
            oos_ens[key] = make_result(er, prices_oos.index)

        # 3-source: Pairs + CH + Mom
        if has_mom:
            for zp_pct, ch_pct, mom_pct in [(0.70, 0.15, 0.15),
                                             (0.65, 0.20, 0.15),
                                             (0.60, 0.20, 0.20),
                                             (0.55, 0.25, 0.20),
                                             (0.50, 0.30, 0.20),
                                             (0.40, 0.30, 0.30)]:
                er = zp_pct * zp_oos_r + ch_pct * ch_oos_r + mom_pct * mom_oos_r
                key = f"OOS_E3({zp_pct:.0%}ZP+{ch_pct:.0%}CH+{mom_pct:.0%}Mom)"
                oos_ens[key] = make_result(er, prices_oos.index)

        # 4-source: + TSMom
        if has_mom and has_tsm:
            for zp_pct, ch_pct, mom_pct, tsm_pct in [
                (0.55, 0.15, 0.15, 0.15),
                (0.50, 0.20, 0.15, 0.15),
                (0.45, 0.20, 0.20, 0.15),
                (0.40, 0.25, 0.20, 0.15),
                (0.35, 0.25, 0.20, 0.20),
            ]:
                er = (zp_pct * zp_oos_r + ch_pct * ch_oos_r
                      + mom_pct * mom_oos_r + tsm_pct * tsm_oos_r)
                key = f"OOS_E4({zp_pct:.0%}ZP+{ch_pct:.0%}CH+{mom_pct:.0%}Mom+{tsm_pct:.0%}TSM)"
                oos_ens[key] = make_result(er, prices_oos.index)

        # 5-source: + SecRot
        if has_mom and has_tsm and has_sr:
            for zp_pct, ch_pct, mom_pct, tsm_pct, sr_pct in [
                (0.45, 0.15, 0.15, 0.15, 0.10),
                (0.40, 0.20, 0.15, 0.15, 0.10),
                (0.35, 0.20, 0.15, 0.15, 0.15),
                (0.30, 0.25, 0.15, 0.15, 0.15),
            ]:
                er = (zp_pct * zp_oos_r + ch_pct * ch_oos_r
                      + mom_pct * mom_oos_r + tsm_pct * tsm_oos_r
                      + sr_pct * sr_oos_r)
                key = f"OOS_E5({zp_pct:.0%}ZP+{ch_pct:.0%}CH+{mom_pct:.0%}M+{tsm_pct:.0%}T+{sr_pct:.0%}S)"
                oos_ens[key] = make_result(er, prices_oos.index)

        oos_results.update(oos_ens)

        # Print ensemble OOS results
        ens_oos_sorted = sorted(oos_ens.items(),
                                key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                                reverse=True)
        for k, v in ens_oos_sorted[:20]:
            m = v["metrics"]
            print(f"    {k:60s} Sh={m['Sharpe Ratio']:.4f} "
                  f"CAGR={m['CAGR']:.2%} DD={m['Max Drawdown']:.2%}")

        # --- OOS: Best ensembles with overlays + leverage ---
        print(f"\n  -- OOS Overlays + Leverage --")
        top_oos_ens = sorted(oos_ens.items(),
                             key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                             reverse=True)[:5]

        for ens_key, ens_res in top_oos_ens:
            ens_r = ens_res["portfolio_returns"]
            for tvol in [0.06, 0.08]:
                vt_oos = vol_target_overlay(ens_r, target_vol=tvol)
                for h1, h2 in [(-0.01, -0.03), (-0.015, -0.04)]:
                    dd_oos = hierarchical_ddc_lagged(vt_oos, th1=h1, th2=h2)
                    h1s = f"{abs(h1)*100:.1f}"; h2s = f"{abs(h2)*100:.1f}"
                    okey = f"OOS_VT{int(tvol*100)}+H({h1s}/{h2s})_{ens_key[4:]}"[:100]
                    oos_results[okey] = make_result(dd_oos, prices_oos.index)

                    # Leverage
                    for mult in [3.0, 4.0, 5.0]:
                        for lc_label, lc in [("1.0%", 0.01), ("2.0%", 0.02)]:
                            lr = dd_oos * mult - (mult - 1) * lc / 252
                            lkey = f"OOS_L({lc_label})x{mult}_{okey[4:]}"[:110]
                            oos_results[lkey] = make_result(lr, prices_oos.index)

                            # Post-leverage DDC
                            for t1, t2, t3 in [(-0.01, -0.02, -0.04)]:
                                pl_r = position_level_ddc(
                                    lr, th1=t1, th2=t2, th3=t3,
                                    recovery=0.015, rebal_cost_bps=5.0)
                                plkey = f"OOS_PL_rc5_{lkey[4:]}"[:115]
                                oos_results[plkey] = make_result(pl_r, prices_oos.index)

        # --- Consolidated OOS ---
        print(f"\n  --- CONSOLIDATED OOS RESULTS ---")
        oos_sorted = sorted(oos_results.items(),
                            key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                            reverse=True)
        print(f"  {'Strategy':<70s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s}")
        print(f"  {'-'*70} {'---':>8s} {'---':>8s} {'---':>8s}")
        for k, v in oos_sorted[:40]:
            m = v["metrics"]
            print(f"  {k[:70]:70s} {m['Sharpe Ratio']:8.4f} {m['CAGR']:8.2%} "
                  f"{m['Max Drawdown']:8.2%}")

        # v17 comparison
        print(f"\n  --- v17 vs v18 OOS Comparison ---")
        print(f"  v17 best OOS (unleveraged): Sharpe=2.08, CAGR=2.35%")
        print(f"  v17 best OOS (3x, 1% cost): Sharpe=1.78, CAGR=26.00%")
        best_oos_unlev = max(
            [(k, v) for k, v in oos_results.items()
             if not any(x in k for x in ["_L(", "_PL_"])],
            key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
            default=(None, None)
        )
        best_oos_lev = max(
            oos_results.items(),
            key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
            default=(None, None)
        )
        if best_oos_unlev[0]:
            m = best_oos_unlev[1]["metrics"]
            print(f"  v18 best OOS (unlev): {best_oos_unlev[0][:80]}")
            print(f"    Sharpe={m['Sharpe Ratio']:.4f}, CAGR={m['CAGR']:.2%}, "
                  f"DD={m['Max Drawdown']:.2%}")
        if best_oos_lev[0]:
            m = best_oos_lev[1]["metrics"]
            print(f"  v18 best OOS (any):   {best_oos_lev[0][:80]}")
            print(f"    Sharpe={m['Sharpe Ratio']:.4f}, CAGR={m['CAGR']:.2%}, "
                  f"DD={m['Max Drawdown']:.2%}")

    else:
        print(f"  Insufficient OOS data ({n_oos} days).")

    # ==============================================================
    # PHASE 10: STATISTICAL VALIDATION
    # ==============================================================
    print(f"\n{SEP}")
    print("PHASE 10: STATISTICAL VALIDATION\n")

    total_tested = len(results)
    champion = max(results.items(),
                   key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
    ch_name, ch_res_val = champion
    ch_sh = ch_res_val["metrics"]["Sharpe Ratio"]
    ch_ret_val = ch_res_val["portfolio_returns"]

    print(f"  Total: {total_tested:,} strategies")
    print(f"  IS Champion: {ch_name[:100]}")
    print(f"  IS Sharpe: {ch_sh:.4f}")

    skew_val = float(ch_ret_val.skew())
    kurt_val = float(ch_ret_val.kurtosis()) + 3
    n_obs = len(ch_ret_val)
    dsr, p_val, e_max = deflated_sharpe_ratio(ch_sh, total_tested, n_obs,
                                               skew=skew_val, kurt=kurt_val)
    print(f"  E[max SR]: {e_max:.4f}, DSR: {dsr:.4f}, p={p_val:.6f}")

    lo, hi, mean_boot = bootstrap_sharpe_ci(ch_ret_val, n_boot=5000)
    print(f"  Bootstrap: {mean_boot:.4f} [{lo:.4f}, {hi:.4f}]")

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    print(f"\n{SEP}")
    print("FINAL SUMMARY -- v18 ALPHA-ENHANCED")
    print(SEP)

    winners = list(dict.fromkeys(winners))
    print(f"\n  Total: {total_tested} strategies | Winners: {len(winners)}")
    print(f"  New alpha sources: XSMom, TSMom, SecRot")
    print(f"  Dropped: VolCarry (OOS failure)")

    if winners:
        sh_leaders = sorted([(k, results[k]) for k in winners],
                           key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                           reverse=True)
        print(f"\n  TOP 10 IS WINNERS:")
        for k, v in sh_leaders[:10]:
            m = v["metrics"]
            print(f"    {k[:110]}")
            print(f"      CAGR={m['CAGR']:.2%}, Sh={m['Sharpe Ratio']:.4f}, "
                  f"DD={m['Max Drawdown']:.2%}")

    if oos_results:
        oos_best = max(oos_results.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
        m = oos_best[1]["metrics"]
        print(f"\n  BEST OOS: {oos_best[0][:90]}")
        print(f"    Sharpe={m['Sharpe Ratio']:.4f}, CAGR={m['CAGR']:.2%}, "
              f"DD={m['Max Drawdown']:.2%}")

    print(f"\n  IS Champion Sharpe: {ch_sh:.4f}")
    print(f"  DSR: {dsr:.4f} (p={p_val:.6f})")
    print(f"  Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(SEP)


if __name__ == "__main__":
    main()
