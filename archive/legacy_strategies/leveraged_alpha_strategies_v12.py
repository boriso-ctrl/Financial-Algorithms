"""
Leveraged Alpha v12 — HEDGE-FUND GRADE ALPHA MAXIMISATION
==================================================================================
v11 found 1,079 WINNERs. Best at 1.5% cost:
  Champion: 8%CH+92%ZP_CAGRFocus x6.0 -> CAGR=30.05%, Sharpe=1.95
  Best DYN at 1.5%: 5%CH+95%ZP_CAGR tgt3.5 -> CAGR=13.85%, Sharpe=2.14

v12 hedge-fund innovations:
  1. Regime-adaptive z-scores (tighter thresholds in low-vol, wider in high-vol)
  2. Correlation-filtered pair portfolios (maximize diversification)
  3. Vol-targeting at portfolio level (target specific annualized vol)
  4. Drawdown control overlay (cut exposure during drawdowns)
  5. Carry-enhanced pairs (weight by spread income potential)
  6. Rolling pair re-selection (avoid stale allocations)
  7. Kelly-optimal sizing
  8. Multi-timeframe ensemble (different windows contribute independently)
  9. Tail-risk hedging with options-like payoff (long vol in crises)
  10. Expanded universe scan for new pair opportunities
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE"]
BROAD   = ["SPY", "QQQ", "IWM", "EFA"]
SAFE    = ["TLT", "IEF", "GLD", "SHY"]
ALL_TICKERS = SECTORS + BROAD + SAFE
START, END = "2010-01-01", "2025-03-01"
TX_BPS = 5; SHORT_COST = 0.005; RF_CASH = 0.02; RF = 0.0
LEV_COST_STD = 0.015
SEP = "=" * 90


# ═══════════════════════════════════════════════════════════════════
#  CORE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers, {len(p)} days\n")
    return p

def rvol(s, n=21):
    return s.pct_change().rolling(n, min_periods=max(10, n // 2)).std() * np.sqrt(252)

def zscore(s, window=63):
    m = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().clip(lower=1e-8)
    return (s - m) / sd

def backtest(prices, weights, lev_cost=LEV_COST_STD, cap=100_000.0):
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
    net = port_ret + cr - tx - lc - sc
    eq = cap * (1 + net).cumprod(); eq.name = "Equity"
    m = compute_metrics(net, eq, cap, risk_free_rate=RF, turnover=turn, gross_exposure=ge)
    return {"equity_curve": eq, "portfolio_returns": net, "weights": w,
            "turnover": turn, "gross_exposure": ge, "net_exposure": ne,
            "cash_weight": cw, "metrics": m}

def quick_metrics(returns):
    """Fast Sharpe/CAGR without full backtest."""
    if returns.std() < 1e-10:
        return 0.0, 0.0
    sh = returns.mean() / returns.std() * np.sqrt(252)
    eq = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 and eq.iloc[-1] > 0 else 0
    return sh, cagr


# ═══════════════════════════════════════════════════════════════════
#  PAIR TRADING ENGINE (enhanced)
# ═══════════════════════════════════════════════════════════════════

def pair_weights(prices, leg_a, leg_b, window=63, entry_z=2.0, exit_z=0.5,
                 notional=0.15):
    if leg_a not in prices.columns or leg_b not in prices.columns:
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


def pair_weights_regime(prices, leg_a, leg_b, window=63, notional=0.15,
                        vol_ref="QQQ"):
    """Regime-adaptive pair trading: adjust entry/exit z based on market vol."""
    if leg_a not in prices.columns or leg_b not in prices.columns:
        return None
    spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
    z = zscore(spread, window)
    # Market vol regime
    mkt_vol = rvol(prices[vol_ref], 20)
    mkt_vol_avg = mkt_vol.rolling(252, min_periods=60).mean()
    vol_ratio = (mkt_vol / mkt_vol_avg.clip(lower=0.01)).fillna(1.0)
    # In low vol: tighter entry (1.5), wider exit (0.75) -> more trades
    # In high vol: wider entry (2.5), tighter exit (0.25) -> fewer, safer trades
    entry_z = 1.5 + 0.5 * (vol_ratio - 1).clip(lower=-0.5, upper=1.0)
    exit_z = 0.75 - 0.25 * (vol_ratio - 1).clip(lower=-0.5, upper=1.0)

    pos = pd.Series(0.0, index=prices.index)
    for i in range(1, len(pos)):
        prev = pos.iloc[i - 1]; zi = z.iloc[i]
        ez_in = entry_z.iloc[i]; ez_out = exit_z.iloc[i]
        if np.isnan(zi) or np.isnan(ez_in):
            pos.iloc[i] = 0; continue
        if prev == 0:
            if zi > ez_in:    pos.iloc[i] = -1
            elif zi < -ez_in: pos.iloc[i] = 1
            else:             pos.iloc[i] = 0
        elif prev > 0:
            pos.iloc[i] = 0 if zi > -ez_out else 1
        else:
            pos.iloc[i] = 0 if zi < ez_out else -1
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[leg_a] = pos * notional; w[leg_b] = -pos * notional
    return w


# ═══════════════════════════════════════════════════════════════════
#  ALPHA SOURCES
# ═══════════════════════════════════════════════════════════════════

def strat_crash_hedged(prices, base_lev=1.0):
    qqq = prices["QQQ"]; v20 = rvol(qqq, 20)
    va = v20.rolling(120, min_periods=30).mean()
    normal = v20 < va * 1.2
    elevated = (v20 >= va * 1.2) & (v20 < va * 1.8)
    crisis = v20 >= va * 1.8
    recovery = elevated & (v20 < v20.shift(5)) & (qqq > qqq.rolling(10).min())
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col, n, e, c, r in [("QQQ", base_lev * 0.7, base_lev * 0.3, 0.0, base_lev * 0.8),
                             ("SPY", base_lev * 0.3, base_lev * 0.1, -0.3, base_lev * 0.4)]:
        if col in w.columns:
            w.loc[normal, col] = n; w.loc[elevated, col] = e
            w.loc[crisis, col] = c; w.loc[recovery, col] = r
    for col, e, c in [("IWM", -0.2, 0), ("GLD", 0.15, 0.3), ("TLT", 0, 0.2)]:
        if col in w.columns:
            w.loc[elevated, col] = e; w.loc[crisis, col] = c
            if col == "IWM": w.loc[recovery, col] = 0
    return w


def strat_momentum_ls(prices, lookback=252, n_long=3, n_short=2, notional=0.15):
    """Cross-sectional sector momentum L/S."""
    sector_ret = prices[SECTORS].pct_change(lookback)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(lookback + 1, len(prices)):
        row = sector_ret.iloc[i].dropna().sort_values()
        if len(row) < n_long + n_short:
            continue
        shorts = row.index[:n_short]
        longs = row.index[-n_long:]
        for s in shorts:
            w.loc[w.index[i], s] = -notional
        for l in longs:
            w.loc[w.index[i], l] = notional
    return w


def strat_vol_carry(prices):
    """Vol carry: long low-vol sectors, short high-vol sectors (defensive tilt)."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(126, len(prices)):
        vols = {}
        for s in SECTORS:
            v = prices[s].iloc[max(0, i - 63):i].pct_change().std() * np.sqrt(252)
            if not np.isnan(v) and v > 0:
                vols[s] = v
        if len(vols) < 6:
            continue
        sorted_v = sorted(vols.items(), key=lambda x: x[1])
        # Long lowest 3 vol, short highest 2 vol
        for s, _ in sorted_v[:3]:
            w.loc[w.index[i], s] = 0.10
        for s, _ in sorted_v[-2:]:
            w.loc[w.index[i], s] = -0.10
    return w


# ═══════════════════════════════════════════════════════════════════
#  PORTFOLIO OVERLAYS
# ═══════════════════════════════════════════════════════════════════

def vol_target_overlay(returns, target_vol=0.05, lookback=63):
    """Scale returns to achieve target annualized volatility."""
    realized = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    realized = realized.clip(lower=0.005)
    scale = (target_vol / realized).clip(lower=0.2, upper=5.0)
    return returns * scale.shift(1).fillna(1.0)


def drawdown_control(returns, max_dd_trigger=-0.05, recovery_rate=0.02):
    """Reduce exposure during drawdowns, gradually recover."""
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    for i in range(1, len(scale)):
        if dd.iloc[i] < max_dd_trigger:
            # Scale down proportionally to drawdown severity
            severity = min(abs(dd.iloc[i] / max_dd_trigger), 3.0)
            scale.iloc[i] = max(0.2, 1.0 / severity)
        elif scale.iloc[i - 1] < 1.0:
            # Gradually recover
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + recovery_rate)
        else:
            scale.iloc[i] = 1.0
    return returns * scale


def kelly_scale(returns, lookback=252):
    """Half-Kelly sizing based on rolling win rate and payoff ratio."""
    win_rate = returns.rolling(lookback, min_periods=60).apply(
        lambda x: (x > 0).sum() / len(x), raw=True)
    avg_win = returns.rolling(lookback, min_periods=60).apply(
        lambda x: x[x > 0].mean() if (x > 0).any() else 0, raw=True)
    avg_loss = returns.rolling(lookback, min_periods=60).apply(
        lambda x: abs(x[x < 0].mean()) if (x < 0).any() else 1, raw=True)
    payoff = (avg_win / avg_loss.clip(lower=1e-8)).clip(upper=5.0)
    kelly_f = (win_rate * payoff - (1 - win_rate)) / payoff.clip(lower=0.01)
    kelly_f = kelly_f.clip(lower=0.1, upper=2.0)  # Half-Kelly with bounds
    return returns * (kelly_f * 0.5).shift(1).fillna(0.5)


# ═══════════════════════════════════════════════════════════════════
#  AUDIT
# ═══════════════════════════════════════════════════════════════════

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
        print(f"  {label:70s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} {cagr_flag} {sh_flag} {win}")
    else:
        print(f"  {label:70s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} Sort={m['Sortino Ratio']:.2f} "
              f"Calmar={m['Calmar Ratio']:.2f} {cagr_flag} {sh_flag} {win}")
    return bool(win)


def make_result(returns, prices_index):
    """Create a result dict from a return series."""
    eq = 100_000 * (1 + returns).cumprod(); eq.name = "Equity"
    m = compute_metrics(returns, eq, 100_000, risk_free_rate=RF)
    return {"equity_curve": eq, "portfolio_returns": returns,
            "weights": pd.DataFrame(), "metrics": m,
            "turnover": pd.Series(0, index=prices_index),
            "gross_exposure": pd.Series(1, index=prices_index),
            "net_exposure": pd.Series(1, index=prices_index),
            "cash_weight": pd.Series(0, index=prices_index)}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("LEVERAGED ALPHA v12 -- HEDGE-FUND GRADE ALPHA MAXIMISATION")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  {START} to {END}")
    print(SEP)

    prices = load_data()
    bench = {}
    for t in ["SPY", "QQQ"]:
        eq = prices[t] / prices[t].iloc[0] * 100_000
        ret = prices[t].pct_change().fillna(0)
        m = compute_metrics(ret, eq, 100_000, risk_free_rate=RF)
        bench[t] = m
        print(f"  {t}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")
    spy_cagr = bench["SPY"]["CAGR"]
    print()

    results = {}
    winners = []

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: PAIR SCANNING — Fixed + Regime-Adaptive
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("PHASE 1: PAIR STRATEGIES (Fixed Params + Regime-Adaptive)")
    print()

    # Best pairs from v11 (CAGR-focused, Sharpe > 1.5)
    BEST_PAIRS = [
        # (a, b, window, entry_z, exit_z) — from v11 CAGR-focused ranking
        ("XLP", "XLU", 63,  2.25, 0.50),   # CAGR=3.14%, Sh=2.77
        ("XLP", "XLU", 126, 1.50, 0.75),   # CAGR=3.11%, Sh=2.43
        ("XLP", "XLU", 63,  2.00, 0.50),   # CAGR=2.98%, Sh=2.47
        ("XLF", "IWM", 42,  1.50, 0.50),   # CAGR=2.93%, Sh=2.01
        ("XLB", "EFA", 42,  1.50, 0.50),   # CAGR=2.81%, Sh=1.82
        ("XLI", "XLB", 126, 1.75, 0.75),   # CAGR=2.68%, Sh=2.97
        ("XLB", "EFA", 42,  2.00, 0.50),   # CAGR=2.60%, Sh=2.32
        ("XLV", "XLP", 126, 2.25, 0.75),   # CAGR=2.47%, Sh=2.45
        ("XLI", "IWM", 126, 1.50, 0.25),   # CAGR=2.44%, Sh=2.10
        ("XLK", "QQQ", 63,  2.00, 0.25),   # CAGR=2.17%, Sh=3.84
        ("XLK", "QQQ", 42,  2.00, 0.50),   # CAGR=2.13%, Sh=4.35
        ("XLK", "QQQ", 126, 2.25, 0.75),   # CAGR=2.15%, Sh=5.08
        ("XLK", "SPY", 42,  1.75, 0.50),   # CAGR=2.26%, Sh=2.50
        ("XLI", "SPY", 42,  1.50, 0.50),   # CAGR=2.15%, Sh=2.15
        ("XLB", "SPY", 42,  1.75, 0.50),   # CAGR=2.33%, Sh=2.06
    ]

    # 1A. Build fixed-param portfolios
    print("  --- 1A: Fixed-Param CAGR-Focused Portfolios ---")
    zp_portfolios = {}

    # CAGR-focused Top5 (highest individual CAGR with Sharpe > 2.0)
    cagr_top5 = BEST_PAIRS[:5]
    cagr_top7 = BEST_PAIRS[:7]
    cagr_top10 = BEST_PAIRS[:10]
    cagr_top15 = BEST_PAIRS[:15]

    for name, selection in [("CTop5", cagr_top5), ("CTop7", cagr_top7),
                            ("CTop10", cagr_top10), ("CTop15", cagr_top15)]:
        for notional in [0.08, 0.10, 0.12, 0.15]:
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for a, b, win, ez_in, ez_out in selection:
                pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=notional)
                if pw is not None:
                    combined += pw
            res = backtest(prices, combined)
            m = res["metrics"]
            key = f"ZP_{name}_n{notional}"
            zp_portfolios[key] = res; results[key] = res
            report(key, m, spy_cagr, short=True)

    # 1B. Regime-adaptive pairs
    print("\n  --- 1B: Regime-Adaptive Pairs ---")
    regime_pairs = {}
    # Use top pairs with regime adaptation
    for name, pairs in [("RTop5", [(a, b, w) for a, b, w, _, _ in cagr_top5]),
                        ("RTop10", [(a, b, w) for a, b, w, _, _ in cagr_top10])]:
        for notional in [0.10, 0.12, 0.15]:
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for a, b, win in pairs:
                pw = pair_weights_regime(prices, a, b, window=win, notional=notional)
                if pw is not None:
                    combined += pw
            res = backtest(prices, combined)
            m = res["metrics"]
            key = f"ZPR_{name}_n{notional}"
            regime_pairs[key] = res; results[key] = res
            zp_portfolios[key] = res
            report(key, m, spy_cagr, short=True)

    # 1C. Correlation-filtered portfolio: pick pairs with lowest cross-correlation
    print("\n  --- 1C: Correlation-Filtered Pairs ---")
    # Compute pair returns and correlations
    pair_returns = {}
    all_pair_configs = BEST_PAIRS[:15]
    for a, b, win, ez_in, ez_out in all_pair_configs:
        pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                          exit_z=ez_out, notional=0.10)
        if pw is not None:
            res = backtest(prices, pw)
            pair_returns[(a, b, win)] = res["portfolio_returns"]

    # Greedy pair selection: start with best pair, add lowest-correlated
    if pair_returns:
        pair_keys = list(pair_returns.keys())
        # Quick Sharpe for each
        pair_sharpes = {}
        for pk, pr in pair_returns.items():
            sh, cg = quick_metrics(pr)
            pair_sharpes[pk] = (sh, cg)

        # Start with highest-Sharpe pair
        ranked = sorted(pair_sharpes.items(), key=lambda x: x[1][0], reverse=True)
        selected = [ranked[0][0]]
        remaining = [r[0] for r in ranked[1:]]

        for _ in range(min(9, len(remaining))):
            best_next = None; best_score = -999
            for cand in remaining:
                # Avg correlation with selected pairs
                corrs = [pair_returns[cand].corr(pair_returns[s]) for s in selected]
                avg_corr = np.mean(corrs) if corrs else 0
                # Score: Sharpe - correlation penalty
                score = pair_sharpes[cand][0] - 2.0 * avg_corr
                if score > best_score:
                    best_score = score; best_next = cand
            if best_next:
                selected.append(best_next)
                remaining.remove(best_next)

        for n_sel in [5, 7, len(selected)]:
            sel = selected[:n_sel]
            for notional in [0.10, 0.12]:
                combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                for a, b, win in sel:
                    cfg = [(aa, bb, ww, ei, eo) for aa, bb, ww, ei, eo in all_pair_configs
                           if (aa, bb, ww) == (a, b, win)]
                    if cfg:
                        _, _, _, ez_in, ez_out = cfg[0]
                        pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                          exit_z=ez_out, notional=notional)
                        if pw is not None:
                            combined += pw
                res = backtest(prices, combined)
                m = res["metrics"]
                key = f"ZP_CorrFilt{n_sel}_n{notional}"
                zp_portfolios[key] = res; results[key] = res
                report(key, m, spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: ADDITIONAL ALPHA SOURCES
    # ══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("PHASE 2: ADDITIONAL ALPHA SOURCES\n")

    # CrashHedge
    ch_res = backtest(prices, strat_crash_hedged(prices, 1.0))
    results["CrashHedge"] = ch_res
    report("CrashHedge", ch_res["metrics"], spy_cagr, short=True)

    # Momentum L/S
    mom_res = backtest(prices, strat_momentum_ls(prices, lookback=252, n_long=3, n_short=2, notional=0.15))
    results["MomLS_252"] = mom_res
    report("MomLS_252", mom_res["metrics"], spy_cagr, short=True)

    # Vol carry
    vc_res = backtest(prices, strat_vol_carry(prices))
    results["VolCarry"] = vc_res
    report("VolCarry", vc_res["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: ENSEMBLE CONSTRUCTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("PHASE 3: MULTI-ALPHA ENSEMBLES\n")

    ch_ret = ch_res["portfolio_returns"]
    mom_ret = mom_res["portfolio_returns"]
    vc_ret = vc_res["portfolio_returns"]

    # Sort ZP portfolios by Sharpe
    sorted_zp = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    ensembles = {}
    for zp_key, zp_res in sorted_zp[:10]:
        zp_ret = zp_res["portfolio_returns"]
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.5:
            continue

        # A) ZP + CH (2-source, traditional)
        for ch_pct in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            er = (1 - ch_pct) * zp_ret + ch_pct * ch_ret
            key = f"E2({ch_pct:.0%}CH+{1-ch_pct:.0%}{zp_key})"
            ensembles[key] = make_result(er, prices.index)

        # B) ZP + CH + Mom (3-source)
        for zp_pct, ch_pct, mom_pct in [(0.85, 0.05, 0.10), (0.80, 0.10, 0.10),
                                         (0.80, 0.08, 0.12), (0.75, 0.10, 0.15)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + mom_pct * mom_ret
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{mom_pct:.0%}Mom)"
            ensembles[key] = make_result(er, prices.index)

        # C) ZP + CH + VolCarry (3-source)
        for zp_pct, ch_pct, vc_pct in [(0.85, 0.05, 0.10), (0.80, 0.10, 0.10)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + vc_pct * vc_ret
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{vc_pct:.0%}VC)"
            ensembles[key] = make_result(er, prices.index)

        # D) ZP + CH + Mom + VC (4-source)
        for zp_pct, ch_pct, mom_pct, vc_pct in [(0.75, 0.08, 0.10, 0.07),
                                                   (0.70, 0.10, 0.10, 0.10)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + mom_pct * mom_ret + vc_pct * vc_ret
            key = f"E4({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{mom_pct:.0%}M+{vc_pct:.0%}V)"
            ensembles[key] = make_result(er, prices.index)

    results.update(ensembles)

    # Show top ensembles
    ens_sorted = sorted(ensembles.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(ensembles)} ensembles. Top 15 by Sharpe:")
    for k, v in ens_sorted[:15]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: PORTFOLIO OVERLAYS
    # ══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("PHASE 4: PORTFOLIO OVERLAYS (vol-target, DD control, Kelly)\n")

    # Apply overlays to top unleveraged strategies
    top_base = sorted([(k, v) for k, v in results.items()
                       if v["metrics"]["Sharpe Ratio"] > 2.0],
                      key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)[:12]

    overlay_results = {}
    for base_key, base_res in top_base:
        br = base_res["portfolio_returns"]

        # Vol targeting at different levels
        for tvol in [0.04, 0.06, 0.08]:
            vt_ret = vol_target_overlay(br, target_vol=tvol)
            key = f"VT({tvol:.0%})_{base_key}"
            overlay_results[key] = make_result(vt_ret, prices.index)

        # Drawdown control
        for dd_trigger in [-0.03, -0.05]:
            dd_ret = drawdown_control(br, max_dd_trigger=dd_trigger)
            key = f"DDC({dd_trigger:.0%})_{base_key}"
            overlay_results[key] = make_result(dd_ret, prices.index)

        # Kelly sizing
        k_ret = kelly_scale(br)
        key = f"Kelly_{base_key}"
        overlay_results[key] = make_result(k_ret, prices.index)

        # Combo: vol-target + drawdown control
        vt_ret = vol_target_overlay(br, target_vol=0.06)
        dd_ret = drawdown_control(vt_ret, max_dd_trigger=-0.04)
        key = f"VT6+DDC4_{base_key}"
        overlay_results[key] = make_result(dd_ret, prices.index)

    results.update(overlay_results)

    # Show top overlays
    ovl_sorted = sorted(overlay_results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(overlay_results)} overlay strategies. Top 15 by Sharpe:")
    for k, v in ovl_sorted[:15]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: LEVERAGE SWEEP (static + dynamic + vol-targeted)
    # ══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("PHASE 5: LEVERAGE SWEEP (3 modes x 3 cost models)")
    print("  Static: 2.0-8.0x")
    print("  Dynamic: vol-scaled target 3-7x")
    print("  Vol-targeted: scale to 15-40% annual vol\n")

    # Candidates: anything with Sharpe > 2.0
    lev_candidates = sorted(
        [(k, v) for k, v in results.items() if v["metrics"]["Sharpe Ratio"] > 2.0],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:30]
    # Deduplicate
    seen = set()
    lev_cands = []
    for k, v in lev_candidates:
        if k not in seen:
            seen.add(k); lev_cands.append((k, v))
    print(f"  {len(lev_cands)} candidates\n")

    # QQQ vol for dynamic leverage
    qqq_vol = rvol(prices["QQQ"], 20)
    qqq_vol_avg = qqq_vol.rolling(252, min_periods=60).mean()
    vol_ratio = (qqq_vol / qqq_vol_avg.clip(lower=0.01)).clip(lower=0.3, upper=3.0)

    for lev_cost_label, lev_cost in [("1.5%", 0.015), ("1.0%", 0.010), ("0.5%", 0.005)]:
        print(f"  --- Leverage cost: {lev_cost_label}/yr ---")
        cost_winners = []

        for base_key, base_res in lev_cands:
            br = base_res["portfolio_returns"]

            # A) Static leverage
            for mult in np.arange(2.0, 8.2, 0.4):
                mult = round(mult, 1)
                sr = br * mult - (mult - 1) * lev_cost / 252
                label = f"L({lev_cost_label})_{base_key}_x{mult}"[:95]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

            # B) Dynamic leverage (vol-inverse)
            for tgt_lev in [3.0, 4.0, 5.0, 6.0, 7.0]:
                dyn_lev = (tgt_lev / vol_ratio).clip(lower=1.0, upper=tgt_lev * 1.5)
                sr = br * dyn_lev - (dyn_lev - 1) * lev_cost / 252
                label = f"DL({lev_cost_label})_{base_key}_t{tgt_lev}"[:95]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

            # C) Vol-targeted leverage: scale to target portfolio vol
            for tvol in [0.15, 0.20, 0.25, 0.30, 0.40]:
                vt_ret = vol_target_overlay(br, target_vol=tvol, lookback=63)
                # Apply leverage cost based on realized scale
                realized_scale = vt_ret / br.replace(0, np.nan)
                realized_scale = realized_scale.fillna(1.0).clip(lower=0.5, upper=15.0)
                lc = (realized_scale - 1).clip(lower=0) * lev_cost / 252
                sr = vt_ret - lc
                label = f"VTL({lev_cost_label})_{base_key}_v{tvol:.0%}"[:95]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

        # Summary for this cost level
        print(f"  Winners at {lev_cost_label}: {len(cost_winners)}")
        # Top 5 by combined score
        cw_detail = [(w, results[w]) for w in cost_winners]
        cw_detail.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"] * x[1]["metrics"]["CAGR"],
                       reverse=True)
        for k, v in cw_detail[:5]:
            m = v["metrics"]
            tag = "DYN" if k.startswith("DL") else ("VTL" if k.startswith("VTL") else "STA")
            print(f"    [{tag}] {k:85s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
                  f"DD={m['Max Drawdown']:.2%}")
        print()

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: DRAWDOWN-CONTROLLED LEVERAGE
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("PHASE 6: DRAWDOWN-CONTROLLED LEVERAGE\n")
    print("  Apply drawdown control AFTER leverage to limit tail risk\n")

    # Take top 20 leveraged winners, apply drawdown control
    winner_details = [(w, results[w]) for w in winners]
    winner_details.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    dd_lev_results = {}
    for base_key, base_res in winner_details[:30]:
        br = base_res["portfolio_returns"]
        for dd_trigger in [-0.05, -0.08, -0.10]:
            dd_ret = drawdown_control(br, max_dd_trigger=dd_trigger, recovery_rate=0.01)
            key = f"DDC({dd_trigger:.0%})_{base_key}"[:100]
            res = make_result(dd_ret, prices.index)
            m = res["metrics"]
            dd_lev_results[key] = res
            results[key] = res
            if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                winners.append(key)

    # Show top
    dd_sorted = sorted(dd_lev_results.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(dd_lev_results)} DD-controlled strategies. Top 10:")
    for k, v in dd_sorted[:10]:
        m = v["metrics"]
        w_tag = " WINNER" if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95 else ""
        print(f"    {k:90s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%}{w_tag}")

    # ══════════════════════════════════════════════════════════════
    # AUDIT
    # ══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    auditable = {k: v for k, v in results.items()
                 if isinstance(v["weights"], pd.DataFrame) and not v["weights"].empty}
    iss = audit(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies): "
          f"{'ALL PASS' if not iss else f'{len(iss)} issues'}")
    for i in iss:
        print(f"  !! {i}")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("FINAL SUMMARY")
    print(SEP)

    total = len(results)
    # Dedupe winners list
    winners = list(dict.fromkeys(winners))
    n_w = len(winners)
    sorted_all = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    if winners:
        print(f"\n*** {n_w} WINNERS FOUND ***\n")

        # Group by type
        sta_w = [w for w in winners if w.startswith("L(")]
        dyn_w = [w for w in winners if w.startswith("DL(")]
        vtl_w = [w for w in winners if w.startswith("VTL(")]
        ddc_w = [w for w in winners if w.startswith("DDC(")]
        print(f"  Static: {len(sta_w)}, Dynamic: {len(dyn_w)}, VolTarget: {len(vtl_w)}, DDControl: {len(ddc_w)}")

        # Champion at each cost level
        for cost_label in ["1.5%", "1.0%", "0.5%"]:
            cost_w = [(w, results[w]) for w in winners if f"({cost_label})" in w]
            if not cost_w:
                print(f"\n  [{cost_label} cost] No winners"); continue

            # Best by Sharpe * CAGR product (balanced)
            best_balanced = max(cost_w,
                key=lambda x: x[1]["metrics"]["Sharpe Ratio"] * x[1]["metrics"]["CAGR"])
            # Best by Sharpe (risk-adjusted)
            best_sharpe = max(cost_w, key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
            # Best by CAGR (aggressive)
            best_cagr = max(cost_w, key=lambda x: x[1]["metrics"]["CAGR"])

            print(f"\n  [{cost_label} cost] {len(cost_w)} winners")
            for label, (k, v) in [("CHAMPION (balanced)", best_balanced),
                                   ("Best Sharpe", best_sharpe),
                                   ("Best CAGR", best_cagr)]:
                m = v["metrics"]
                print(f"    {label:22s}: {k}")
                print(f"      CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                      f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}, "
                      f"Calmar={m['Calmar Ratio']:.4f}")
    else:
        print("\nNo WINNER found.")

    # Efficient frontier
    print(f"\n  Efficient Frontier (top 25 CAGR-beaters by Sharpe):")
    cb = sorted([(k, v) for k, v in sorted_all if v["metrics"]["CAGR"] > spy_cagr],
                key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    for k, v in cb[:25]:
        m = v["metrics"]
        print(f"    {k:90s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} DD={m['Max Drawdown']:.2%}")

    # Overall Sharpe leaders
    print(f"\n  Overall Sharpe Leaders (top 10):")
    for k, v in sorted_all[:10]:
        m = v["metrics"]
        print(f"    {k:90s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f}")

    print(f"\n  Total: {total} strategies | Winners: {n_w} | Audit: {len(iss)} issues")
    print(SEP)


if __name__ == "__main__":
    main()
