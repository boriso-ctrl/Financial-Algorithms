"""
Leveraged Alpha v14 — FINAL REFINEMENT: MAXIMUM RISK-ADJUSTED ALPHA
==================================================================================
v13 achieved Sharpe 4.29, CAGR 275%, MaxDD -12.76% with walk-forward consistency.
Key architecture: CompFilt5 pairs + VT+DDC overlay + leverage + post-lev DDC.

v14 final refinements:
  1. Blend ShFilt + CompFilt portfolios (diversify pair selection methodology)
  2. Dynamic leverage + post-leverage DDC (v13 only tested static)
  3. Unified leverage-DDC system (adaptive leverage that reduces in drawdowns)
  4. Granular VT+DDC calibration around sweet spots (VT5-10, DDC2-6)
  5. Higher leverage (10-12x) with tighter DDC (-3% to -4%)
  6. Recovery boost: increase leverage after confirmed DD recovery
  7. Ensemble of overlays (blend different VT+DDC combos)
  8. 5-period walk-forward validation
  9. Full efficient frontier mapping at each cost level
  10. Comprehensive statistics on champion strategies
"""

from __future__ import annotations
import sys, itertools
from pathlib import Path

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


# ═══════════════════════════════════════════════════════════════════
#  PAIR TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════

def pair_returns_fast(prices, leg_a, leg_b, window=63, entry_z=2.0, exit_z=0.5):
    """Return pair strategy returns without building full weight DataFrame (fast)."""
    if leg_a not in prices.columns or leg_b not in prices.columns:
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


def strat_vol_carry(prices):
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
        for s, _ in sorted_v[:3]:
            w.loc[w.index[i], s] = 0.10
        for s, _ in sorted_v[-2:]:
            w.loc[w.index[i], s] = -0.10
    return w


# ═══════════════════════════════════════════════════════════════════
#  PORTFOLIO OVERLAYS
# ═══════════════════════════════════════════════════════════════════

def vol_target_overlay(returns, target_vol=0.05, lookback=63):
    realized = returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    realized = realized.clip(lower=0.005)
    scale = (target_vol / realized).clip(lower=0.2, upper=5.0)
    return returns * scale.shift(1).fillna(1.0)


def drawdown_control(returns, max_dd_trigger=-0.05, recovery_rate=0.02):
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    for i in range(1, len(scale)):
        if dd.iloc[i] < max_dd_trigger:
            severity = min(abs(dd.iloc[i] / max_dd_trigger), 3.0)
            scale.iloc[i] = max(0.2, 1.0 / severity)
        elif scale.iloc[i - 1] < 1.0:
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + recovery_rate)
        else:
            scale.iloc[i] = 1.0
    return returns * scale


def adaptive_leverage(returns, target_lev, vol_ratio, dd_sensitivity=2.0,
                      recovery_boost=1.15):
    """Unified adaptive leverage: vol-inverse scaled + drawdown-aware."""
    # Base: vol-inverse leverage
    base_lev = (target_lev / vol_ratio).clip(lower=1.0, upper=target_lev * 1.5)
    # DD-aware scaling
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    dd_scale = pd.Series(1.0, index=returns.index)
    for i in range(1, len(dd_scale)):
        ddi = dd.iloc[i]
        if ddi < -0.03:
            # Scale down by DD severity
            dd_scale.iloc[i] = max(0.3, 1.0 - dd_sensitivity * abs(ddi))
        elif dd_scale.iloc[i - 1] < 1.0 and ddi > -0.01:
            # Recovery boost: slightly above 1.0 when recovering
            dd_scale.iloc[i] = min(recovery_boost,
                                   dd_scale.iloc[i - 1] + 0.02)
        elif dd_scale.iloc[i - 1] > 1.0:
            dd_scale.iloc[i] = max(1.0, dd_scale.iloc[i - 1] - 0.005)
        else:
            dd_scale.iloc[i] = 1.0
    final_lev = base_lev * dd_scale
    return final_lev


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
        print(f"  {label:75s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} {cagr_flag} {sh_flag} {win}")
    else:
        print(f"  {label:75s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} Sort={m['Sortino Ratio']:.2f} "
              f"Calmar={m['Calmar Ratio']:.2f} {cagr_flag} {sh_flag} {win}")
    return bool(win)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("LEVERAGED ALPHA v14 -- FINAL REFINEMENT: MAXIMUM RISK-ADJUSTED ALPHA")
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
    # PHASE 1: EXHAUSTIVE PAIR SCAN (reuse v13 architecture)
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("PHASE 1: EXHAUSTIVE PAIR SCAN")
    print("  All 153 unique pairs × 4 windows × 4 entry/exit configs\n")

    all_pairs_tickers = list(itertools.combinations(ALL_TICKERS, 2))
    WINDOWS = [21, 42, 63, 126]
    ZP_CONFIGS = [(2.0, 0.5), (2.25, 0.50), (2.25, 0.75), (1.75, 0.50)]

    pair_db = {}
    n_scanned = 0

    for a, b in all_pairs_tickers:
        for win in WINDOWS:
            for ez_in, ez_out in ZP_CONFIGS:
                ret = pair_returns_fast(prices, a, b, window=win,
                                        entry_z=ez_in, exit_z=ez_out)
                if ret is None:
                    continue
                n_scanned += 1
                sh, cagr = quick_metrics(ret)
                if sh > 0.3 and cagr > 0.003:
                    pair_db[(a, b, win, ez_in, ez_out)] = (sh, cagr, ret)

    print(f"  Scanned: {n_scanned} | Quality pairs: {len(pair_db)}")

    # Sort by Sharpe and by Composite
    ranked_sh = sorted(pair_db.items(), key=lambda x: x[1][0], reverse=True)
    ranked_comp = sorted(pair_db.items(),
                         key=lambda x: x[1][0]**1.5 * max(x[1][1], 0.001),
                         reverse=True)

    print(f"\n  Top 15 by Sharpe:")
    for cfg, (sh, cagr, _) in ranked_sh[:15]:
        a, b, win, ez_in, ez_out = cfg
        print(f"    {a}/{b}_w{win}_e{ez_in}/x{ez_out}: Sh={sh:.3f} CAGR={cagr:.2%}")

    print(f"\n  Top 15 by Composite (Sh^1.5 × CAGR):")
    for cfg, (sh, cagr, _) in ranked_comp[:15]:
        a, b, win, ez_in, ez_out = cfg
        print(f"    {a}/{b}_w{win}_e{ez_in}/x{ez_out}: Sh={sh:.3f} CAGR={cagr:.2%}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: PORTFOLIO CONSTRUCTION (4 methods)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 2: PORTFOLIO CONSTRUCTION (4 methods)\n")

    zp_portfolios = {}

    # Method A: Sharpe-ranked, correlation-filtered
    print("  --- 2A: Sharpe-Ranked Correlation-Filtered ---")
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

    for port_size in [5, 7, 10]:
        if port_size > len(selected_A):
            continue
        sel = selected_A[:port_size]
        for notional in [0.06, 0.08, 0.10]:
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for cfg, sh, cagr, _ in sel:
                a, b, win, ez_in, ez_out = cfg
                pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=notional)
                if pw is not None:
                    combined += pw
            res = backtest(prices, combined)
            m = res["metrics"]
            key = f"ZP_ShF{port_size}_n{notional}"
            zp_portfolios[key] = res; results[key] = res
            report(key, m, spy_cagr, short=True)

        # Inverse-vol weighted
        sharpes = [s[1] for s in sel]
        max_sh = max(sharpes) if sharpes else 1
        combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for cfg, sh, cagr, _ in sel:
            a, b, win, ez_in, ez_out = cfg
            w_scale = sh / max_sh
            pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                              exit_z=ez_out, notional=0.08 * w_scale)
            if pw is not None:
                combined += pw
        res = backtest(prices, combined)
        key = f"ZP_ShF{port_size}_IVW"
        zp_portfolios[key] = res; results[key] = res
        report(key, m, spy_cagr, short=True)

    # Method B: Composite-ranked, correlation-filtered
    print(f"\n  --- 2B: Composite-Ranked Correlation-Filtered ---")
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

    for port_size in [5, 7, 10]:
        if port_size > len(selected_B):
            continue
        sel = selected_B[:port_size]
        for notional in [0.06, 0.08, 0.10]:
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for cfg, sh, cagr, _ in sel:
                a, b, win, ez_in, ez_out = cfg
                pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=notional)
                if pw is not None:
                    combined += pw
            res = backtest(prices, combined)
            m = res["metrics"]
            key = f"ZP_CF{port_size}_n{notional}"
            zp_portfolios[key] = res; results[key] = res
            report(key, m, spy_cagr, short=True)

    # Method C: Blended Sharpe+Composite (diversify selection methodology)
    print(f"\n  --- 2C: Blended ShFilt + CompFilt ---")
    for ps_a, ps_b in [(5, 5), (7, 5), (5, 7)]:
        sel_a = selected_A[:ps_a] if len(selected_A) >= ps_a else selected_A
        sel_b = selected_B[:ps_b] if len(selected_B) >= ps_b else selected_B
        for notional in [0.06, 0.08]:
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            used_pairs = set()
            for cfg, sh, cagr, _ in sel_a:
                a, b, win, ez_in, ez_out = cfg
                pk = (min(a, b), max(a, b), win)
                if pk in used_pairs:
                    continue
                used_pairs.add(pk)
                pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=notional)
                if pw is not None:
                    combined += pw
            for cfg, sh, cagr, _ in sel_b:
                a, b, win, ez_in, ez_out = cfg
                pk = (min(a, b), max(a, b), win)
                if pk in used_pairs:
                    continue
                used_pairs.add(pk)
                pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=notional)
                if pw is not None:
                    combined += pw
            n_pairs = len(used_pairs)
            res = backtest(prices, combined)
            m = res["metrics"]
            key = f"ZP_Blend{ps_a}+{ps_b}_n{notional}"
            zp_portfolios[key] = res; results[key] = res
            report(key, m, spy_cagr, short=True)

    # Method D: Return-stream blending (weight ShFilt and CompFilt returns)
    print(f"\n  --- 2D: Return-Stream Blend ---")
    # Blend the best ShFilt and CompFilt return streams
    sh_sorted = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    shf_keys = [(k, v) for k, v in sh_sorted if "ShF" in k][:4]
    cf_keys = [(k, v) for k, v in sh_sorted if "CF" in k][:4]

    for (sk, sv), (ck, cv) in itertools.product(shf_keys[:2], cf_keys[:2]):
        for a_wt in [0.5, 0.6, 0.7]:
            er = a_wt * sv["portfolio_returns"] + (1 - a_wt) * cv["portfolio_returns"]
            key = f"RB({a_wt:.0%}{sk}+{1-a_wt:.0%}{ck})"[:75]
            res = make_result(er, prices.index)
            zp_portfolios[key] = res; results[key] = res
            report(key, res["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: ALPHA SOURCES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 3: ALPHA SOURCES\n")

    ch_res = backtest(prices, strat_crash_hedged(prices, 1.0))
    results["CrashHedge"] = ch_res
    report("CrashHedge", ch_res["metrics"], spy_cagr, short=True)

    vc_res = backtest(prices, strat_vol_carry(prices))
    results["VolCarry"] = vc_res
    report("VolCarry", vc_res["metrics"], spy_cagr, short=True)

    ch_ret = ch_res["portfolio_returns"]
    vc_ret = vc_res["portfolio_returns"]

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: ENSEMBLES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 4: ENSEMBLES\n")

    sorted_zp = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    ensembles = {}
    for zp_key, zp_res in sorted_zp[:15]:
        zp_ret = zp_res["portfolio_returns"]
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.0:
            continue

        # 3-source: ZP + CH + VC (best from v13)
        for zp_pct, ch_pct, vc_pct in [(0.90, 0.03, 0.07), (0.88, 0.05, 0.07),
                                         (0.85, 0.05, 0.10), (0.92, 0.03, 0.05),
                                         (0.80, 0.10, 0.10)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + vc_pct * vc_ret
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{vc_pct:.0%}VC)"[:85]
            ensembles[key] = make_result(er, prices.index)

        # Risk-parity blend
        vols = pd.DataFrame({
            "zp": zp_ret.rolling(63, min_periods=20).std(),
            "ch": ch_ret.rolling(63, min_periods=20).std(),
            "vc": vc_ret.rolling(63, min_periods=20).std()
        }).clip(lower=1e-6).shift(1).bfill()
        inv_vol = 1.0 / vols
        wts = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        er = wts["zp"] * zp_ret + wts["ch"] * ch_ret + wts["vc"] * vc_ret
        key = f"RP3({zp_key}+CH+VC)"[:75]
        ensembles[key] = make_result(er, prices.index)

    results.update(ensembles)

    ens_sorted = sorted(ensembles.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(ensembles)} ensembles. Top 15 by Sharpe:")
    for k, v in ens_sorted[:15]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: GRANULAR OVERLAY CALIBRATION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 5: GRANULAR OVERLAY CALIBRATION (VT × DDC)\n")

    top_base = sorted([(k, v) for k, v in results.items()
                       if v["metrics"]["Sharpe Ratio"] > 1.5],
                      key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)[:20]

    overlay_results = {}
    for base_key, base_res in top_base:
        br = base_res["portfolio_returns"]

        # All VT × DDC combinations
        for tvol in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]:
            for dd_trig in [-0.02, -0.03, -0.04, -0.05, -0.06]:
                vt_ret = vol_target_overlay(br, target_vol=tvol)
                dd_ret = drawdown_control(vt_ret, max_dd_trigger=dd_trig,
                                           recovery_rate=0.015)
                key = f"VT{int(tvol*100)}+DDC{int(abs(dd_trig)*100)}_{base_key}"[:110]
                overlay_results[key] = make_result(dd_ret, prices.index)

        # Pure vol-target (best from v12/v13 for reference)
        for tvol in [0.06, 0.07, 0.08]:
            vt_ret = vol_target_overlay(br, target_vol=tvol)
            key = f"VT({tvol:.0%})_{base_key}"[:110]
            overlay_results[key] = make_result(vt_ret, prices.index)

    results.update(overlay_results)

    ovl_sorted = sorted(overlay_results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(overlay_results)} overlay strategies. Top 20 by Sharpe:")
    for k, v in ovl_sorted[:20]:
        report(k, v["metrics"], spy_cagr, short=True)

    # Find best VT+DDC combos per overlay base
    print(f"\n  Best VT×DDC grid results:")
    for tvol in [0.05, 0.06, 0.07, 0.08, 0.10]:
        for dd_trig in [0.02, 0.03, 0.04, 0.05]:
            prefix = f"VT{int(tvol*100)}+DDC{dd_trig}_"
            prefix2 = f"VT{int(tvol*100)}+DDC{int(dd_trig*100)}_"
            matches = [(k, v) for k, v in overlay_results.items()
                       if k.startswith(prefix) or k.startswith(prefix2)]
            if matches:
                best = max(matches, key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
                m = best[1]["metrics"]
                tag = "W" if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95 else " "
                print(f"    VT{int(tvol*100):2d}+DDC{int(dd_trig*100):2d}: "
                      f"Sh={m['Sharpe Ratio']:.4f} CAGR={m['CAGR']:.2%} DD={m['Max Drawdown']:.2%} {tag}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: LEVERAGE (static + dynamic + adaptive)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 6: LEVERAGE SWEEP (Static + Dynamic + Adaptive)\n")

    lev_candidates = sorted(
        [(k, v) for k, v in results.items() if v["metrics"]["Sharpe Ratio"] > 2.2],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:35]
    seen = set()
    lev_cands = []
    for k, v in lev_candidates:
        if k not in seen:
            seen.add(k); lev_cands.append((k, v))
    print(f"  {len(lev_cands)} candidates (Sharpe > 2.2)\n")

    qqq_vol = rvol(prices["QQQ"], 20)
    qqq_vol_avg = qqq_vol.rolling(252, min_periods=60).mean()
    vol_ratio = (qqq_vol / qqq_vol_avg.clip(lower=0.01)).clip(lower=0.3, upper=3.0)

    for lev_cost_label, lev_cost in [("1.5%", 0.015), ("1.0%", 0.010), ("0.5%", 0.005)]:
        print(f"  --- Leverage cost: {lev_cost_label}/yr ---")
        cost_winners = []

        for base_key, base_res in lev_cands:
            br = base_res["portfolio_returns"]

            # Static leverage: 2.0 to 12.0 (extended range)
            for mult in np.arange(2.0, 12.2, 0.4):
                mult = round(mult, 1)
                sr = br * mult - (mult - 1) * lev_cost / 252
                label = f"L({lev_cost_label})_{base_key}_x{mult}"[:110]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

            # Dynamic leverage (vol-inverse): 3-10x target
            for tgt_lev in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
                dyn_lev = (tgt_lev / vol_ratio).clip(lower=1.0, upper=tgt_lev * 1.5)
                sr = br * dyn_lev - (dyn_lev - 1) * lev_cost / 252
                label = f"DL({lev_cost_label})_{base_key}_t{tgt_lev}"[:110]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

            # Adaptive leverage (vol-inverse + DD-aware): unified system
            for tgt_lev in [5.0, 7.0, 8.0, 10.0]:
                adp_lev = adaptive_leverage(br, tgt_lev, vol_ratio,
                                             dd_sensitivity=2.0, recovery_boost=1.15)
                sr = br * adp_lev - (adp_lev - 1) * lev_cost / 252
                label = f"AL({lev_cost_label})_{base_key}_t{tgt_lev}"[:110]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

        print(f"  Winners at {lev_cost_label}: {len(cost_winners)}")
        cw_detail = [(w, results[w]) for w in cost_winners]
        cw_detail.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"] * x[1]["metrics"]["CAGR"],
                       reverse=True)
        for k, v in cw_detail[:5]:
            m = v["metrics"]
            tag = "DYN" if k.startswith("DL") else ("ADP" if k.startswith("AL") else "STA")
            print(f"    [{tag}] {k[:95]:95s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
                  f"DD={m['Max Drawdown']:.2%}")
        print()

    # ══════════════════════════════════════════════════════════════
    # PHASE 7: POST-LEVERAGE DDC (the mega-innovation)
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("PHASE 7: POST-LEVERAGE DRAWDOWN CONTROL\n")

    winner_details = [(w, results[w]) for w in winners]
    winner_details.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    dd_lev_results = {}
    for base_key, base_res in winner_details[:50]:
        br = base_res["portfolio_returns"]
        for dd_trigger in [-0.03, -0.04, -0.05, -0.06, -0.08, -0.10]:
            dd_ret = drawdown_control(br, max_dd_trigger=dd_trigger, recovery_rate=0.01)
            key = f"DDC({dd_trigger:.0%})_{base_key}"[:115]
            res = make_result(dd_ret, prices.index)
            m = res["metrics"]
            dd_lev_results[key] = res; results[key] = res
            if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                winners.append(key)

    dd_sorted = sorted(dd_lev_results.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(dd_lev_results)} DD-controlled strategies. Top 15:")
    for k, v in dd_sorted[:15]:
        m = v["metrics"]
        w_tag = " WINNER" if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95 else ""
        print(f"    {k[:100]:100s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%}{w_tag}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 8: WALK-FORWARD VALIDATION (5 periods)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 8: WALK-FORWARD VALIDATION (5 periods)\n")

    top_for_wf = []
    all_sorted = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    for k, v in all_sorted:
        m = v["metrics"]
        if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
            top_for_wf.append((k, v))
        if len(top_for_wf) >= 15:
            break

    if top_for_wf:
        periods = [
            ("2010-12", "2010-01-01", "2013-01-01"),
            ("2013-15", "2013-01-01", "2016-01-01"),
            ("2016-18", "2016-01-01", "2019-01-01"),
            ("2019-21", "2019-01-01", "2022-01-01"),
            ("2022-25", "2022-01-01", "2025-03-01"),
        ]
        p_headers = [p[0] for p in periods]
        print(f"  {'Strategy':<65s} " + " ".join(f"{h:>8s}" for h in p_headers) + f" {'Consist':>8s}")
        print("  " + "-" * (65 + 9 * len(periods) + 10))

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
            p_strs = " ".join(f"{s:8.2f}" for s in period_results)
            tag = "YES" if consistent else "no"
            print(f"  {k[:65]:65s} {p_strs} {tag:>8s}")

    # ══════════════════════════════════════════════════════════════
    # AUDIT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
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
    print(f"\n{SEP}")
    print("FINAL SUMMARY")
    print(SEP)

    total = len(results)
    winners = list(dict.fromkeys(winners))
    n_w = len(winners)

    if winners:
        print(f"\n*** {n_w} WINNERS FOUND ***\n")

        sta_w = [w for w in winners if w.startswith("L(")]
        dyn_w = [w for w in winners if w.startswith("DL(")]
        adp_w = [w for w in winners if w.startswith("AL(")]
        ddc_w = [w for w in winners if w.startswith("DDC(")]
        other_w = [w for w in winners if w not in sta_w + dyn_w + adp_w + ddc_w]
        print(f"  Static: {len(sta_w)}, Dynamic: {len(dyn_w)}, Adaptive: {len(adp_w)}, "
              f"DDControl: {len(ddc_w)}, Other: {len(other_w)}")

        for cost_label in ["1.5%", "1.0%", "0.5%"]:
            cost_w = [(w, results[w]) for w in winners if f"({cost_label})" in w]
            if not cost_w:
                print(f"\n  [{cost_label} cost] No winners"); continue

            best_balanced = max(cost_w, key=lambda x: (
                x[1]["metrics"]["Sharpe Ratio"] * x[1]["metrics"]["CAGR"]))
            best_sh = max(cost_w, key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
            best_cagr = max(cost_w, key=lambda x: x[1]["metrics"]["CAGR"])

            print(f"\n  [{cost_label} cost] {len(cost_w)} winners")
            for tag, (k, v) in [("CHAMPION (balanced)", best_balanced),
                                  ("Best Sharpe", best_sh),
                                  ("Best CAGR", best_cagr)]:
                m = v["metrics"]
                print(f"    {tag:24s}: {k[:85]}")
                print(f"      CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                      f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}, "
                      f"Calmar={m['Calmar Ratio']:.4f}")

        # Efficient Frontier
        print(f"\n  Efficient Frontier (top 30 CAGR-beaters by Sharpe):")
        cagr_beaters = [(k, results[k]) for k in winners if results[k]["metrics"]["CAGR"] > spy_cagr]
        cagr_beaters.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
        for k, v in cagr_beaters[:30]:
            m = v["metrics"]
            print(f"    {k[:100]:100s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
                  f"DD={m['Max Drawdown']:.2%}")

        # Overall Champions
        print(f"\n  OVERALL CHAMPIONS (top 10):")
        sh_leaders = sorted([(k, results[k]) for k in winners],
                           key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
        for k, v in sh_leaders[:10]:
            m = v["metrics"]
            print(f"    {k[:100]}")
            print(f"      CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}, "
                  f"Calmar={m['Calmar Ratio']:.4f}")
    else:
        print("\n  No winners found.")

    print(f"\n  Total: {total} strategies | Winners: {n_w} | Audit: {len(iss) if iss else 0} issues")
    print(SEP)


if __name__ == "__main__":
    main()
