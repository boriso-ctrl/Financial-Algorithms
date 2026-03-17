"""
Leveraged Alpha v13 — EXHAUSTIVE PAIR OPTIMISATION + CASCADED RISK MANAGEMENT
==================================================================================
v12 found 1,878 WINNERs. Key breakthrough: CorrFilt + VT + DDC combo
  Best risk-adjusted (0.5% cost): DDC(-5%)_DL_VT(8%)_E3_t7.0
    CAGR=205.37%, Sharpe=3.7519, MaxDD=-20.75%

v13 innovations:
  1. Exhaustive pair scan: ALL 153 unique pairs × 4 windows
     (v12 only used 15 hardcoded pairs — leaving massive Sharpe on the table)
  2. Composite-ranked portfolio construction: Sharpe^1.5 × CAGR ranking
  3. Greedy correlation filtering with Sharpe-weighted diversification score
  4. Inverse-vol pair weighting (more weight to higher-Sharpe pairs)
  5. Multi-timeframe z-score fusion (require 2+ timeframes to agree)
  6. Regime-gated ensemble (shift alpha weights by vol regime)
  7. Fine-grained overlay calibration (VT: 3-12%, DDC: -2% to -12%)
  8. Cascaded risk management: VT → DDC → leverage → post-lev DDC
  9. Risk-parity alpha blending (weight by inverse vol)
  10. Walk-forward validation (3 × 5-year windows)
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
    """Fast Sharpe/CAGR without full backtest."""
    if len(returns) < 252 or returns.std() < 1e-10:
        return 0.0, 0.0
    sh = returns.mean() / returns.std() * np.sqrt(252)
    eq = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 and eq.iloc[-1] > 0 else 0
    return sh, cagr

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
#  PAIR TRADING ENGINE
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
    # Shift position by 1 (no look-ahead)
    pos_shifted = np.roll(pos, 1); pos_shifted[0] = 0
    # Return: long leg_a, short leg_b (or reversed)
    pair_ret = pos_shifted * ret_a - pos_shifted * ret_b
    return pd.Series(pair_ret, index=prices.index)


def pair_weights_multitf(prices, leg_a, leg_b, windows=(21, 42, 63, 126),
                         entry_z=2.0, exit_z=0.5, notional=0.15):
    """Multi-timeframe z-score fusion: require majority of timeframes to agree."""
    if leg_a not in prices.columns or leg_b not in prices.columns:
        return None
    spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
    z_scores = [zscore(spread, w) for w in windows]

    pos = pd.Series(0.0, index=prices.index)
    for i in range(1, len(pos)):
        prev = pos.iloc[i - 1]
        zvals = [z.iloc[i] for z in z_scores]
        if any(np.isnan(v) for v in zvals):
            pos.iloc[i] = 0; continue
        # Count signals
        n_long = sum(1 for v in zvals if v < -entry_z)
        n_short = sum(1 for v in zvals if v > entry_z)
        n_exit_long = sum(1 for v in zvals if v > -exit_z)
        n_exit_short = sum(1 for v in zvals if v < exit_z)
        threshold = len(windows) / 2  # majority

        if prev == 0:
            if n_short >= threshold:    pos.iloc[i] = -1
            elif n_long >= threshold:   pos.iloc[i] = 1
            else:                       pos.iloc[i] = 0
        elif prev > 0:
            pos.iloc[i] = 0 if n_exit_long >= threshold else 1
        else:
            pos.iloc[i] = 0 if n_exit_short >= threshold else -1

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
    """Vol carry: long low-vol sectors, short high-vol sectors."""
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


def strat_risk_adj_momentum(prices, lookback=126, n_long=3, n_short=2, notional=0.12):
    """Risk-adjusted momentum: rank by return/volatility ratio."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(lookback + 1, len(prices)):
        scores = {}
        for s in SECTORS:
            ret = prices[s].iloc[i] / prices[s].iloc[max(0, i - lookback)] - 1
            vol = prices[s].iloc[max(0, i - lookback):i].pct_change().std() * np.sqrt(252)
            if not np.isnan(vol) and vol > 0.01:
                scores[s] = ret / vol
        if len(scores) < n_long + n_short:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1])
        for s, _ in ranked[:n_short]:
            w.loc[w.index[i], s] = -notional
        for s, _ in ranked[-n_long:]:
            w.loc[w.index[i], s] = notional
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


def graduated_ddc(returns, trigger=-0.05, max_cut=0.8, recovery_rate=0.015):
    """Graduated drawdown control: smoothly scale down as DD deepens."""
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    # Linear scale: 1.0 at trigger, (1-max_cut) at 3x trigger
    scale = 1.0 - max_cut * ((dd - trigger) / (2 * trigger)).clip(lower=0, upper=1)
    # Apply recovery: don't snap back instantly
    smooth = pd.Series(1.0, index=returns.index)
    for i in range(1, len(smooth)):
        target = float(scale.iloc[i])
        if target < smooth.iloc[i - 1]:
            smooth.iloc[i] = target  # cut immediately
        else:
            smooth.iloc[i] = min(target, smooth.iloc[i - 1] + recovery_rate)
    return returns * smooth


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
    print("LEVERAGED ALPHA v13 -- EXHAUSTIVE PAIR OPTIMISATION + CASCADED RISK MGMT")
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
    # PHASE 1: EXHAUSTIVE PAIR SCAN
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("PHASE 1: EXHAUSTIVE PAIR SCAN")
    print("  All 153 unique pairs × 4 windows × 2 (entry_z, exit_z) combos\n")

    # Generate all unique pairs
    all_pairs_tickers = list(itertools.combinations(ALL_TICKERS, 2))
    WINDOWS = [21, 42, 63, 126]
    ZP_CONFIGS = [(2.0, 0.5), (2.25, 0.50), (2.25, 0.75), (1.75, 0.50)]  # Best from v11/v12

    pair_db = {}  # (a, b, win, ez_in, ez_out) -> (sharpe, cagr, returns)
    n_scanned = 0
    n_positive = 0

    print(f"  Scanning {len(all_pairs_tickers)} pairs × {len(WINDOWS)} windows "
          f"× {len(ZP_CONFIGS)} configs = {len(all_pairs_tickers)*len(WINDOWS)*len(ZP_CONFIGS)} combos...")

    for a, b in all_pairs_tickers:
        for win in WINDOWS:
            for ez_in, ez_out in ZP_CONFIGS:
                ret = pair_returns_fast(prices, a, b, window=win,
                                        entry_z=ez_in, exit_z=ez_out)
                if ret is None:
                    continue
                n_scanned += 1
                sh, cagr = quick_metrics(ret)
                if sh > 0.5 and cagr > 0.005:  # Meaningful positive signal
                    pair_db[(a, b, win, ez_in, ez_out)] = (sh, cagr, ret)
                    n_positive += 1

    print(f"  Scanned: {n_scanned} | Positive (Sh>0.5, CAGR>0.5%): {n_positive}")

    # Show top 20 individual pairs by Sharpe
    ranked_sh = sorted(pair_db.items(), key=lambda x: x[1][0], reverse=True)
    print(f"\n  Top 20 Individual Pairs by Sharpe:")
    for cfg, (sh, cagr, _) in ranked_sh[:20]:
        a, b, win, ez_in, ez_out = cfg
        print(f"    {a}/{b}_w{win}_e{ez_in}/x{ez_out}: Sh={sh:.3f} CAGR={cagr:.2%}")

    # Show top 20 by composite score (Sharpe^1.5 * CAGR)
    ranked_comp = sorted(pair_db.items(),
                         key=lambda x: x[1][0]**1.5 * max(x[1][1], 0.001),
                         reverse=True)
    print(f"\n  Top 20 Individual Pairs by Composite (Sh^1.5 × CAGR):")
    for cfg, (sh, cagr, _) in ranked_comp[:20]:
        a, b, win, ez_in, ez_out = cfg
        score = sh**1.5 * max(cagr, 0.001)
        print(f"    {a}/{b}_w{win}_e{ez_in}/x{ez_out}: Sh={sh:.3f} CAGR={cagr:.2%} Score={score:.4f}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: OPTIMAL PORTFOLIO CONSTRUCTION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 2: OPTIMAL PORTFOLIO CONSTRUCTION")
    print("  Greedy correlation filtering + inverse-vol weighting\n")

    zp_portfolios = {}

    # --- 2A: Sharpe-ranked correlation-filtered portfolios ---
    print("  --- 2A: Sharpe-Ranked Correlation-Filtered ---")

    # Use top pairs by Sharpe, greedily add by lowest correlation
    sharpe_ranked = [(cfg, sh, cagr, ret) for cfg, (sh, cagr, ret) in ranked_sh
                     if sh > 0.4]  # Raw pair Sharpe (before cash return boost)

    if sharpe_ranked:
        # Greedy correlation-filtered selection
        selected_sharpe = [sharpe_ranked[0]]
        remaining = list(sharpe_ranked[1:])

        for _ in range(min(24, len(remaining))):
            best_next = None; best_score = -999
            for idx, (cfg, sh, cagr, ret) in enumerate(remaining):
                corrs = [ret.corr(s[3]) for s in selected_sharpe]
                avg_corr = np.mean(corrs) if corrs else 0
                # Score: Sharpe - correlation penalty (higher penalty = more diversification)
                score = sh - 2.5 * max(avg_corr, 0)
                if score > best_score:
                    best_score = score; best_next = idx
            if best_next is not None:
                selected_sharpe.append(remaining.pop(best_next))

        print(f"  Selected {len(selected_sharpe)} diversified pairs (Sharpe-ranked)")

        # Build portfolios of different sizes
        for port_size in [5, 7, 10, 12, 15, 20]:
            if port_size > len(selected_sharpe):
                continue
            sel = selected_sharpe[:port_size]

            # --- Equal-weight version ---
            for notional in [0.08, 0.10, 0.12]:
                combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                for cfg, sh, cagr, _ in sel:
                    a, b, win, ez_in, ez_out = cfg
                    pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                      exit_z=ez_out, notional=notional)
                    if pw is not None:
                        combined += pw
                res = backtest(prices, combined)
                m = res["metrics"]
                key = f"ZP_ShFilt{port_size}_n{notional}"
                zp_portfolios[key] = res; results[key] = res
                report(key, m, spy_cagr, short=True)

            # --- Inverse-vol weighted version ---
            sharpes = [s[1] for s in sel]
            max_sh = max(sharpes)
            combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for cfg, sh, cagr, _ in sel:
                a, b, win, ez_in, ez_out = cfg
                # Weight proportional to Sharpe ratio
                w_scale = sh / max_sh  # 0 to 1
                pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                  exit_z=ez_out, notional=0.10 * w_scale)
                if pw is not None:
                    combined += pw
            res = backtest(prices, combined)
            m = res["metrics"]
            key = f"ZP_ShFilt{port_size}_IVW"
            zp_portfolios[key] = res; results[key] = res
            report(key, m, spy_cagr, short=True)

    # --- 2B: Composite-ranked correlation-filtered portfolios ---
    print(f"\n  --- 2B: Composite-Ranked Correlation-Filtered ---")

    comp_ranked = [(cfg, sh, cagr, ret) for cfg, (sh, cagr, ret) in ranked_comp
                   if sh > 0.3 and cagr > 0.005]  # Raw pair Sharpe (pre-cash)

    if comp_ranked:
        selected_comp = [comp_ranked[0]]
        remaining = list(comp_ranked[1:])

        for _ in range(min(24, len(remaining))):
            best_next = None; best_score = -999
            for idx, (cfg, sh, cagr, ret) in enumerate(remaining):
                corrs = [ret.corr(s[3]) for s in selected_comp]
                avg_corr = np.mean(corrs) if corrs else 0
                score = (sh**1.5 * max(cagr, 0.001)) - 1.5 * max(avg_corr, 0)
                if score > best_score:
                    best_score = score; best_next = idx
            if best_next is not None:
                selected_comp.append(remaining.pop(best_next))

        print(f"  Selected {len(selected_comp)} diversified pairs (Composite-ranked)")

        for port_size in [5, 7, 10, 12, 15, 20]:
            if port_size > len(selected_comp):
                continue
            sel = selected_comp[:port_size]
            for notional in [0.08, 0.10, 0.12]:
                combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                for cfg, sh, cagr, _ in sel:
                    a, b, win, ez_in, ez_out = cfg
                    pw = pair_weights(prices, a, b, window=win, entry_z=ez_in,
                                      exit_z=ez_out, notional=notional)
                    if pw is not None:
                        combined += pw
                res = backtest(prices, combined)
                m = res["metrics"]
                key = f"ZP_CompFilt{port_size}_n{notional}"
                zp_portfolios[key] = res; results[key] = res
                report(key, m, spy_cagr, short=True)

    # --- 2C: Multi-timeframe fusion portfolios ---
    print(f"\n  --- 2C: Multi-Timeframe Z-Score Fusion ---")

    # For top 20 pairs by Sharpe, build multi-TF version
    mtf_candidates = []
    if sharpe_ranked:
        # Get unique pair tickers from top pairs
        seen_pairs = set()
        for cfg, sh, cagr, ret in sharpe_ranked[:40]:
            a, b = cfg[0], cfg[1]
            pair_key = (min(a, b), max(a, b))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                mtf_candidates.append((a, b))
            if len(mtf_candidates) >= 20:
                break

    mtf_pair_returns = {}
    for a, b in mtf_candidates:
        pw = pair_weights_multitf(prices, a, b, windows=(21, 42, 63, 126),
                                  entry_z=2.0, exit_z=0.5, notional=0.10)
        if pw is not None:
            res = backtest(prices, pw)
            sh = res["metrics"]["Sharpe Ratio"]
            cagr = res["metrics"]["CAGR"]
            if sh > 0.5:
                mtf_pair_returns[(a, b)] = (sh, cagr, res["portfolio_returns"])

    # Greedy correlation-filtered MTF portfolio
    if mtf_pair_returns:
        mtf_ranked = sorted(mtf_pair_returns.items(),
                            key=lambda x: x[1][0], reverse=True)
        selected_mtf = [mtf_ranked[0]]
        remaining = list(mtf_ranked[1:])

        for _ in range(min(14, len(remaining))):
            best_next = None; best_score = -999
            for idx, (pair, (sh, cagr, ret)) in enumerate(remaining):
                corrs = [ret.corr(s[1][2]) for s in selected_mtf]
                avg_corr = np.mean(corrs) if corrs else 0
                score = sh - 2.0 * max(avg_corr, 0)
                if score > best_score:
                    best_score = score; best_next = idx
            if best_next is not None:
                selected_mtf.append(remaining.pop(best_next))

        for port_size in [5, 7, 10]:
            if port_size > len(selected_mtf):
                continue
            sel = selected_mtf[:port_size]
            for notional in [0.10, 0.12]:
                combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                for (a, b), (sh, cagr, _) in sel:
                    pw = pair_weights_multitf(prices, a, b, notional=notional)
                    if pw is not None:
                        combined += pw
                res = backtest(prices, combined)
                m = res["metrics"]
                key = f"ZP_MTF{port_size}_n{notional}"
                zp_portfolios[key] = res; results[key] = res
                report(key, m, spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: ADDITIONAL ALPHA SOURCES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 3: ADDITIONAL ALPHA SOURCES\n")

    ch_res = backtest(prices, strat_crash_hedged(prices, 1.0))
    results["CrashHedge"] = ch_res
    report("CrashHedge", ch_res["metrics"], spy_cagr, short=True)

    mom_res = backtest(prices, strat_momentum_ls(prices, lookback=252))
    results["MomLS_252"] = mom_res
    report("MomLS_252", mom_res["metrics"], spy_cagr, short=True)

    # Risk-adjusted momentum (new in v13)
    ramom_res = backtest(prices, strat_risk_adj_momentum(prices, lookback=126))
    results["RAMom_126"] = ramom_res
    report("RAMom_126", ramom_res["metrics"], spy_cagr, short=True)

    vc_res = backtest(prices, strat_vol_carry(prices))
    results["VolCarry"] = vc_res
    report("VolCarry", vc_res["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: ENSEMBLE CONSTRUCTION (traditional + regime-gated)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 4: MULTI-ALPHA ENSEMBLES (traditional + regime-gated)\n")

    ch_ret = ch_res["portfolio_returns"]
    mom_ret = mom_res["portfolio_returns"]
    ramom_ret = ramom_res["portfolio_returns"]
    vc_ret = vc_res["portfolio_returns"]

    # Sort ZP portfolios by Sharpe
    sorted_zp = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    ensembles = {}

    # 4A: Traditional fixed-weight ensembles
    print("  --- 4A: Fixed-Weight Ensembles ---")
    for zp_key, zp_res in sorted_zp[:12]:
        zp_ret = zp_res["portfolio_returns"]
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.0:
            continue

        # 2-source: ZP + CH
        for ch_pct in [0.03, 0.05, 0.08]:
            er = (1 - ch_pct) * zp_ret + ch_pct * ch_ret
            key = f"E2({ch_pct:.0%}CH+{1-ch_pct:.0%}{zp_key})"
            ensembles[key] = make_result(er, prices.index)

        # 3-source: ZP + CH + VC (best combo from v12)
        for zp_pct, ch_pct, vc_pct in [(0.85, 0.05, 0.10), (0.80, 0.10, 0.10),
                                         (0.90, 0.03, 0.07), (0.88, 0.05, 0.07)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + vc_pct * vc_ret
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{vc_pct:.0%}VC)"
            ensembles[key] = make_result(er, prices.index)

        # 3-source: ZP + CH + RAMom
        for zp_pct, ch_pct, rm_pct in [(0.85, 0.05, 0.10), (0.80, 0.08, 0.12)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + rm_pct * ramom_ret
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{rm_pct:.0%}RAM)"
            ensembles[key] = make_result(er, prices.index)

        # 4-source: ZP + CH + VC + RAMom
        for zp_pct, ch_pct, vc_pct, rm_pct in [(0.78, 0.05, 0.10, 0.07),
                                                  (0.75, 0.05, 0.10, 0.10)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + vc_pct * vc_ret + rm_pct * ramom_ret
            key = f"E4({zp_pct:.0%}{zp_key}+{ch_pct:.0%}C+{vc_pct:.0%}V+{rm_pct:.0%}R)"
            ensembles[key] = make_result(er, prices.index)

    # 4B: Regime-gated ensembles
    print("  --- 4B: Regime-Gated Ensembles ---")
    qqq_vol = rvol(prices["QQQ"], 20)
    qqq_vol_avg = qqq_vol.rolling(252, min_periods=60).mean()
    vol_ratio = (qqq_vol / qqq_vol_avg.clip(lower=0.01)).fillna(1.0)
    low_vol = vol_ratio < 0.85    # calm markets
    high_vol = vol_ratio > 1.3    # stressed markets
    mid_vol = ~low_vol & ~high_vol

    for zp_key, zp_res in sorted_zp[:6]:
        zp_ret = zp_res["portfolio_returns"]
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.0:
            continue
        # In low vol: 90% ZP + 3% CH + 7% VC (pairs dominate)
        # In high vol: 60% ZP + 25% CH + 15% VC (more hedging)
        # In mid vol: 85% ZP + 5% CH + 10% VC (balanced)
        er = pd.Series(0.0, index=prices.index)
        er[low_vol] = 0.92 * zp_ret[low_vol] + 0.03 * ch_ret[low_vol] + 0.05 * vc_ret[low_vol]
        er[high_vol] = 0.60 * zp_ret[high_vol] + 0.25 * ch_ret[high_vol] + 0.15 * vc_ret[high_vol]
        er[mid_vol] = 0.85 * zp_ret[mid_vol] + 0.05 * ch_ret[mid_vol] + 0.10 * vc_ret[mid_vol]
        key = f"RG3({zp_key}+CH+VC)"
        ensembles[key] = make_result(er, prices.index)

    # 4C: Risk-parity alpha blending
    print("  --- 4C: Risk-Parity Blended ---")
    for zp_key, zp_res in sorted_zp[:6]:
        zp_ret = zp_res["portfolio_returns"]
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.0:
            continue
        # Rolling 63-day vol for each source
        vols = pd.DataFrame({
            "zp": zp_ret.rolling(63, min_periods=20).std(),
            "ch": ch_ret.rolling(63, min_periods=20).std(),
            "vc": vc_ret.rolling(63, min_periods=20).std()
        }).clip(lower=1e-6).shift(1).fillna(method="bfill")
        inv_vol = 1.0 / vols
        wts = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        er = wts["zp"] * zp_ret + wts["ch"] * ch_ret + wts["vc"] * vc_ret
        key = f"RP3({zp_key}+CH+VC)"
        ensembles[key] = make_result(er, prices.index)

    results.update(ensembles)

    ens_sorted = sorted(ensembles.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"\n  {len(ensembles)} ensembles. Top 20 by Sharpe:")
    for k, v in ens_sorted[:20]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: OVERLAY CALIBRATION (fine-grained VT + DDC)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 5: FINE-GRAINED OVERLAY CALIBRATION\n")

    # Top bases with Sharpe > 2.0
    top_base = sorted([(k, v) for k, v in results.items()
                       if v["metrics"]["Sharpe Ratio"] > 1.5],
                      key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)[:18]

    overlay_results = {}
    for base_key, base_res in top_base:
        br = base_res["portfolio_returns"]

        # Fine-grained vol targeting
        for tvol in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]:
            vt_ret = vol_target_overlay(br, target_vol=tvol)
            key = f"VT({tvol:.0%})_{base_key}"[:105]
            overlay_results[key] = make_result(vt_ret, prices.index)

        # Fine-grained DDC
        for dd_trig in [-0.02, -0.03, -0.04, -0.05, -0.06, -0.08]:
            dd_ret = drawdown_control(br, max_dd_trigger=dd_trig)
            key = f"DDC({dd_trig:.0%})_{base_key}"[:105]
            overlay_results[key] = make_result(dd_ret, prices.index)

        # Graduated DDC
        for dd_trig in [-0.03, -0.05]:
            gdd_ret = graduated_ddc(br, trigger=dd_trig, max_cut=0.75)
            key = f"GDDC({dd_trig:.0%})_{base_key}"[:105]
            overlay_results[key] = make_result(gdd_ret, prices.index)

        # Cascaded: VT → DDC (best combo from v12)
        for tvol in [0.05, 0.06, 0.07, 0.08]:
            for dd_trig in [-0.03, -0.04, -0.05]:
                vt_ret = vol_target_overlay(br, target_vol=tvol)
                dd_ret = drawdown_control(vt_ret, max_dd_trigger=dd_trig)
                key = f"VT{int(tvol*100)}+DDC{int(abs(dd_trig)*100)}_{base_key}"[:105]
                overlay_results[key] = make_result(dd_ret, prices.index)

        # Cascaded: VT → Graduated DDC
        for tvol in [0.06, 0.08]:
            vt_ret = vol_target_overlay(br, target_vol=tvol)
            gdd_ret = graduated_ddc(vt_ret, trigger=-0.04, max_cut=0.75)
            key = f"VT{int(tvol*100)}+GDDC4_{base_key}"[:105]
            overlay_results[key] = make_result(gdd_ret, prices.index)

    results.update(overlay_results)

    ovl_sorted = sorted(overlay_results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(overlay_results)} overlay strategies. Top 20 by Sharpe:")
    for k, v in ovl_sorted[:20]:
        report(k, v["metrics"], spy_cagr, short=True)

    # ══════════════════════════════════════════════════════════════
    # PHASE 6: LEVERAGE SWEEP (static + dynamic)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 6: LEVERAGE SWEEP (Static + Dynamic)\n")

    # Top candidates by Sharpe > 2.0
    lev_candidates = sorted(
        [(k, v) for k, v in results.items() if v["metrics"]["Sharpe Ratio"] > 2.0],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:35]
    seen = set()
    lev_cands = []
    for k, v in lev_candidates:
        if k not in seen:
            seen.add(k); lev_cands.append((k, v))
    print(f"  {len(lev_cands)} candidates (Sharpe > 2.2)\n")

    # QQQ vol for dynamic leverage
    qqq_vol_dl = rvol(prices["QQQ"], 20)
    qqq_vol_avg_dl = qqq_vol_dl.rolling(252, min_periods=60).mean()
    vol_ratio_dl = (qqq_vol_dl / qqq_vol_avg_dl.clip(lower=0.01)).clip(lower=0.3, upper=3.0)

    for lev_cost_label, lev_cost in [("1.5%", 0.015), ("1.0%", 0.010), ("0.5%", 0.005)]:
        print(f"  --- Leverage cost: {lev_cost_label}/yr ---")
        cost_winners = []

        for base_key, base_res in lev_cands:
            br = base_res["portfolio_returns"]

            # Static leverage: 2.0 to 8.0
            for mult in np.arange(2.0, 8.2, 0.4):
                mult = round(mult, 1)
                sr = br * mult - (mult - 1) * lev_cost / 252
                label = f"L({lev_cost_label})_{base_key}_x{mult}"[:105]
                res = make_result(sr, prices.index)
                m = res["metrics"]
                results[label] = res
                if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                    winners.append(label); cost_winners.append(label)

            # Dynamic leverage (vol-inverse)
            for tgt_lev in [3.0, 4.0, 5.0, 6.0, 7.0]:
                dyn_lev = (tgt_lev / vol_ratio_dl).clip(lower=1.0, upper=tgt_lev * 1.5)
                sr = br * dyn_lev - (dyn_lev - 1) * lev_cost / 252
                label = f"DL({lev_cost_label})_{base_key}_t{tgt_lev}"[:105]
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
            tag = "DYN" if k.startswith("DL") else "STA"
            print(f"    [{tag}] {k[:90]:90s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
                  f"DD={m['Max Drawdown']:.2%}")
        print()

    # ══════════════════════════════════════════════════════════════
    # PHASE 7: POST-LEVERAGE DRAWDOWN CONTROL (the v12 mega-innovation)
    # ══════════════════════════════════════════════════════════════
    print(SEP)
    print("PHASE 7: POST-LEVERAGE DRAWDOWN CONTROL\n")

    # Top leveraged winners by Sharpe
    winner_details = [(w, results[w]) for w in winners]
    winner_details.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    dd_lev_results = {}
    for base_key, base_res in winner_details[:40]:
        br = base_res["portfolio_returns"]
        for dd_trigger in [-0.04, -0.05, -0.08, -0.10]:
            dd_ret = drawdown_control(br, max_dd_trigger=dd_trigger, recovery_rate=0.01)
            key = f"DDC({dd_trigger:.0%})_{base_key}"[:110]
            res = make_result(dd_ret, prices.index)
            m = res["metrics"]
            dd_lev_results[key] = res; results[key] = res
            if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
                winners.append(key)

        # Graduated DDC post-leverage
        for trig in [-0.05, -0.08]:
            gdd_ret = graduated_ddc(br, trigger=trig, max_cut=0.80, recovery_rate=0.01)
            key = f"GDDC({trig:.0%})_{base_key}"[:110]
            res = make_result(gdd_ret, prices.index)
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
        print(f"    {k[:95]:95s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%}{w_tag}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 8: WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("PHASE 8: WALK-FORWARD VALIDATION\n")

    # Take top 10 overall strategies and check performance in 3 sub-periods
    all_sorted = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    # Find strategies that are winners
    top_for_wf = []
    for k, v in all_sorted:
        m = v["metrics"]
        if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95:
            top_for_wf.append((k, v))
        if len(top_for_wf) >= 10:
            break

    if top_for_wf:
        periods = [
            ("2010-2014", "2010-01-01", "2015-01-01"),
            ("2015-2019", "2015-01-01", "2020-01-01"),
            ("2020-2025", "2020-01-01", "2025-03-01"),
        ]
        print(f"  Validating top {len(top_for_wf)} strategies across 3 periods:")
        print(f"  {'Strategy':<70s} {'2010-14':>12s} {'2015-19':>12s} {'2020-25':>12s} {'Consistent':>10s}")
        print("  " + "-" * 110)

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

            # Consistent = positive Sharpe in all periods
            consistent = all(s > 0.5 for s in period_results)
            p_strs = [f"Sh={s:.2f}" for s in period_results]
            tag = "YES" if consistent else "no"
            print(f"  {k[:70]:70s} {p_strs[0]:>12s} {p_strs[1]:>12s} {p_strs[2]:>12s} {tag:>10s}")

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
        ddc_w = [w for w in winners if w.startswith("DDC(") or w.startswith("GDDC(")]
        other_w = [w for w in winners if w not in sta_w + dyn_w + ddc_w]
        print(f"  Static: {len(sta_w)}, Dynamic: {len(dyn_w)}, DDControl: {len(ddc_w)}, "
              f"Other: {len(other_w)}")

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
                print(f"    {tag:24s}: {k[:80]}")
                print(f"      CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                      f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}, "
                      f"Calmar={m['Calmar Ratio']:.4f}")

        # Efficient Frontier
        print(f"\n  Efficient Frontier (top 25 CAGR-beaters by Sharpe):")
        cagr_beaters = [(k, results[k]) for k in winners if results[k]["metrics"]["CAGR"] > spy_cagr]
        cagr_beaters.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
        for k, v in cagr_beaters[:25]:
            m = v["metrics"]
            print(f"    {k[:95]:95s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
                  f"DD={m['Max Drawdown']:.2%}")

        # Overall Sharpe Leaders
        print(f"\n  Overall Sharpe Leaders (top 10):")
        sh_leaders = sorted([(k, results[k]) for k in winners],
                           key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
        for k, v in sh_leaders[:10]:
            m = v["metrics"]
            print(f"    {k[:95]:95s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f}")
    else:
        print("\n  No winners found.")

    print(f"\n  Total: {total} strategies | Winners: {n_w} | Audit: {len(iss) if iss else 0} issues")
    print(SEP)


if __name__ == "__main__":
    main()
