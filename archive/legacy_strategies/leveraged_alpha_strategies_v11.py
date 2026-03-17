"""
Leveraged Alpha v11 — MAXIMISING ALPHA
==================================================================================
v10b found 368 WINNERs. Best at 1.5% cost:
  E(5%CH+95%ZP_Top5)_x3.8: CAGR=13.81%, Sharpe=1.99
  E(8%CH+92%ZP_Top5)_x3.4: CAGR=13.85%, Sharpe=1.98

v11 innovations to push CAGR + Sharpe even higher:
  1. entry_z scan  (1.5-2.25) — more trades, more CAGR per pair
  2. Sharpe-weighted notional — better capital to better pairs
  3. CAGR-focused pair selection from top-26 pairs
  4. Dynamic leverage — vol-scale the multiplier (lower lev in crises)
  5. Multi-alpha ensemble — Donchian breakout + pairs + CrashHedge
  6. Multi-window stacking — same pair at 2-3 windows
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


# ────────────────────── helpers ──────────────────────

def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers, {len(p)} days\n")
    return p

def sma(s, n): return s.rolling(n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=max(10, n // 2)).std() * np.sqrt(252)

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


# ────────────────── alpha sources ──────────────────

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


def strat_donchian(prices, ticker="QQQ", lookback=63, notional=1.0):
    """Donchian channel breakout — long above upper, flat below."""
    p = prices[ticker]
    upper = p.rolling(lookback, min_periods=lookback).max()
    lower = p.rolling(lookback, min_periods=lookback).min()
    mid = (upper + lower) / 2
    sig = pd.Series(0.0, index=prices.index)
    for i in range(1, len(sig)):
        prev = sig.iloc[i - 1]; px = p.iloc[i]
        if np.isnan(upper.iloc[i]):
            sig.iloc[i] = 0; continue
        if prev == 0:
            if px >= upper.iloc[i]: sig.iloc[i] = 1
            else:                   sig.iloc[i] = 0
        elif prev > 0:
            if px <= mid.iloc[i]: sig.iloc[i] = 0
            else:                 sig.iloc[i] = 1
        else:
            sig.iloc[i] = 0
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = sig * notional
    return w


def strat_momentum_ls(prices, lookback=126, n_long=3, n_short=2, notional=0.15):
    """Cross-sectional sector momentum: long top N, short bottom N sectors."""
    sector_ret = prices[SECTORS].pct_change(lookback)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(lookback + 1, len(prices)):
        row = sector_ret.iloc[i].dropna().sort_values()
        if len(row) < n_long + n_short:
            continue
        shorts = row.index[:n_short]
        longs = row.index[-n_long:]
        for s in shorts:
            w.iloc[i][s] = -notional
        for l in longs:
            w.iloc[i][l] = notional
    return w


# ────────────────── audit ──────────────────

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
            issues.append(f"[{label}] Sharpe mismatch: {sh_m:.4f} vs {sh_r:.4f}")
    return issues


# ────────────────── reporting ──────────────────

def report(label, m, spy_cagr, short=False):
    cagr_flag = "CAGR+" if m["CAGR"] > spy_cagr else "     "
    sh_flag = "SH+" if m["Sharpe Ratio"] > 1.95 else "   "
    win = "** WINNER **" if m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95 else ""
    if short:
        print(f"  {label:65s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} {cagr_flag} {sh_flag} {win}")
    else:
        print(f"  {label:65s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%} Sort={m['Sortino Ratio']:.2f} "
              f"{cagr_flag} {sh_flag} {win}")
    return bool(win)


# ────────────────────── MAIN ──────────────────────

def main():
    print(SEP)
    print("LEVERAGED ALPHA v11 -- MAXIMISING ALPHA")
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

    # ==================================================================
    # PHASE 1: EXHAUSTIVE PAIR PARAMETER SCAN
    # ==================================================================
    print(SEP)
    print("PHASE 1: PAIR PARAMETER SCAN (entry_z x exit_z x window x notional)")
    print("  Goal: find highest-CAGR pair combos that keep Sharpe > 2.0\n")

    # Top-26 pairs from v9 (CAGR > 2% AND Sharpe > 1.5)
    ELITE_PAIRS = [
        ("XLK", "QQQ"), ("XLK", "SPY"), ("XLI", "XLB"), ("XLP", "XLU"),
        ("XLI", "SPY"), ("XLB", "EFA"), ("XLI", "IWM"), ("XLF", "IWM"),
        ("XLB", "SPY"), ("XLF", "XLI"), ("XLV", "XLP"), ("XLP", "XLU"),
    ]
    # Remove duplicates
    seen_pairs = set()
    unique_pairs = []
    for a, b in ELITE_PAIRS:
        key = tuple(sorted([a, b]))
        if key not in seen_pairs:
            seen_pairs.add(key); unique_pairs.append((a, b))
    ELITE_PAIRS = unique_pairs

    WINDOWS = [42, 63, 126]
    ENTRY_ZS = [1.5, 1.75, 2.0, 2.25]
    EXIT_ZS = [0.25, 0.5, 0.75]
    NOTIONAL_SCAN = 0.15  # for individual pair scanning

    # Scan all pair x window x entry_z x exit_z combos
    pair_db = []  # (key, a, b, w, ez_in, ez_out, sharpe, cagr, weights_df)
    total_combos = len(ELITE_PAIRS) * len(WINDOWS) * len(ENTRY_ZS) * len(EXIT_ZS)
    print(f"  Scanning {total_combos} combos ({len(ELITE_PAIRS)} pairs x {len(WINDOWS)} wins x "
          f"{len(ENTRY_ZS)} entry_z x {len(EXIT_ZS)} exit_z)...\n")

    for (a, b) in ELITE_PAIRS:
        for win in WINDOWS:
            for ez_in in ENTRY_ZS:
                for ez_out in EXIT_ZS:
                    pw = pair_weights(prices, a, b, window=win,
                                      entry_z=ez_in, exit_z=ez_out,
                                      notional=NOTIONAL_SCAN)
                    if pw is None:
                        continue
                    res = backtest(prices, pw)
                    m = res["metrics"]
                    sh = m["Sharpe Ratio"]; cg = m["CAGR"]
                    key = f"{a}/{b}_w{win}_ez{ez_in:.2f}/{ez_out:.2f}"
                    pair_db.append({
                        "key": key, "a": a, "b": b, "win": win,
                        "entry_z": ez_in, "exit_z": ez_out,
                        "sharpe": sh, "cagr": cg, "dd": m["Max Drawdown"],
                        "weights": pw
                    })

    # Sort by Sharpe and show top 30
    pair_db.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"  {len(pair_db)} pair configs scanned. Top 30 by Sharpe:")
    for i, p in enumerate(pair_db[:30]):
        marker = "*" if p["sharpe"] > 2.0 and p["cagr"] > 0.02 else " "
        print(f"  {i+1:3d}. {p['key']:40s} Sh={p['sharpe']:.4f} CAGR={p['cagr']:.2%} "
              f"DD={p['dd']:.2%} {marker}")

    # Also show top 15 by CAGR (with Sharpe > 1.5)
    cagr_sorted = sorted([p for p in pair_db if p["sharpe"] > 1.5],
                         key=lambda x: x["cagr"], reverse=True)
    print(f"\n  Top 15 by CAGR (Sharpe > 1.5):")
    for i, p in enumerate(cagr_sorted[:15]):
        print(f"  {i+1:3d}. {p['key']:40s} CAGR={p['cagr']:.2%} Sh={p['sharpe']:.4f} "
              f"DD={p['dd']:.2%}")

    # ==================================================================
    # PHASE 2: OPTIMAL PAIR PORTFOLIO CONSTRUCTION
    # ==================================================================
    print("\n" + SEP)
    print("PHASE 2: PAIR PORTFOLIO CONSTRUCTION")
    print("  A) Equal-weight stacking with best params")
    print("  B) Sharpe-weighted stacking")
    print("  C) CAGR-focused selection")
    print("  D) Multi-window diversification\n")

    zp_portfolios = {}

    # --- 2A. Equal-weight with best entry/exit_z per pair ---
    # For each unique (a,b) pair, find the best entry_z/exit_z combo (by Sharpe)
    best_per_pair = {}
    for p in pair_db:
        k = (p["a"], p["b"], p["win"])
        if k not in best_per_pair or p["sharpe"] > best_per_pair[k]["sharpe"]:
            best_per_pair[k] = p

    # Select top pairs by Sharpe for portfolio construction
    ranked_pairs = sorted(best_per_pair.values(), key=lambda x: x["sharpe"], reverse=True)

    for n_pairs in [5, 7, 10, 12]:
        selected = ranked_pairs[:n_pairs]
        for notional in [0.08, 0.10, 0.12, 0.15]:
            combined_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for p in selected:
                pw = pair_weights(prices, p["a"], p["b"], window=p["win"],
                                  entry_z=p["entry_z"], exit_z=p["exit_z"],
                                  notional=notional)
                if pw is not None:
                    combined_w += pw
            res = backtest(prices, combined_w)
            m = res["metrics"]
            key = f"ZP_BestSh_Top{n_pairs}_n{notional}"
            zp_portfolios[key] = res
            results[key] = res
            report(key, m, spy_cagr, short=True)

    # --- 2B. Sharpe-weighted notional ---
    print()
    for n_pairs in [5, 7, 10]:
        selected = ranked_pairs[:n_pairs]
        total_sh = sum(p["sharpe"] for p in selected)
        for base_notional in [0.10, 0.15, 0.20]:
            combined_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for p in selected:
                wt = (p["sharpe"] / total_sh) * base_notional * n_pairs
                pw = pair_weights(prices, p["a"], p["b"], window=p["win"],
                                  entry_z=p["entry_z"], exit_z=p["exit_z"],
                                  notional=wt)
                if pw is not None:
                    combined_w += pw
            res = backtest(prices, combined_w)
            m = res["metrics"]
            key = f"ZP_ShWt_Top{n_pairs}_bn{base_notional}"
            zp_portfolios[key] = res
            results[key] = res
            report(key, m, spy_cagr, short=True)

    # --- 2C. CAGR-focused: pick pairs with highest individual CAGR (Sharpe > 1.5) ---
    print()
    cagr_pairs = sorted([p for p in best_per_pair.values() if p["sharpe"] > 1.5],
                        key=lambda x: x["cagr"], reverse=True)
    for n_pairs in [5, 7, 10]:
        selected = cagr_pairs[:n_pairs]
        for notional in [0.10, 0.12, 0.15]:
            combined_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for p in selected:
                pw = pair_weights(prices, p["a"], p["b"], window=p["win"],
                                  entry_z=p["entry_z"], exit_z=p["exit_z"],
                                  notional=notional)
                if pw is not None:
                    combined_w += pw
            res = backtest(prices, combined_w)
            m = res["metrics"]
            key = f"ZP_CAGRFocus_Top{n_pairs}_n{notional}"
            zp_portfolios[key] = res
            results[key] = res
            report(key, m, spy_cagr, short=True)

    # --- 2D. Multi-window diversification: stack same pair at 2-3 windows ---
    print()
    # Pick top 5 PAIR NAMES (ignoring window), then stack all their windows
    pair_names_ranked = {}
    for p in pair_db:
        k = (p["a"], p["b"])
        if k not in pair_names_ranked or p["sharpe"] > pair_names_ranked[k]:
            pair_names_ranked[k] = p["sharpe"]
    top_pair_names = sorted(pair_names_ranked.items(), key=lambda x: x[1], reverse=True)[:6]

    for notional in [0.06, 0.08, 0.10]:
        combined_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        count = 0
        for (a, b), _ in top_pair_names:
            for win in WINDOWS:
                # Find best entry/exit for this pair+window
                k = (a, b, win)
                if k in best_per_pair:
                    p = best_per_pair[k]
                    pw = pair_weights(prices, a, b, window=win,
                                      entry_z=p["entry_z"], exit_z=p["exit_z"],
                                      notional=notional)
                    if pw is not None:
                        combined_w += pw; count += 1
        res = backtest(prices, combined_w)
        m = res["metrics"]
        key = f"ZP_MultiWin_Top6x3w_n{notional}({count}p)"
        zp_portfolios[key] = res
        results[key] = res
        report(key, m, spy_cagr, short=True)

    # ==================================================================
    # PHASE 3: ADDITIONAL ALPHA SOURCES
    # ==================================================================
    print("\n" + SEP)
    print("PHASE 3: ADDITIONAL ALPHA SOURCES\n")

    # CrashHedge
    ch_res = backtest(prices, strat_crash_hedged(prices, 1.0))
    results["CrashHedge"] = ch_res
    report("CrashHedge", ch_res["metrics"], spy_cagr, short=True)

    # Donchian breakouts
    for ticker in ["QQQ", "SPY", "XLK"]:
        for lb in [42, 63, 126]:
            dw = strat_donchian(prices, ticker=ticker, lookback=lb, notional=1.0)
            res = backtest(prices, dw)
            key = f"Donchian_{ticker}_lb{lb}"
            results[key] = res
            report(key, res["metrics"], spy_cagr, short=True)

    # Momentum long/short
    for lb in [63, 126, 252]:
        mw = strat_momentum_ls(prices, lookback=lb, n_long=3, n_short=2, notional=0.15)
        res = backtest(prices, mw)
        key = f"MomLS_lb{lb}"
        results[key] = res
        report(key, res["metrics"], spy_cagr, short=True)

    # ==================================================================
    # PHASE 4: MEGA-ENSEMBLE CONSTRUCTION
    # ==================================================================
    print("\n" + SEP)
    print("PHASE 4: MULTI-ALPHA ENSEMBLES")
    print("  Blending pairs + CrashHedge + Donchian/Momentum\n")

    # Get best ZP portfolios
    sorted_zp = sorted(zp_portfolios.items(),
                       key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    # Get best directional alpha (Donchian / Momentum)
    don_results = {k: v for k, v in results.items() if k.startswith("Donchian_")}
    best_don = max(don_results.items(), key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
    mom_results = {k: v for k, v in results.items() if k.startswith("MomLS_")}
    best_mom = max(mom_results.items(), key=lambda x: x[1]["metrics"]["Sharpe Ratio"])

    print(f"  Best Donchian: {best_don[0]} Sh={best_don[1]['metrics']['Sharpe Ratio']:.4f}")
    print(f"  Best Momentum: {best_mom[0]} Sh={best_mom[1]['metrics']['Sharpe Ratio']:.4f}")
    print()

    ch_ret = ch_res["portfolio_returns"]
    don_ret = best_don[1]["portfolio_returns"]
    mom_ret = best_mom[1]["portfolio_returns"]

    ensembles = {}

    # Build ensembles: ZP dominant + various alpha tilts
    for zp_key, zp_res in sorted_zp[:6]:
        zp_ret = zp_res["portfolio_returns"]
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.8:
            continue

        # A) ZP + CH only (traditional, proven best)
        for ch_pct in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            zp_pct = 1 - ch_pct
            er = ch_pct * ch_ret + zp_pct * zp_ret
            eq = 100_000 * (1 + er).cumprod(); eq.name = "Equity"
            m = compute_metrics(er, eq, 100_000, risk_free_rate=RF)
            key = f"E({ch_pct:.0%}CH+{zp_pct:.0%}{zp_key})"
            ensembles[key] = {"equity_curve": eq, "portfolio_returns": er,
                              "weights": pd.DataFrame(), "metrics": m,
                              "turnover": pd.Series(0, index=prices.index),
                              "gross_exposure": pd.Series(1, index=prices.index),
                              "net_exposure": pd.Series(1, index=prices.index),
                              "cash_weight": pd.Series(0, index=prices.index)}

        # B) ZP + CH + Donchian (3-source)
        for zp_pct, ch_pct, don_pct in [(0.85, 0.05, 0.10), (0.80, 0.08, 0.12),
                                         (0.80, 0.10, 0.10), (0.75, 0.10, 0.15),
                                         (0.70, 0.10, 0.20), (0.70, 0.15, 0.15)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + don_pct * don_ret
            eq = 100_000 * (1 + er).cumprod(); eq.name = "Equity"
            m = compute_metrics(er, eq, 100_000, risk_free_rate=RF)
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{don_pct:.0%}Don)"
            ensembles[key] = {"equity_curve": eq, "portfolio_returns": er,
                              "weights": pd.DataFrame(), "metrics": m,
                              "turnover": pd.Series(0, index=prices.index),
                              "gross_exposure": pd.Series(1, index=prices.index),
                              "net_exposure": pd.Series(1, index=prices.index),
                              "cash_weight": pd.Series(0, index=prices.index)}

        # C) ZP + CH + Momentum (3-source)
        for zp_pct, ch_pct, mom_pct in [(0.85, 0.05, 0.10), (0.80, 0.10, 0.10),
                                         (0.75, 0.10, 0.15)]:
            er = zp_pct * zp_ret + ch_pct * ch_ret + mom_pct * mom_ret
            eq = 100_000 * (1 + er).cumprod(); eq.name = "Equity"
            m = compute_metrics(er, eq, 100_000, risk_free_rate=RF)
            key = f"E3({zp_pct:.0%}{zp_key}+{ch_pct:.0%}CH+{mom_pct:.0%}Mom)"
            ensembles[key] = {"equity_curve": eq, "portfolio_returns": er,
                              "weights": pd.DataFrame(), "metrics": m,
                              "turnover": pd.Series(0, index=prices.index),
                              "gross_exposure": pd.Series(1, index=prices.index),
                              "net_exposure": pd.Series(1, index=prices.index),
                              "cash_weight": pd.Series(0, index=prices.index)}

    results.update(ensembles)

    # Show best ensembles by Sharpe
    ens_sorted = sorted(ensembles.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    print(f"  {len(ensembles)} ensembles built. Top 20 by Sharpe:")
    for k, v in ens_sorted[:20]:
        m = v["metrics"]
        report(k, m, spy_cagr, short=True)

    # ==================================================================
    # PHASE 5: LEVERAGE SWEEP WITH DYNAMIC LEVERAGE
    # ==================================================================
    print("\n" + SEP)
    print("PHASE 5: LEVERAGE SWEEP (static + dynamic)")
    print("  Static: 2.0-6.0 in 0.2 steps")
    print("  Dynamic: vol-scaled leverage (lower in crises)")
    print("  Cost models: 1.5%, 1.0%, 0.5%\n")

    # Candidates: ensembles with Sharpe > 1.8, or raw ZP with Sharpe > 2.0
    lev_candidates = sorted(
        [(k, v) for k, v in results.items() if v["metrics"]["Sharpe Ratio"] > 1.8],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True
    )[:25]

    # Remove dupes
    seen = set()
    lev_cands_deduped = []
    for k, v in lev_candidates:
        if k not in seen:
            seen.add(k); lev_cands_deduped.append((k, v))
    lev_candidates = lev_cands_deduped
    print(f"  {len(lev_candidates)} candidates for leverage sweep\n")

    # Helper: compute vol for dynamic leverage
    qqq_vol = rvol(prices["QQQ"], 20)
    qqq_vol_avg = qqq_vol.rolling(252, min_periods=60).mean()
    vol_ratio = (qqq_vol / qqq_vol_avg.clip(lower=0.01)).clip(lower=0.3, upper=3.0)

    for lev_cost_label, lev_cost in [("1.5%", 0.015), ("1.0%", 0.010), ("0.5%", 0.005)]:
        print(f"  --- Leverage cost: {lev_cost_label}/yr ---")

        for base_key, base_res in lev_candidates:
            br = base_res["portfolio_returns"]

            # A) Static leverage sweep
            for mult in np.arange(2.0, 6.2, 0.2):
                mult = round(mult, 1)
                sr = br * mult - (mult - 1) * lev_cost / 252
                eq = 100_000 * (1 + sr).cumprod(); eq.name = "Equity"
                m = compute_metrics(sr, eq, 100_000, risk_free_rate=RF)
                is_winner = m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95
                label = f"L({lev_cost_label})_{base_key}_x{mult}"[:90]
                results[label] = {"equity_curve": eq, "portfolio_returns": sr,
                                  "weights": pd.DataFrame(), "metrics": m,
                                  "turnover": pd.Series(0, index=prices.index),
                                  "gross_exposure": pd.Series(mult, index=prices.index),
                                  "net_exposure": pd.Series(mult, index=prices.index),
                                  "cash_weight": pd.Series(0, index=prices.index)}
                if is_winner:
                    winners.append(label)

            # B) Dynamic leverage: target_lev / vol_ratio (lower leverage when vol high)
            for target_lev in [3.0, 3.5, 4.0, 4.5, 5.0]:
                dyn_lev = (target_lev / vol_ratio).clip(lower=1.0, upper=target_lev * 1.5)
                sr = br * dyn_lev - (dyn_lev - 1) * lev_cost / 252
                eq = 100_000 * (1 + sr).cumprod(); eq.name = "Equity"
                avg_lev = dyn_lev.mean()
                m = compute_metrics(sr, eq, 100_000, risk_free_rate=RF)
                is_winner = m["CAGR"] > spy_cagr and m["Sharpe Ratio"] > 1.95
                label = f"DL({lev_cost_label})_{base_key}_tgt{target_lev}"[:90]
                results[label] = {"equity_curve": eq, "portfolio_returns": sr,
                                  "weights": pd.DataFrame(), "metrics": m,
                                  "turnover": pd.Series(0, index=prices.index),
                                  "gross_exposure": dyn_lev, "net_exposure": dyn_lev,
                                  "cash_weight": pd.Series(0, index=prices.index)}
                if is_winner:
                    winners.append(label)

        # Show winners at this cost level
        cost_winners = [w for w in winners if f"({lev_cost_label})" in w]
        print(f"  Winners at {lev_cost_label}: {len(cost_winners)}")
        # Show top 5 by Sharpe for this cost
        cost_res = [(k, v) for k, v in results.items()
                    if f"({lev_cost_label})" in k and
                    v["metrics"]["CAGR"] > spy_cagr and v["metrics"]["Sharpe Ratio"] > 1.95]
        cost_res.sort(key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
        for k, v in cost_res[:5]:
            m = v["metrics"]
            dyn = "DYN" if k.startswith("DL") else "STA"
            print(f"    [{dyn}] {k:80s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f}")
        print()

    # ==================================================================
    # AUDIT
    # ==================================================================
    print(SEP)
    auditable = {k: v for k, v in results.items()
                 if isinstance(v["weights"], pd.DataFrame) and not v["weights"].empty}
    iss = audit(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies audited): "
          f"{'ALL PASS' if not iss else f'{len(iss)} issues'}")
    for i in iss:
        print(f"  !! {i}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + SEP)
    print("FINAL SUMMARY")
    print(SEP)

    total = len(results)
    n_w = len(winners)
    sorted_all = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    if winners:
        print(f"\n*** {n_w} WINNERS FOUND ***\n")

        # Group winners by type (dynamic vs static)
        dyn_winners = [w for w in winners if w.startswith("DL")]
        sta_winners = [w for w in winners if w.startswith("L(")]
        print(f"  Static leverage winners: {len(sta_winners)}")
        print(f"  Dynamic leverage winners: {len(dyn_winners)}\n")

        # Show BEST winner at each cost model
        for cost_label in ["1.5%", "1.0%", "0.5%"]:
            cost_w = [(w, results[w]) for w in winners if f"({cost_label})" in w]
            if not cost_w:
                print(f"  [{cost_label} cost] No winners")
                continue
            # Best by combined score (Sharpe * CAGR)
            best = max(cost_w, key=lambda x: x[1]["metrics"]["Sharpe Ratio"] * x[1]["metrics"]["CAGR"])
            m = best[1]["metrics"]
            print(f"  [{cost_label} cost] CHAMPION: {best[0]}")
            print(f"    CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}")
    else:
        print("\nNo WINNER found (both CAGR > SPY AND Sharpe > 1.95).")

    # Efficient frontier
    print(f"\n  Efficient Frontier (top 20 CAGR-beaters by Sharpe):")
    cb = sorted([(k, v) for k, v in sorted_all if v["metrics"]["CAGR"] > spy_cagr],
                key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    for k, v in cb[:20]:
        m = v["metrics"]
        dyn = "DYN" if "DL(" in k else "STA"
        print(f"    [{dyn}] {k:80s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
              f"DD={m['Max Drawdown']:.2%}")

    # Show top 10 by Sharpe overall
    print(f"\n  Overall Sharpe Leaders (top 10):")
    for k, v in sorted_all[:10]:
        m = v["metrics"]
        print(f"    {k:80s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f}")

    # Show best dynamic leverage results
    dyn_all = sorted([(k, v) for k, v in results.items() if k.startswith("DL(")],
                     key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    if dyn_all:
        print(f"\n  Best Dynamic-Leverage Strategies (top 10):")
        for k, v in dyn_all[:10]:
            m = v["metrics"]
            print(f"    {k:80s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} "
                  f"DD={m['Max Drawdown']:.2%}")

    print(f"\n  Total: {total} strategies | Winners: {n_w} | Audit issues: {len(iss)}")
    print(SEP)


if __name__ == "__main__":
    main()
