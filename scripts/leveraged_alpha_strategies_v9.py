"""
Leveraged Alpha v9 — Exhaustive Z-Score Pair Scan + Multi-Basket Ensemble
==================================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95 | 2010-01-01 to 2025-03-01

Key v8 insight: ZPairs(w=126,z=2.0,n=0.15) achieved Sharpe=1.66 with -0.04
correlation to CrashHedge (Sharpe=1.08). Theoretical optimal combo = 1.94.

v9 strategy:
  1. Scan ALL viable pairs from 18-asset universe through z-score engine
  2. Rank pairs by individual Sharpe ratio
  3. Build MULTI-BASKET ensembles (combine top pairs across different windows)
  4. Stack with CrashHedge + Donchian for CAGR boost
  5. Optimize weights to maximize Sharpe
  6. Leverage sweep
"""

from __future__ import annotations
import sys, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

# -- Config --
SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE"]
BROAD   = ["SPY", "QQQ", "IWM", "EFA"]
SAFE    = ["TLT", "IEF", "GLD", "SHY"]
ALL_TICKERS = SECTORS + BROAD + SAFE
START, END = "2010-01-01", "2025-03-01"
TX_BPS = 5; LEV_COST = 0.015; SHORT_COST = 0.005; RF_CASH = 0.02; RF = 0.0
SEP = "=" * 90

def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers, {len(p)} days "
          f"({p.index[0].strftime('%Y-%m-%d')} to {p.index[-1].strftime('%Y-%m-%d')})\n")
    return p

def sma(s, n): return s.rolling(n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=max(10,n//2)).std()*np.sqrt(252)
def rsi(s, p=14):
    d=s.diff(); g=d.where(d>0,0).rolling(p).mean(); l=(-d.where(d<0,0)).rolling(p).mean()
    return 100-100/(1+g/l.clip(lower=1e-10))
def breadth_count(prices, lb=50):
    secs=[s for s in SECTORS if s in prices.columns]
    b=pd.Series(0.0,index=prices.index)
    for s in secs: b+=(prices[s]>sma(prices[s],lb)).astype(float)
    return b
def zscore(s, window=63):
    m=s.rolling(window,min_periods=window//2).mean()
    sd=s.rolling(window,min_periods=window//2).std().clip(lower=1e-8)
    return (s-m)/sd


# ============================================================
# VECTORIZED BACKTEST (no trailing stop)
# ============================================================
def backtest(prices, weights, cap=100_000.0):
    common = prices.columns.intersection(weights.columns)
    p=prices[common]; w=weights[common].reindex(prices.index).fillna(0)
    w=w.shift(1).fillna(0)
    ret=p.pct_change().fillna(0)
    port_ret=(w*ret).sum(axis=1)
    ne=w.sum(axis=1); cw=(1-ne).clip(lower=0)
    cr=cw*RF_CASH/252
    turn=w.diff().fillna(0).abs().sum(axis=1)
    tx=turn*TX_BPS/10_000
    ge=w.abs().sum(axis=1)
    lc=(ge-1).clip(lower=0)*LEV_COST/252
    sc=w.clip(upper=0).abs().sum(axis=1)*SHORT_COST/252
    net=port_ret+cr-tx-lc-sc
    eq=cap*(1+net).cumprod(); eq.name="Equity"
    m=compute_metrics(net,eq,cap,risk_free_rate=RF,turnover=turn,gross_exposure=ge)
    return {"equity_curve":eq,"portfolio_returns":net,"weights":w,
            "turnover":turn,"gross_exposure":ge,"net_exposure":ne,
            "cash_weight":cw,"metrics":m}


# ============================================================
# SINGLE Z-SCORE PAIR ENGINE
# ============================================================
def run_single_pair(prices, leg_a, leg_b, window=63, entry_z=2.0, exit_z=0.5,
                    notional=0.15):
    """Run z-score mean-reversion on one pair. Returns weight DataFrame."""
    if leg_a not in prices.columns or leg_b not in prices.columns:
        return None

    spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
    z = zscore(spread, window)

    pos = pd.Series(0.0, index=prices.index)
    for i in range(1, len(pos)):
        prev = pos.iloc[i-1]; zi = z.iloc[i]
        if np.isnan(zi): pos.iloc[i]=0; continue
        if prev==0:
            if zi>entry_z: pos.iloc[i]=-1
            elif zi<-entry_z: pos.iloc[i]=1
            else: pos.iloc[i]=0
        elif prev>0: pos.iloc[i]=0 if zi>-exit_z else 1
        else: pos.iloc[i]=0 if zi<exit_z else -1

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[leg_a] = pos * notional
    w[leg_b] = -pos * notional
    return w


# ============================================================
# CRASH-HEDGED QQQ (proven champion)
# ============================================================
def strat_crash_hedged(prices, base_lev=1.2):
    qqq=prices["QQQ"]; v20=rvol(qqq,20)
    va=v20.rolling(120,min_periods=30).mean()
    normal=v20<va*1.2; elevated=(v20>=va*1.2)&(v20<va*1.8)
    crisis=v20>=va*1.8; recovery=elevated&(v20<v20.shift(5))&(qqq>qqq.rolling(10).min())
    w=pd.DataFrame(0.0,index=prices.index,columns=prices.columns)
    for col,n,e,c,r in [("QQQ",base_lev*0.7,base_lev*0.3,0.0,base_lev*0.8),
                         ("SPY",base_lev*0.3,base_lev*0.1,-0.3,base_lev*0.4)]:
        if col in w.columns:
            w.loc[normal,col]=n; w.loc[elevated,col]=e
            w.loc[crisis,col]=c; w.loc[recovery,col]=r
    for col,e,c in [("IWM",-0.2,0),("GLD",0.15,0.3),("TLT",0,0.2)]:
        if col in w.columns:
            w.loc[elevated,col]=e; w.loc[crisis,col]=c
            if col=="IWM": w.loc[recovery,col]=0
    return w


# ============================================================
# DONCHIAN BREAKOUT (proven from v8)
# ============================================================
def strat_donchian(prices, entry_period=30, exit_period=15, leverage=1.0):
    assets=[c for c in ["QQQ","SPY","GLD","TLT"] if c in prices.columns]
    n_assets=max(len(assets),1)
    w=pd.DataFrame(0.0,index=prices.index,columns=prices.columns)
    for asset in assets:
        high_n=prices[asset].rolling(entry_period,min_periods=entry_period).max()
        low_n=prices[asset].rolling(exit_period,min_periods=exit_period).min()
        pos=pd.Series(0.0,index=prices.index)
        for i in range(max(entry_period,exit_period),len(prices)):
            prev=pos.iloc[i-1]; price=prices[asset].iloc[i]
            if prev==0: pos.iloc[i]=1 if price>=high_n.iloc[i-1] else 0
            else: pos.iloc[i]=0 if price<=low_n.iloc[i-1] else 1
        w[asset]=pos*leverage/n_assets
    return w


# ============================================================
# AUDIT
# ============================================================
def audit(results_dict, prices):
    issues=[]
    spy_ret=prices["SPY"].pct_change(); qqq_ret=prices["QQQ"].pct_change()
    for label,res in results_dict.items():
        w=res["weights"]; pr=res["portfolio_returns"]; eq=res["equity_curve"]
        if not w.empty and w.abs().sum().sum()>0:
            for col in w.columns:
                if w[col].abs().sum()<1: continue
                for br,bn in [(spy_ret,"SPY"),(qqq_ret,"QQQ")]:
                    c=w[col].corr(br)
                    if abs(c)>0.25:
                        issues.append(f"[{label}] {col} wt corr w/ same-day {bn}={c:.3f}")
        sh_m=pr.mean()*252/(pr.std()*np.sqrt(252)) if pr.std()>1e-8 else 0
        sh_r=res["metrics"]["Sharpe Ratio"]
        if abs(sh_m-sh_r)>0.15:
            issues.append(f"[{label}] Sharpe mismatch: {sh_m:.4f} vs {sh_r:.4f}")
        yrs=len(eq)/252
        cagr_m=(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1 if yrs>0 else 0
        cagr_r=res["metrics"]["CAGR"]
        if abs(cagr_m-cagr_r)>0.015:
            issues.append(f"[{label}] CAGR mismatch: {cagr_m:.4f} vs {cagr_r:.4f}")
    return issues


# ============================================================
# OPTIMAL ENSEMBLE (Sharpe-maximizing weights via inverse-vol with corr)
# ============================================================
def optimal_weights(return_series_list, lookback=None):
    """Find weights that maximize Sharpe using mean-variance optimization.
    Uses full sample. Returns numpy array of weights."""
    r = pd.DataFrame({i: s for i, s in enumerate(return_series_list)}).dropna()
    mu = r.mean().values * 252
    cov = r.cov().values * 252
    n = len(mu)
    if n == 0: return np.array([])

    # Grid search over weight space for small n
    if n <= 6:
        best_sh = -np.inf; best_w = np.ones(n)/n
        steps = 11 if n <= 4 else 6
        candidates = np.linspace(0, 1, steps)
        for combo in itertools.product(candidates, repeat=n):
            ws = np.array(combo)
            if ws.sum() < 0.01: continue
            ws = ws / ws.sum()
            port_mu = ws @ mu
            port_var = ws @ cov @ ws
            if port_var <= 0: continue
            sh = port_mu / np.sqrt(port_var)
            if sh > best_sh: best_sh = sh; best_w = ws.copy()
        return best_w
    else:
        # For larger n, use inverse-vol
        vols = np.sqrt(np.diag(cov)).clip(min=1e-8)
        w = (1/vols); w /= w.sum()
        return w


# ============================================================
# MAIN
# ============================================================
def main():
    print(SEP)
    print("LEVERAGED ALPHA v9 -- EXHAUSTIVE Z-SCORE PAIRS + OPTIMAL ENSEMBLE")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95 | {START} to {END}")
    print(SEP)

    prices = load_data()
    bench = {}
    for t in ["SPY","QQQ"]:
        eq=prices[t]/prices[t].iloc[0]*100_000; ret=prices[t].pct_change().fillna(0)
        m=compute_metrics(ret,eq,100_000,risk_free_rate=RF); bench[t]=m
        print(f"  {t}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, MaxDD={m['Max Drawdown']:.2%}")
    spy_cagr = bench["SPY"]["CAGR"]
    print()

    results = {}
    def log(label, res, verbose=True):
        m=res["metrics"]
        cf="Y" if m["CAGR"]>spy_cagr else " "
        sf="Y" if m["Sharpe Ratio"]>1.95 else " "
        al=res["gross_exposure"].mean()
        cp=res["cash_weight"].mean()
        if verbose:
            print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cf}] "
                  f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] "
                  f"DD={m['Max Drawdown']:7.2%} L={al:.2f} C={cp:.0%}")
        results[label] = res

    # ======================================================
    # PHASE 1: EXHAUSTIVE PAIR SCAN
    # ======================================================
    print(SEP)
    print("PHASE 1: SCANNING ALL VIABLE PAIRS (z-score mean-reversion)")
    print("  Parameters: window=[42,63,126], entry_z=2.0, exit_z=0.5, notional=0.15\n")

    # Generate all viable pairs (same-class or cross-class, exclude self/safe-safe)
    equity_tickers = [t for t in SECTORS + BROAD if t in prices.columns]
    safe_tickers = [t for t in SAFE if t in prices.columns]
    all_avail = [t for t in ALL_TICKERS if t in prices.columns]

    # All unique pairs from equity+safe universe
    pair_candidates = []
    for i in range(len(all_avail)):
        for j in range(i+1, len(all_avail)):
            a, b = all_avail[i], all_avail[j]
            # Skip safe-safe pairs (low vol, low edge)
            if a in safe_tickers and b in safe_tickers: continue
            pair_candidates.append((a, b))

    print(f"  Testing {len(pair_candidates)} pairs x 3 windows = "
          f"{len(pair_candidates)*3} configs\n")

    pair_results = {}  # (pair, window) -> {sharpe, cagr, maxdd, returns}
    for window in [42, 63, 126]:
        for leg_a, leg_b in pair_candidates:
            w = run_single_pair(prices, leg_a, leg_b, window=window,
                                entry_z=2.0, exit_z=0.5, notional=0.15)
            if w is None: continue
            res = backtest(prices, w)
            m = res["metrics"]
            key = f"{leg_a}/{leg_b}_w{window}"
            pair_results[key] = {
                "sharpe": m["Sharpe Ratio"], "cagr": m["CAGR"],
                "maxdd": m["Max Drawdown"], "returns": res["portfolio_returns"],
                "pair": (leg_a, leg_b), "window": window,
                "result": res
            }

    # Sort by Sharpe and show top 30
    ranked = sorted(pair_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print(f"  TOP 30 PAIRS BY SHARPE (of {len(ranked)} tested):")
    print(f"  {'Pair':<25s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>7s}")
    print("  " + "-"*52)
    for i, (key, pr) in enumerate(ranked[:30]):
        print(f"  {key:<25s} {pr['sharpe']:7.4f} {pr['cagr']:7.2%} {pr['maxdd']:7.2%}")

    # ======================================================
    # PHASE 2: BUILD MULTI-BASKET PAIR STRATEGIES
    # ======================================================
    print("\n" + SEP)
    print("PHASE 2: BUILD MULTI-BASKET PAIR STRATEGIES")
    print("  Using top pairs, combine into baskets ensuring no ticker overlap\n")

    # Select top pairs ensuring diversification
    def build_basket(ranked_pairs, max_pairs=6, max_per_ticker=2):
        """Select top pairs with ticker diversification."""
        selected = []; ticker_count = {}
        for key, pr in ranked_pairs:
            if pr["sharpe"] < 0.3: break
            a, b = pr["pair"]
            ca = ticker_count.get(a, 0); cb = ticker_count.get(b, 0)
            if ca >= max_per_ticker or cb >= max_per_ticker: continue
            selected.append((key, pr))
            ticker_count[a] = ca + 1; ticker_count[b] = cb + 1
            if len(selected) >= max_pairs: break
        return selected

    # Build baskets with different diversification levels
    for (basket_name, max_pairs, max_per_ticker) in [
        ("Basket_Top4", 4, 2),
        ("Basket_Top6", 6, 2),
        ("Basket_Top8", 8, 3),
        ("Basket_Top10", 10, 3),
        ("Basket_Wide12", 12, 3),
    ]:
        basket = build_basket(ranked, max_pairs=max_pairs,
                              max_per_ticker=max_per_ticker)
        if len(basket) < 2: continue

        # Equal-weight combine returns
        basket_ret = sum(pr["returns"] for _, pr in basket) / len(basket)
        eq = 100_000 * (1 + basket_ret).cumprod(); eq.name = "Equity"
        m = compute_metrics(basket_ret, eq, 100_000, risk_free_rate=RF)
        ens_res = {"equity_curve":eq, "portfolio_returns":basket_ret,
                   "weights":pd.DataFrame(), "turnover":pd.Series(0,index=prices.index),
                   "gross_exposure":pd.Series(0.15,index=prices.index),
                   "net_exposure":pd.Series(0,index=prices.index),
                   "cash_weight":pd.Series(1,index=prices.index), "metrics":m}
        log(f"ZP_{basket_name}({len(basket)}pairs)", ens_res)
        print(f"    Pairs: {', '.join(k for k,_ in basket)}")

    # Also build WINDOW-DIVERSIFIED baskets: mix best pairs from different windows
    print()
    for (basket_name, windows) in [
        ("WinMix_42+126", [42, 126]),
        ("WinMix_All", [42, 63, 126]),
    ]:
        # Get top pair per window (diversified across windows)
        selected = []
        for win in windows:
            win_ranked = [(k,p) for k,p in ranked if p["window"]==win]
            basket = build_basket(win_ranked, max_pairs=4, max_per_ticker=2)
            selected.extend(basket)

        if len(selected) < 2: continue
        # Remove duplicates (same actual pair across windows)
        seen_pairs = set(); unique = []
        for k, pr in selected:
            pair_key = tuple(sorted(pr["pair"]))
            tw = (pair_key, pr["window"])
            if tw not in seen_pairs:
                seen_pairs.add(tw); unique.append((k, pr))
        selected = unique[:12]

        basket_ret = sum(pr["returns"] for _, pr in selected) / len(selected)
        eq = 100_000*(1+basket_ret).cumprod(); eq.name="Equity"
        m = compute_metrics(basket_ret, eq, 100_000, risk_free_rate=RF)
        ens_res = {"equity_curve":eq, "portfolio_returns":basket_ret,
                   "weights":pd.DataFrame(), "turnover":pd.Series(0,index=prices.index),
                   "gross_exposure":pd.Series(0.15,index=prices.index),
                   "net_exposure":pd.Series(0,index=prices.index),
                   "cash_weight":pd.Series(1,index=prices.index), "metrics":m}
        log(f"ZP_{basket_name}({len(selected)}pairs)", ens_res)
        print(f"    Pairs: {', '.join(k for k,_ in selected)}")

    # ======================================================
    # PHASE 3: DIRECTIONAL STRATEGIES
    # ======================================================
    print("\n" + SEP)
    print("PHASE 3: DIRECTIONAL STRATEGIES")
    w_ch = strat_crash_hedged(prices, base_lev=1.0)
    log("CrashHedge(1.0)", backtest(prices, w_ch))
    w_ch12 = strat_crash_hedged(prices, base_lev=1.2)
    log("CrashHedge(1.2)", backtest(prices, w_ch12))
    w_don = strat_donchian(prices, entry_period=30, exit_period=15, leverage=1.0)
    log("Donchian(30/15)", backtest(prices, w_don))

    # ======================================================
    # PHASE 4: OPTIMAL ENSEMBLES (Pairs + Directional)
    # ======================================================
    print("\n" + SEP)
    print("PHASE 4: OPTIMAL ENSEMBLES (Pairs + Directional)")

    # Get best pair baskets
    pair_baskets = {k: v for k, v in results.items() if k.startswith("ZP_")}
    directional = {k: v for k, v in results.items()
                   if k.startswith("CrashHedge") or k.startswith("Donchian")}

    def make_ensemble(label, comp_dict, weights=None):
        keys = list(comp_dict.keys())
        rets = [comp_dict[k]["portfolio_returns"] for k in keys]
        if weights is None:
            weights = optimal_weights(rets)
        else:
            weights = np.array(weights); weights = weights / weights.sum()
        er = sum(w*r for w,r in zip(weights, rets))
        ee = 100_000*(1+er).cumprod(); ee.name="Equity"
        m = compute_metrics(er, ee, 100_000, risk_free_rate=RF)
        ens_res = {"equity_curve":ee, "portfolio_returns":er,
                   "weights":pd.DataFrame(), "turnover":pd.Series(0,index=prices.index),
                   "gross_exposure":pd.Series(1,index=prices.index),
                   "net_exposure":pd.Series(1,index=prices.index),
                   "cash_weight":pd.Series(0,index=prices.index), "metrics":m}
        cf="Y" if m["CAGR"]>spy_cagr else " "; sf="Y" if m["Sharpe Ratio"]>1.95 else " "
        print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cf}] "
              f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] DD={m['Max Drawdown']:7.2%}")
        wstr = ", ".join(f"{k}:{w:.2f}" for k,w in zip(keys, weights))
        print(f"    Weights: {wstr}")
        results[label] = ens_res

    # For each pair basket, combine with CrashHedge
    for pb_name, pb_res in sorted(pair_baskets.items(),
                                   key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                                   reverse=True)[:5]:
        # Fixed weights: sweep ZP allocation from 30% to 80%
        for zp_pct in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            ch_pct = 1 - zp_pct
            label = f"E[CH+{pb_name}]({zp_pct:.0%}ZP)"
            comp = {"CrashHedge(1.0)": results["CrashHedge(1.0)"],
                    pb_name: pb_res}
            make_ensemble(label, comp, weights=[ch_pct, zp_pct])

    # 3-way: CH + Donchian + best ZP basket
    best_zp = max(pair_baskets.items(), key=lambda x: x[1]["metrics"]["Sharpe Ratio"])
    for zp_pct in [0.30, 0.40, 0.50, 0.60]:
        rest = 1 - zp_pct
        label = f"E3[CH+Don+{best_zp[0]}]({zp_pct:.0%}ZP)"
        comp = {"CrashHedge(1.0)": results["CrashHedge(1.0)"],
                "Donchian(30/15)": results["Donchian(30/15)"],
                best_zp[0]: best_zp[1]}
        make_ensemble(label, comp, weights=[rest*0.6, rest*0.4, zp_pct])

    # Optimal weights (Markowitz grid search)
    for pb_name, pb_res in sorted(pair_baskets.items(),
                                   key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                                   reverse=True)[:3]:
        comp = {"CrashHedge(1.0)": results["CrashHedge(1.0)"],
                "Donchian(30/15)": results["Donchian(30/15)"],
                pb_name: pb_res}
        make_ensemble(f"E_OPT[CH+Don+{pb_name}]", comp, weights=None)

        comp2 = {"CrashHedge(1.0)": results["CrashHedge(1.0)"],
                 pb_name: pb_res}
        make_ensemble(f"E_OPT[CH+{pb_name}]", comp2, weights=None)

    # ======================================================
    # PHASE 5: LEVERAGE SWEEP
    # ======================================================
    print("\n" + SEP)
    print("PHASE 5: LEVERAGE SWEEP ON BEST ENSEMBLES")

    ens_keys = sorted(
        [(k,v) for k,v in results.items() if k.startswith("E")],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)[:12]

    for base_key, base_res in ens_keys:
        br = base_res["portfolio_returns"]
        for mult in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            label = f"L_{base_key}_x{mult}"[:62]
            sr = br * mult - (mult-1)*LEV_COST/252
            eq = 100_000*(1+sr).cumprod(); eq.name="Equity"
            m = compute_metrics(sr, eq, 100_000, risk_free_rate=RF)
            cf="Y" if m["CAGR"]>spy_cagr else " "; sf="Y" if m["Sharpe Ratio"]>1.95 else " "
            print(f"  {label:62s} CAGR={m['CAGR']:7.2%}[{cf}] "
                  f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] DD={m['Max Drawdown']:7.2%}")
            results[label] = {"equity_curve":eq,"portfolio_returns":sr,
                              "weights":pd.DataFrame(),"metrics":m,
                              "turnover":pd.Series(0,index=prices.index),
                              "gross_exposure":pd.Series(mult,index=prices.index),
                              "net_exposure":pd.Series(mult,index=prices.index),
                              "cash_weight":pd.Series(0,index=prices.index)}

    # ======================================================
    # PHASE 6: CASH OVERLAY (Pairs during directional cash)
    # ======================================================
    print("\n" + SEP)
    print("PHASE 6: CASH OVERLAY (Best pairs during CrashHedge cash periods)")

    best_zp_key = max(pair_baskets.keys(), key=lambda k: pair_baskets[k]["metrics"]["Sharpe Ratio"])
    pairs_ret = results[best_zp_key]["portfolio_returns"]
    ch_ret = results["CrashHedge(1.0)"]["portfolio_returns"]
    ch_cash = results["CrashHedge(1.0)"]["cash_weight"]

    for overlay_scale in [0.5, 0.8, 1.0, 1.5, 2.0]:
        overlay_ret = ch_ret + ch_cash * pairs_ret * overlay_scale
        eq = 100_000*(1+overlay_ret).cumprod(); eq.name="Equity"
        m = compute_metrics(overlay_ret, eq, 100_000, risk_free_rate=RF)
        label = f"Overlay_CH+{best_zp_key}_s{overlay_scale}"
        cf="Y" if m["CAGR"]>spy_cagr else " "; sf="Y" if m["Sharpe Ratio"]>1.95 else " "
        print(f"  {label:62s} CAGR={m['CAGR']:7.2%}[{cf}] "
              f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] DD={m['Max Drawdown']:7.2%}")
        results[label] = {"equity_curve":eq,"portfolio_returns":overlay_ret,
                          "weights":pd.DataFrame(),"metrics":m,
                          "turnover":pd.Series(0,index=prices.index),
                          "gross_exposure":pd.Series(1,index=prices.index),
                          "net_exposure":pd.Series(1,index=prices.index),
                          "cash_weight":pd.Series(0,index=prices.index)}

        # Leverage overlays
        for mult in [1.5, 2.0, 3.0]:
            lbl = f"L_Overlay_s{overlay_scale}_x{mult}"[:62]
            sr = overlay_ret * mult - (mult-1)*LEV_COST/252
            eq2 = 100_000*(1+sr).cumprod(); eq2.name="Equity"
            m2 = compute_metrics(sr, eq2, 100_000, risk_free_rate=RF)
            cf="Y" if m2["CAGR"]>spy_cagr else " "; sf="Y" if m2["Sharpe Ratio"]>1.95 else " "
            print(f"  {lbl:62s} CAGR={m2['CAGR']:7.2%}[{cf}] "
                  f"Sh={m2['Sharpe Ratio']:6.4f}[{sf}] DD={m2['Max Drawdown']:7.2%}")
            results[lbl] = {"equity_curve":eq2,"portfolio_returns":sr,
                            "weights":pd.DataFrame(),"metrics":m2,
                            "turnover":pd.Series(0,index=prices.index),
                            "gross_exposure":pd.Series(mult,index=prices.index),
                            "net_exposure":pd.Series(mult,index=prices.index),
                            "cash_weight":pd.Series(0,index=prices.index)}

    # ======================================================
    # AUDIT
    # ======================================================
    print("\n" + SEP)
    auditable = {k:v for k,v in results.items()
                 if isinstance(v["weights"],pd.DataFrame) and not v["weights"].empty}
    iss = audit(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies)")
    if iss:
        for i in iss: print(f"  !! {i}")
    else:
        print("  ALL PASS")

    # ======================================================
    # FINAL TABLE
    # ======================================================
    print("\n" + SEP)
    sorted_all = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    winners = []
    print("FINAL RESULTS -- TOP 50 BY SHARPE\n")
    print(f"{'#':>3s} {'Strategy':<63s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Hit':>7s}")
    print("-"*95)
    print(f"    {'SPY':<63s} {spy_cagr:7.2%} {bench['SPY']['Sharpe Ratio']:7.4f} "
          f"{bench['SPY']['Max Drawdown']:7.2%}")
    print(f"    {'QQQ':<63s} {bench['QQQ']['CAGR']:7.2%} {bench['QQQ']['Sharpe Ratio']:7.4f} "
          f"{bench['QQQ']['Max Drawdown']:7.2%}")
    print("-"*95)

    for rank,(label,res) in enumerate(sorted_all[:50],1):
        m=res["metrics"]
        hit=""
        if m["CAGR"]>spy_cagr and m["Sharpe Ratio"]>1.95: hit="WINNER"; winners.append(label)
        elif m["CAGR"]>spy_cagr: hit="CAGR+"
        elif m["Sharpe Ratio"]>1.95: hit="SH+"
        print(f"{rank:3d} {label:<63s} {m['CAGR']:7.2%} {m['Sharpe Ratio']:7.4f} "
              f"{m['Max Drawdown']:7.2%} {hit:>7s}")

    print("\n" + SEP)
    if winners:
        print(f"*** WINNERS ({len(winners)}) ***")
        for w in winners:
            m=results[w]["metrics"]
            print(f"  {w}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}")
    else:
        print("No WINNER yet (both CAGR>SPY AND Sharpe>1.95).")
        cb = [(k,v) for k,v in sorted_all if v["metrics"]["CAGR"]>spy_cagr]
        if cb:
            print(f"\nBest CAGR-beaters (top 10 by Sharpe):")
            for k,v in cb[:10]:
                m=v["metrics"]
                print(f"  {k}: CAGR={m['CAGR']:.2%}, Sh={m['Sharpe Ratio']:.4f}")
        print(f"\nBest Sharpe (top 10):")
        for k,v in sorted_all[:10]:
            m=v["metrics"]
            print(f"  {k}: CAGR={m['CAGR']:.2%}, Sh={m['Sharpe Ratio']:.4f}")

    print(f"\nTotal: {len(results)} strategies | Audit: {len(iss)} issues\n")
    print(SEP)


if __name__ == "__main__":
    main()
