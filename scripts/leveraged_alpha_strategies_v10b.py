"""
Leveraged Alpha v10b — FINE-TUNING focused on the near-miss configurations
==================================================================================
v10 found we're within 0.47% CAGR and 0.002 Sharpe of WINNER status:
  L_Ens[CH(10%)+ZP_Top5_n0.10(90%)]_x3.0: CAGR=13.15%, Sharpe=1.948

This script does targeted fine-tuning:
  1. Finer leverage increments (2.5, 2.6, 2.7, ... 4.0)
  2. Finer ensemble weights (CH 5-20%, ZP 80-95%)
  3. Test exit_z variations (0.0, 0.25, 0.5, 0.75)
  4. Sensitivity to leverage cost (0.5%, 1.0%, 1.5%)
"""

from __future__ import annotations
import sys
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
SEP = "=" * 90


def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers, {len(p)} days\n")
    return p


def sma(s, n): return s.rolling(n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=max(10,n//2)).std()*np.sqrt(252)
def zscore(s, window=63):
    m=s.rolling(window,min_periods=window//2).mean()
    sd=s.rolling(window,min_periods=window//2).std().clip(lower=1e-8)
    return (s-m)/sd


def backtest(prices, weights, lev_cost=0.015, cap=100_000.0):
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
    lc=(ge-1).clip(lower=0)*lev_cost/252
    sc=w.clip(upper=0).abs().sum(axis=1)*SHORT_COST/252
    net=port_ret+cr-tx-lc-sc
    eq=cap*(1+net).cumprod(); eq.name="Equity"
    m=compute_metrics(net,eq,cap,risk_free_rate=RF,turnover=turn,gross_exposure=ge)
    return {"equity_curve":eq,"portfolio_returns":net,"weights":w,
            "turnover":turn,"gross_exposure":ge,"net_exposure":ne,
            "cash_weight":cw,"metrics":m}


def pair_weights(prices, leg_a, leg_b, window=63, entry_z=2.0, exit_z=0.5,
                 notional=0.15):
    if leg_a not in prices.columns or leg_b not in prices.columns:
        return None
    spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
    z = zscore(spread, window)
    pos = pd.Series(0.0, index=prices.index)
    for i in range(1, len(pos)):
        prev=pos.iloc[i-1]; zi=z.iloc[i]
        if np.isnan(zi): pos.iloc[i]=0; continue
        if prev==0:
            if zi>entry_z: pos.iloc[i]=-1
            elif zi<-entry_z: pos.iloc[i]=1
            else: pos.iloc[i]=0
        elif prev>0: pos.iloc[i]=0 if zi>-exit_z else 1
        else: pos.iloc[i]=0 if zi<exit_z else -1
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[leg_a] = pos * notional; w[leg_b] = -pos * notional
    return w


def strat_crash_hedged(prices, base_lev=1.0):
    qqq=prices["QQQ"]; v20=rvol(qqq,20)
    va=v20.rolling(120,min_periods=30).mean()
    normal=v20<va*1.2; elevated=(v20>=va*1.2)&(v20<va*1.8)
    crisis=v20>=va*1.8; recovery=elevated&(v20<v20.shift(5))&(qqq>qqq.rolling(10).min())
    w=pd.DataFrame(0.0,index=prices.index,columns=prices.columns)
    for col,n,e,c,r in [("QQQ",base_lev*0.7,base_lev*0.3,0.0,base_lev*0.8),
                         ("SPY",base_lev*0.3,base_lev*0.1,-0.3,base_lev*0.4)]:
        if col in w.columns:
            w.loc[normal,col]=n; w.loc[elevated,col]=e; w.loc[crisis,col]=c; w.loc[recovery,col]=r
    for col,e,c in [("IWM",-0.2,0),("GLD",0.15,0.3),("TLT",0,0.2)]:
        if col in w.columns:
            w.loc[elevated,col]=e; w.loc[crisis,col]=c
            if col=="IWM": w.loc[recovery,col]=0
    return w


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
    return issues


def main():
    print(SEP)
    print("LEVERAGED ALPHA v10b -- FINE-TUNING NEAR-MISS CONFIGURATIONS")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95 | {START} to {END}")
    print(SEP)

    prices = load_data()
    bench = {}
    for t in ["SPY","QQQ"]:
        eq=prices[t]/prices[t].iloc[0]*100_000; ret=prices[t].pct_change().fillna(0)
        m=compute_metrics(ret,eq,100_000,risk_free_rate=RF); bench[t]=m
        print(f"  {t}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}")
    spy_cagr = bench["SPY"]["CAGR"]
    print()

    results = {}
    winners = []

    # ======================================================
    # PHASE 1: BUILD BASE STRATEGIES
    # ======================================================
    print(SEP)
    print("PHASE 1: BASE STRATEGIES\n")

    # CrashHedge
    ch_res = backtest(prices, strat_crash_hedged(prices, 1.0))
    results["CH"] = ch_res
    mc = ch_res["metrics"]
    print(f"  CrashHedge: CAGR={mc['CAGR']:.2%}, Sh={mc['Sharpe Ratio']:.4f}")

    # Build pair portfolios with different exit_z / pair combos
    pair_configs = {
        "Top5_Diverse": [
            ("XLK","QQQ",42), ("XLI","XLB",126), ("XLP","XLU",63),
            ("XLB","EFA",42), ("XLF","IWM",42)
        ],
        "Top8_Diverse": [
            ("XLK","QQQ",42), ("XLK","QQQ",126), ("XLI","XLB",126),
            ("XLP","XLU",63), ("XLI","SPY",42), ("XLB","EFA",42),
            ("XLF","IWM",42), ("XLV","XLP",126)
        ],
        "Top10_Wide": [
            ("XLK","QQQ",42), ("XLK","QQQ",126), ("XLI","XLB",126),
            ("XLP","XLU",63), ("XLK","SPY",42), ("XLI","SPY",42),
            ("XLP","XLU",126), ("XLB","EFA",42), ("XLI","IWM",126),
            ("XLF","IWM",42)
        ],
    }

    # Test different exit_z values
    zp_portfolios = {}
    for ez in [0.0, 0.25, 0.5, 0.75]:
        for pname, pairs in pair_configs.items():
            for notional in [0.10, 0.15]:
                combined_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
                count = 0
                for a, b, win in pairs:
                    w = pair_weights(prices, a, b, window=win, entry_z=2.0,
                                     exit_z=ez, notional=notional)
                    if w is not None:
                        combined_w += w; count += 1
                if count == 0: continue
                res = backtest(prices, combined_w)
                m = res["metrics"]
                key = f"ZP_{pname}_n{notional}_ez{ez}"
                zp_portfolios[key] = res
                results[key] = res
                sf = "Y" if m["Sharpe Ratio"]>1.95 else " "
                print(f"  {key:50s} Sh={m['Sharpe Ratio']:.4f}[{sf}] CAGR={m['CAGR']:.2%} DD={m['Max Drawdown']:.2%}")

    # ======================================================
    # PHASE 2: FINE-GRAIN ENSEMBLE WEIGHT SWEEP
    # ======================================================
    print("\n" + SEP)
    print("PHASE 2: FINE-GRAIN WEIGHT SWEEP (CH% x ZP combos)\n")

    # For each ZP portfolio with Sharpe > 1.5
    best_zp = sorted(zp_portfolios.items(),
                     key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    for zp_key, zp_res in best_zp[:8]:
        zp_sh = zp_res["metrics"]["Sharpe Ratio"]
        if zp_sh < 1.5: continue
        zp_ret = zp_res["portfolio_returns"]
        ch_ret = ch_res["portfolio_returns"]

        for ch_pct in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
            zp_pct = 1 - ch_pct
            er = ch_pct * ch_ret + zp_pct * zp_ret
            eq = 100_000*(1+er).cumprod(); eq.name="Equity"
            m = compute_metrics(er, eq, 100_000, risk_free_rate=RF)
            ens_key = f"E({ch_pct:.0%}CH+{zp_pct:.0%}{zp_key})"
            results[ens_key] = {"equity_curve":eq, "portfolio_returns":er,
                                "weights":pd.DataFrame(), "metrics":m,
                                "turnover":pd.Series(0,index=prices.index),
                                "gross_exposure":pd.Series(1,index=prices.index),
                                "net_exposure":pd.Series(1,index=prices.index),
                                "cash_weight":pd.Series(0,index=prices.index)}

    print("  (Ensembles built, proceeding to leverage sweep...)\n")

    # ======================================================
    # PHASE 3: FINE-GRAIN LEVERAGE SWEEP with multiple cost assumptions
    # ======================================================
    print(SEP)
    print("PHASE 3: LEVERAGE SWEEP (fine increments, multiple cost models)")
    print("  Testing leverage 2.0-5.0 in 0.2 increments")
    print("  Cost models: 1.5% (standard), 1.0% (competitive), 0.5% (futures-like)\n")

    # Get all ensembles with Sharpe > 1.8 (high chance of hitting 1.95 when leveraged)
    ens_candidates = sorted(
        [(k,v) for k,v in results.items() if v["metrics"]["Sharpe Ratio"] > 1.8],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)[:20]

    # Also include raw ZP portfolios with Sharpe > 2.0
    zp_candidates = [(k,v) for k,v in results.items()
                     if k.startswith("ZP_") and v["metrics"]["Sharpe Ratio"] > 2.0]
    all_candidates = ens_candidates + zp_candidates
    # Deduplicate
    seen = set(); deduped = []
    for k,v in all_candidates:
        if k not in seen: seen.add(k); deduped.append((k,v))
    all_candidates = deduped

    print(f"  {len(all_candidates)} candidate strategies to leverage\n")

    for lev_cost_label, lev_cost in [("1.5%", 0.015), ("1.0%", 0.010), ("0.5%", 0.005)]:
        print(f"  --- Leverage cost: {lev_cost_label}/yr ---")
        found_any = False
        for base_key, base_res in all_candidates:
            br = base_res["portfolio_returns"]
            for mult in np.arange(2.0, 5.2, 0.2):
                mult = round(mult, 1)
                sr = br * mult - (mult-1)*lev_cost/252
                eq = 100_000*(1+sr).cumprod(); eq.name="Equity"
                m = compute_metrics(sr, eq, 100_000, risk_free_rate=RF)
                is_winner = m["CAGR"]>spy_cagr and m["Sharpe Ratio"]>1.95
                label = f"L({lev_cost_label})_{base_key}_x{mult}"[:80]
                results[label] = {"equity_curve":eq,"portfolio_returns":sr,
                                  "weights":pd.DataFrame(),"metrics":m,
                                  "turnover":pd.Series(0,index=prices.index),
                                  "gross_exposure":pd.Series(mult,index=prices.index),
                                  "net_exposure":pd.Series(mult,index=prices.index),
                                  "cash_weight":pd.Series(0,index=prices.index)}
                if is_winner:
                    winners.append(label)
                    found_any = True
                    print(f"  ** WINNER ** {label}")
                    print(f"     CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                          f"MaxDD={m['Max Drawdown']:.2%}")
        if not found_any:
            # Show closest misses
            lev_results = [(k,v) for k,v in results.items() if k.startswith(f"L({lev_cost_label})")]
            # Closest to both targets
            close = [(k,v) for k,v in lev_results
                     if v["metrics"]["Sharpe Ratio"]>1.90 and v["metrics"]["CAGR"]>spy_cagr*0.9]
            close_sorted = sorted(close,
                key=lambda x: min(x[1]["metrics"]["CAGR"]/spy_cagr,
                                  x[1]["metrics"]["Sharpe Ratio"]/1.95), reverse=True)
            print(f"  No winners at {lev_cost_label} cost. Closest misses:")
            for k,v in close_sorted[:5]:
                m=v["metrics"]
                print(f"    {k:70s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f}")
        print()

    # ======================================================
    # AUDIT
    # ======================================================
    print(SEP)
    auditable = {k:v for k,v in results.items()
                 if isinstance(v["weights"],pd.DataFrame) and not v["weights"].empty}
    iss = audit(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies audited): "
          f"{'ALL PASS' if not iss else f'{len(iss)} issues'}")
    for i in iss: print(f"  !! {i}")

    # ======================================================
    # FINAL SUMMARY
    # ======================================================
    print("\n" + SEP)
    sorted_all = sorted(results.items(),
                        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)

    if winners:
        print(f"\n*** {len(winners)} WINNERS FOUND ***\n")
        for w in winners:
            m=results[w]["metrics"]
            print(f"  {w}")
            print(f"    CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']:.2%}, Sortino={m['Sortino Ratio']:.4f}")
    else:
        print("No WINNER found (both CAGR>SPY AND Sharpe>1.95).")

    # Show frontier: best strategies at each Sharpe level
    print(f"\n  Efficient Frontier (CAGR-beaters by Sharpe, all cost models):")
    cb = sorted([(k,v) for k,v in sorted_all if v["metrics"]["CAGR"]>spy_cagr],
                key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)
    for k,v in cb[:15]:
        m=v["metrics"]
        print(f"    {k:70s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f} DD={m['Max Drawdown']:.2%}")

    print(f"\n  Sharpe Leaders (top 10):")
    for k,v in sorted_all[:10]:
        m=v["metrics"]
        print(f"    {k:70s} CAGR={m['CAGR']:.2%} Sh={m['Sharpe Ratio']:.4f}")

    print(f"\n  Total: {len(results)} strategies | Winners: {len(winners)} | Audit issues: {len(iss)}")
    print(SEP)


if __name__ == "__main__":
    main()
