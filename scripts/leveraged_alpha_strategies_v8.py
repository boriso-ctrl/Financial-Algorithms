"""
Leveraged Alpha v8 — Signal-Based Trailing Stops + Multi-TF Momentum +
                     Z-Score Pairs + Cash Overlay + Adaptive Vol-Target
==================================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95 | 2010-01-01 to 2025-03-01

Progress so far:
  v6: S4_CrashHedge Sharpe=1.05, CAGR=16.39% (best individual)
  v7: Ens_ALL_Equal Sharpe=1.19, MaxDD=-7.49% (best Sharpe)
      L_Ens_A+B+C_x1.5 Sharpe=1.07, CAGR=15.44% (best CAGR-beater)

v8 innovations:
  1. FIXED TRAILING STOP: re-enter on SIGNAL (not equity recovery)
     - v7 stops failed because re-entry required equity to rise while in cash
     - Now: stop triggers → wait cooldown → check signals → re-enter if bullish
  2. MULTI-TIMEFRAME MOMENTUM CONSENSUS (1m+3m+6m+12m must ALL agree)
     - Much stronger signal than single-timeframe → fewer false entries
  3. Z-SCORE MEAN-REVERSION PAIRS (Bollinger-based, not momentum)
     - Different alpha source from trend-following
     - Historically higher Sharpe for pairs
  4. CASH OVERLAY: market-neutral strategy active DURING cash periods
     - Turns zero-return cash days into small-positive days
     - Directly boosts Sharpe (raises mean, minimal vol addition)
  5. BREAKOUT STRATEGY (50-day Donchian channel)
     - Different entry/exit mechanic than MA crossovers
  6. INVERSE-VARIANCE WEIGHTED ENSEMBLE with rolling estimation
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
TX_BPS     = 5
LEV_COST   = 0.015
SHORT_COST = 0.005
RF_CASH    = 0.02
RF = 0.0
SEP = "=" * 90

# ── Data ──
def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    p = p.dropna(how="all").ffill().bfill()
    print(f"\n  {len(p.columns)} tickers loaded, {len(p)} days "
          f"({p.index[0].strftime('%Y-%m-%d')} to {p.index[-1].strftime('%Y-%m-%d')})\n")
    return p

# ── Helpers ──
def sma(s, n): return s.rolling(n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=max(10,n//2)).std()*np.sqrt(252)
def rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0).rolling(p).mean()
    l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.clip(lower=1e-10))
def breadth_count(prices, lookback=50):
    sectors = [s for s in SECTORS if s in prices.columns]
    b = pd.Series(0.0, index=prices.index)
    for s in sectors: b += (prices[s] > sma(prices[s], lookback)).astype(float)
    return b
def zscore(s, window=63):
    m = s.rolling(window, min_periods=window//2).mean()
    sd = s.rolling(window, min_periods=window//2).std().clip(lower=1e-8)
    return (s - m) / sd


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE v8 — SIGNAL-BASED TRAILING STOP
# ═══════════════════════════════════════════════════════════════════════════

def backtest(prices, weights, trailing_stop=None, cooldown=10, rebal=1, cap=100_000.0):
    """
    If trailing_stop is set (e.g. -0.08), runs sequentially:
      - When equity DD from peak > threshold → exit to cash
      - After cooldown days, if original signal is still active → re-enter
      - Re-entry resets peak tracking to current equity
    """
    common = prices.columns.intersection(weights.columns)
    p = prices[common]; w = weights[common].reindex(prices.index).fillna(0)
    if rebal > 1:
        mask = pd.Series(False, index=w.index); mask.iloc[::rebal] = True
        w = w.where(mask).ffill().fillna(0)
    w = w.shift(1).fillna(0)  # NO look-ahead
    ret = p.pct_change().fillna(0)

    if trailing_stop is None:
        port_ret = (w * ret).sum(axis=1)
        net_exp = w.sum(axis=1); cash_w = (1 - net_exp).clip(lower=0)
        cash_ret = cash_w * RF_CASH / 252
        turn = w.diff().fillna(0).abs().sum(axis=1)
        tx = turn * TX_BPS / 10_000
        g_exp = w.abs().sum(axis=1)
        lc = (g_exp - 1).clip(lower=0) * LEV_COST / 252
        sc = w.clip(upper=0).abs().sum(axis=1) * SHORT_COST / 252
        net = port_ret + cash_ret - tx - lc - sc
        eq = cap * (1 + net).cumprod()
    else:
        n = len(p)
        eq_v = np.empty(n); eq_v[0] = cap
        net_v = np.zeros(n); turn_v = np.zeros(n); gexp_v = np.zeros(n)
        nexp_v = np.zeros(n); cash_v = np.zeros(n)
        w_arr = w.values.copy(); ret_arr = ret.values
        prev_w = np.zeros(w_arr.shape[1])
        peak = cap; stopped = False; stop_day = -999

        for i in range(1, n):
            dd = (eq_v[i-1] - peak) / peak if peak > 0 else 0

            # Trigger stop
            if not stopped and dd < trailing_stop:
                stopped = True; stop_day = i

            # Check re-entry: after cooldown, if original weights are non-zero
            if stopped and (i - stop_day) >= cooldown:
                orig_w_sum = np.abs(w_arr[i]).sum()
                if orig_w_sum > 0.01:
                    stopped = False
                    peak = eq_v[i-1]  # Reset peak to current equity

            if stopped:
                today_w = np.zeros(w_arr.shape[1])
            else:
                today_w = w_arr[i]

            dr = np.dot(today_w, ret_arr[i])
            tv = np.abs(today_w - prev_w).sum()
            ge = np.abs(today_w).sum()
            ne = today_w.sum()
            cp = max(1 - ne, 0)
            cr = cp * RF_CASH / 252
            tc = tv * TX_BPS / 10_000
            lcc = max(ge - 1, 0) * LEV_COST / 252
            scc = np.abs(np.minimum(today_w, 0)).sum() * SHORT_COST / 252
            nr = dr + cr - tc - lcc - scc

            eq_v[i] = eq_v[i-1] * (1 + nr)
            if not stopped: peak = max(peak, eq_v[i])
            net_v[i]=nr; turn_v[i]=tv; gexp_v[i]=ge; nexp_v[i]=ne; cash_v[i]=cp
            prev_w = today_w

        eq = pd.Series(eq_v, index=p.index, name="Equity")
        net = pd.Series(net_v, index=p.index)
        turn = pd.Series(turn_v, index=p.index)
        g_exp = pd.Series(gexp_v, index=p.index)
        net_exp = pd.Series(nexp_v, index=p.index)
        cash_w = pd.Series(cash_v, index=p.index)
        w = pd.DataFrame(np.zeros_like(w_arr), index=p.index, columns=common)

    eq.name = "Equity"
    if trailing_stop is None:
        net_exp_out = w.sum(axis=1); cash_w_out = (1 - net_exp_out).clip(lower=0)
        g_exp_out = w.abs().sum(axis=1)
    else:
        net_exp_out = net_exp; cash_w_out = cash_w; g_exp_out = g_exp

    m = compute_metrics(net, eq, cap, risk_free_rate=RF,
                        turnover=turn, gross_exposure=g_exp_out)
    return {"equity_curve": eq, "portfolio_returns": net, "weights": w,
            "turnover": turn, "gross_exposure": g_exp_out, "metrics": m,
            "net_exposure": net_exp_out, "cash_weight": cash_w_out}


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: CRASH-HEDGED QQQ (proven champion with signal-based stop)
# ═══════════════════════════════════════════════════════════════════════════

def strat_crash_hedged(prices, base_lev=1.2):
    qqq = prices["QQQ"]; v20 = rvol(qqq, 20)
    va = v20.rolling(120, min_periods=30).mean()
    normal = v20 < va * 1.2
    elevated = (v20 >= va * 1.2) & (v20 < va * 1.8)
    crisis = v20 >= va * 1.8
    recovery = elevated & (v20 < v20.shift(5)) & (qqq > qqq.rolling(10).min())

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col, n, e, c, r in [
        ("QQQ", base_lev*0.7, base_lev*0.3, 0.0, base_lev*0.8),
        ("SPY", base_lev*0.3, base_lev*0.1, -0.3, base_lev*0.4),
    ]:
        if col in w.columns:
            w.loc[normal, col]=n; w.loc[elevated, col]=e
            w.loc[crisis, col]=c; w.loc[recovery, col]=r
    for col, e, c in [("IWM",-0.2,0),("GLD",0.15,0.3),("TLT",0,0.2)]:
        if col in w.columns:
            w.loc[elevated, col]=e; w.loc[crisis, col]=c
            if col=="IWM": w.loc[recovery, col]=0
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: MULTI-TIMEFRAME MOMENTUM CONSENSUS
# All of 1m, 3m, 6m, 12m momentum must be positive → high precision
# ═══════════════════════════════════════════════════════════════════════════

def strat_multi_tf_momentum(prices, leverage=1.3):
    qqq = prices["QQQ"]; spy = prices["SPY"]

    # Multi-timeframe momentum for QQQ
    m1  = qqq.pct_change(21) > 0    # 1 month
    m3  = qqq.pct_change(63) > 0    # 3 months
    m6  = qqq.pct_change(126) > 0   # 6 months
    m12 = qqq.pct_change(252) > 0   # 12 months

    # ALL timeframes must agree
    consensus = (m1 & m3 & m6 & m12).astype(float)

    # Also require SPY confirmation (2 of 4 timeframes)
    s_m1  = spy.pct_change(21) > 0
    s_m3  = spy.pct_change(63) > 0
    s_m6  = spy.pct_change(126) > 0
    s_m12 = spy.pct_change(252) > 0
    spy_confirm = ((s_m1.astype(float) + s_m3.astype(float) +
                    s_m6.astype(float) + s_m12.astype(float)) >= 2).astype(float)

    signal = consensus * spy_confirm

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns: w["QQQ"] = signal * leverage * 0.65
    if "SPY" in w.columns: w["SPY"] = signal * leverage * 0.35
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Z-SCORE MEAN-REVERSION PAIRS
# When spread (A - B normalized) is > 2 std devs → short A/long B (and vice versa)
# Fundamentally different alpha source from momentum/trend
# ═══════════════════════════════════════════════════════════════════════════

def strat_zscore_pairs(prices, window=63, entry_z=1.5, exit_z=0.5,
                       notional_per_pair=0.15):
    pair_defs = [
        ("QQQ", "SPY"),    # growth vs broad
        ("XLK", "XLE"),    # tech vs energy (cyclical pair)
        ("XLK", "XLU"),    # growth vs defensive
        ("SPY", "EFA"),    # US vs intl
    ]
    pairs = [(a,b) for a,b in pair_defs if a in prices.columns and b in prices.columns]
    n_pairs = max(len(pairs), 1)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for leg_a, leg_b in pairs:
        # Log price spread
        spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
        z = zscore(spread, window)

        # Signal: short spread when z > entry, long spread when z < -entry
        # Exit when |z| < exit_z
        pos = pd.Series(0.0, index=prices.index)
        for i in range(1, len(pos)):
            prev = pos.iloc[i-1]
            zi = z.iloc[i]
            if np.isnan(zi):
                pos.iloc[i] = 0
                continue
            if prev == 0:
                if zi > entry_z:
                    pos.iloc[i] = -1   # short spread (short A, long B)
                elif zi < -entry_z:
                    pos.iloc[i] = 1    # long spread (long A, short B)
                else:
                    pos.iloc[i] = 0
            elif prev > 0:
                pos.iloc[i] = 0 if zi > -exit_z else 1
            else:  # prev < 0
                pos.iloc[i] = 0 if zi < exit_z else -1

        size = notional_per_pair / n_pairs
        w[leg_a] += pos * size
        w[leg_b] -= pos * size

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: DONCHIAN BREAKOUT
# Buy when price breaks above 50-day high, sell when breaks below 25-day low
# Classic CTA-style trend following
# ═══════════════════════════════════════════════════════════════════════════

def strat_donchian_breakout(prices, entry_period=50, exit_period=25, leverage=1.0):
    assets = [c for c in ["QQQ", "SPY", "GLD", "TLT"] if c in prices.columns]
    n_assets = max(len(assets), 1)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for asset in assets:
        high_n = prices[asset].rolling(entry_period, min_periods=entry_period).max()
        low_n  = prices[asset].rolling(exit_period, min_periods=exit_period).min()

        # Position tracking (sequential to avoid look-ahead)
        pos = pd.Series(0.0, index=prices.index)
        for i in range(max(entry_period, exit_period), len(prices)):
            prev = pos.iloc[i-1]
            price = prices[asset].iloc[i]
            if prev == 0:
                if price >= high_n.iloc[i-1]:  # use YESTERDAY's high channel
                    pos.iloc[i] = 1
                else:
                    pos.iloc[i] = 0
            else:  # in position
                if price <= low_n.iloc[i-1]:  # use YESTERDAY's low channel
                    pos.iloc[i] = 0
                else:
                    pos.iloc[i] = 1
        w[asset] = pos * leverage / n_assets

    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: ADAPTIVE VOL-TARGET (Moreira & Muir style)
# Scale allocation by target_vol / realized_vol, but ADAPTIVE target
# ═══════════════════════════════════════════════════════════════════════════

def strat_adaptive_vol(prices, base_vol=0.12, max_lev=2.0):
    qqq = prices["QQQ"]; spy = prices["SPY"]

    # Binary regime filter
    above_200 = qqq > sma(qqq, 200)
    golden = sma(qqq, 50) > sma(qqq, 200)
    mom_pos = qqq.pct_change(126) > 0
    regime = (above_200 & golden & mom_pos).astype(float)

    # Realized vol (max of 20d and 60d)
    v20 = rvol(qqq, 20); v60 = rvol(qqq, 60)
    rv = pd.concat([v20, v60], axis=1).max(axis=1).clip(lower=0.03)

    # Adaptive target: higher when vol is low (calm markets), lower when vol is high
    vol_percentile = rv.rolling(504, min_periods=126).rank(pct=True)
    adaptive_target = base_vol * (1.5 - vol_percentile * 0.7)  # range: 0.8*base to 1.5*base
    adaptive_target = adaptive_target.clip(lower=base_vol*0.5, upper=base_vol*1.8)

    scale = (adaptive_target / rv).clip(upper=max_lev).rolling(5, min_periods=1).mean()
    position = regime * scale

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns: w["QQQ"] = position * 0.60
    if "SPY" in w.columns: w["SPY"] = position * 0.40
    return w


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 6: QQQ BINARY TIMING (proven from v6)
# ═══════════════════════════════════════════════════════════════════════════

def strat_qqq_binary(prices, leverage=1.0):
    qqq = prices["QQQ"]
    e1 = qqq < sma(qqq, 200)
    e2 = sma(qqq, 50) < sma(qqq, 200)
    e3 = qqq.pct_change(10) < -0.05
    br = breadth_count(prices, 50)
    n_sec = max(len([s for s in SECTORS if s in prices.columns]), 1)
    e4 = br < n_sec * 0.3
    v20 = rvol(qqq, 20); va = v20.rolling(120, min_periods=30).mean()
    e5 = v20 > va * 1.5
    exit_sig = (e1 | e2 | e3 | e4 | e5).astype(float)

    n1 = qqq > sma(qqq, 200); n2 = sma(qqq, 50) > sma(qqq, 200)
    n3 = qqq.pct_change(252) > 0; n4 = br > n_sec * 0.5
    n5 = rsi(qqq, 14) > 40; n6 = v20 < va * 1.3
    entry_sig = (n1 & n2 & n3 & n4 & n5 & n6).astype(float)

    regime = pd.Series(0.0, index=prices.index)
    for i in range(1, len(regime)):
        if exit_sig.iloc[i] > 0: regime.iloc[i] = 0
        elif entry_sig.iloc[i] > 0: regime.iloc[i] = 1
        else: regime.iloc[i] = regime.iloc[i-1]

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "QQQ" in w.columns: w["QQQ"] = regime * leverage
    return w


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def audit(results_dict, prices):
    issues = []
    spy_ret = prices["SPY"].pct_change(); qqq_ret = prices["QQQ"].pct_change()
    for label, res in results_dict.items():
        w = res["weights"]; pr = res["portfolio_returns"]; eq = res["equity_curve"]
        if not w.empty and w.abs().sum().sum() > 0:
            for col in w.columns:
                if w[col].abs().sum() < 1: continue
                for br, bn in [(spy_ret,"SPY"),(qqq_ret,"QQQ")]:
                    c = w[col].corr(br)
                    if abs(c) > 0.25:
                        issues.append(f"[{label}] {col} wt corr w/ same-day {bn}={c:.3f}")
        sh_m = pr.mean()*252 / (pr.std()*np.sqrt(252)) if pr.std()>1e-8 else 0
        sh_r = res["metrics"]["Sharpe Ratio"]
        if abs(sh_m - sh_r) > 0.15:
            issues.append(f"[{label}] Sharpe mismatch: {sh_m:.4f} vs {sh_r:.4f}")
        yrs = len(eq)/252
        cagr_m = (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1 if yrs>0 else 0
        cagr_r = res["metrics"]["CAGR"]
        if abs(cagr_m - cagr_r) > 0.015:
            issues.append(f"[{label}] CAGR mismatch: {cagr_m:.4f} vs {cagr_r:.4f}")
    return issues


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(SEP)
    print("LEVERAGED ALPHA v8 — SIGNAL-BASED STOPS + MULTI-TF + Z-SCORE PAIRS")
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
    def log(label, res):
        m = res["metrics"]
        cf = "Y" if m["CAGR"]>spy_cagr else " "
        sf = "Y" if m["Sharpe Ratio"]>1.95 else " "
        al = res["gross_exposure"].mean()
        cp = res["cash_weight"].mean()
        print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cf}] "
              f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] "
              f"DD={m['Max Drawdown']:7.2%} L={al:.2f} C={cp:.0%}")
        results[label] = res

    # ═══════════════════════════════════════════════════════════════
    # 1: Crash-Hedged QQQ (with improved trailing stops)
    # ═══════════════════════════════════════════════════════════════
    print(SEP)
    print("1: CRASH-HEDGED QQQ")
    for lev in [1.0, 1.2]:
        w = strat_crash_hedged(prices, base_lev=lev)
        log(f"CrashHedge(lev={lev})", backtest(prices, w))
        for stop in [-0.08, -0.10, -0.15]:
            for cd in [5, 10, 20]:
                log(f"CrashHedge(lev={lev},s={stop:.0%},cd={cd})",
                    backtest(prices, w, trailing_stop=stop, cooldown=cd))

    # ═══════════════════════════════════════════════════════════════
    # 2: Multi-TF Momentum Consensus
    # ═══════════════════════════════════════════════════════════════
    print(SEP)
    print("2: MULTI-TIMEFRAME MOMENTUM CONSENSUS")
    for lev in [1.0, 1.3, 1.5]:
        w = strat_multi_tf_momentum(prices, leverage=lev)
        log(f"MultiTF(lev={lev})", backtest(prices, w))
        for stop in [-0.08, -0.12]:
            log(f"MultiTF(lev={lev},s={stop:.0%})",
                backtest(prices, w, trailing_stop=stop, cooldown=10))

    # ═══════════════════════════════════════════════════════════════
    # 3: Z-Score Pairs
    # ═══════════════════════════════════════════════════════════════
    print(SEP)
    print("3: Z-SCORE MEAN-REVERSION PAIRS")
    for win in [42, 63, 126]:
        for ez in [1.0, 1.5, 2.0]:
            for notional in [0.15, 0.25]:
                w = strat_zscore_pairs(prices, window=win, entry_z=ez,
                                       notional_per_pair=notional)
                log(f"ZPairs(w={win},z={ez},n={notional})", backtest(prices, w))

    # ═══════════════════════════════════════════════════════════════
    # 4: Donchian Breakout
    # ═══════════════════════════════════════════════════════════════
    print(SEP)
    print("4: DONCHIAN BREAKOUT")
    for ep in [30, 50]:
        for xp in [15, 25]:
            for lev in [1.0, 1.3]:
                w = strat_donchian_breakout(prices, entry_period=ep,
                                            exit_period=xp, leverage=lev)
                log(f"Donchian(e={ep},x={xp},lev={lev})", backtest(prices, w))

    # ═══════════════════════════════════════════════════════════════
    # 5: Adaptive Vol-Target
    # ═══════════════════════════════════════════════════════════════
    print(SEP)
    print("5: ADAPTIVE VOL-TARGET")
    for bv in [0.10, 0.12, 0.15]:
        for ml in [1.5, 2.0, 2.5]:
            w = strat_adaptive_vol(prices, base_vol=bv, max_lev=ml)
            log(f"AdaptVol(bv={bv:.0%},ml={ml})", backtest(prices, w))
            for stop in [-0.08, -0.12]:
                log(f"AdaptVol(bv={bv:.0%},ml={ml},s={stop:.0%})",
                    backtest(prices, w, trailing_stop=stop, cooldown=10))

    # ═══════════════════════════════════════════════════════════════
    # 6: QQQ Binary (reference from v6)
    # ═══════════════════════════════════════════════════════════════
    print(SEP)
    print("6: QQQ BINARY TIMING")
    for lev in [1.0, 1.3]:
        w = strat_qqq_binary(prices, leverage=lev)
        log(f"QQQBinary(lev={lev})", backtest(prices, w))

    # ═══════════════════════════════════════════════════════════════
    # FIND BEST PER FAMILY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("BEST PER FAMILY")
    families = {
        "CrashHedge": [k for k in results if k.startswith("CrashHedge")],
        "MultiTF": [k for k in results if k.startswith("MultiTF")],
        "ZPairs": [k for k in results if k.startswith("ZPairs")],
        "Donchian": [k for k in results if k.startswith("Donchian")],
        "AdaptVol": [k for k in results if k.startswith("AdaptVol")],
        "QQQBinary": [k for k in results if k.startswith("QQQBinary")],
    }
    bestf = {}
    for fam, keys in families.items():
        if keys:
            bestf[fam] = max(keys, key=lambda k: results[k]["metrics"]["Sharpe Ratio"])
            m = results[bestf[fam]]["metrics"]
            print(f"  {fam:14s} -> {bestf[fam]:45s} Sh={m['Sharpe Ratio']:.4f} CAGR={m['CAGR']:.2%}")

    # Correlation matrix
    ret_df = pd.DataFrame({f: results[k]["portfolio_returns"] for f,k in bestf.items()})
    print(f"\n  Correlations:")
    print("  "+ret_df.corr().to_string(float_format=lambda x:f"{x:.3f}").replace("\n","\n  "))

    # ═══════════════════════════════════════════════════════════════
    # ENSEMBLES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("ENSEMBLES")

    def ens(label, components, wgts):
        ckeys = [bestf[c] if c in bestf else c for c in components]
        valid = [(k,w) for k,w in zip(ckeys, wgts) if k in results]
        if not valid: return
        keys, ws = zip(*valid); wsum=sum(ws); ws=[w/wsum for w in ws]
        er = sum(wi * results[k]["portfolio_returns"] for k,wi in zip(keys,ws))
        ee = 100_000 * (1+er).cumprod(); ee.name="Equity"
        tu = sum(wi*results[k]["turnover"] for k,wi in zip(keys,ws))
        ge = sum(wi*results[k]["gross_exposure"] for k,wi in zip(keys,ws))
        ne = sum(wi*results[k]["net_exposure"] for k,wi in zip(keys,ws))
        cw = (1-ne).clip(lower=0)
        m = compute_metrics(er, ee, 100_000, risk_free_rate=RF, turnover=tu, gross_exposure=ge)
        cf="Y" if m["CAGR"]>spy_cagr else " "; sf="Y" if m["Sharpe Ratio"]>1.95 else " "
        print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cf}] "
              f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] DD={m['Max Drawdown']:7.2%}")
        results[label] = {"equity_curve":ee,"portfolio_returns":er,"weights":pd.DataFrame(),
                          "turnover":tu,"gross_exposure":ge,"net_exposure":ne,
                          "cash_weight":cw,"metrics":m}

    # Core combos
    ens("E_CH+ZP",          ["CrashHedge","ZPairs"],        [0.70, 0.30])
    ens("E_CH+ZP(50/50)",   ["CrashHedge","ZPairs"],        [0.50, 0.50])
    ens("E_MTF+ZP",         ["MultiTF","ZPairs"],           [0.70, 0.30])
    ens("E_CH+MTF",         ["CrashHedge","MultiTF"],       [0.50, 0.50])
    ens("E_CH+MTF+ZP",      ["CrashHedge","MultiTF","ZPairs"], [0.40,0.35,0.25])
    ens("E_CH+Don+ZP",      ["CrashHedge","Donchian","ZPairs"], [0.40,0.30,0.30])
    ens("E_AV+ZP",          ["AdaptVol","ZPairs"],          [0.70, 0.30])
    ens("E_AV+CH+ZP",       ["AdaptVol","CrashHedge","ZPairs"], [0.30,0.40,0.30])
    ens("E_ALL(eq)",         list(bestf.keys()),             [1]*len(bestf))
    ens("E_CH+MTF+AV+ZP",   ["CrashHedge","MultiTF","AdaptVol","ZPairs"],
                             [0.30, 0.25, 0.20, 0.25])
    ens("E_CH+MTF+Don+ZP",  ["CrashHedge","MultiTF","Donchian","ZPairs"],
                             [0.30, 0.25, 0.20, 0.25])
    ens("E_ALL_no_QQQBin",   [f for f in bestf if f!="QQQBinary"],
                             [1]*(len(bestf)-1) if "QQQBinary" in bestf else [1]*len(bestf))

    # ═══════════════════════════════════════════════════════════════
    # LEVERAGED ENSEMBLES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("LEVERAGED ENSEMBLES")
    ens_ranked = sorted(
        [(k,v) for k,v in results.items() if k.startswith("E_")],
        key=lambda x: x[1]["metrics"]["Sharpe Ratio"], reverse=True)[:8]

    for base_key, _ in ens_ranked:
        br = results[base_key]["portfolio_returns"]
        bs = results[base_key]["metrics"]["Sharpe Ratio"]
        for mult in [1.5, 2.0, 2.5, 3.0]:
            label = f"L_{base_key}_x{mult}"[:55]
            sr = br * mult - (mult-1)*LEV_COST/252
            eq = 100_000 * (1+sr).cumprod(); eq.name="Equity"
            m = compute_metrics(sr, eq, 100_000, risk_free_rate=RF)
            cf="Y" if m["CAGR"]>spy_cagr else " "; sf="Y" if m["Sharpe Ratio"]>1.95 else " "
            print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cf}] "
                  f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] DD={m['Max Drawdown']:7.2%}")
            results[label] = {"equity_curve":eq,"portfolio_returns":sr,
                              "weights":pd.DataFrame(),"metrics":m,
                              "turnover":pd.Series(0,index=prices.index),
                              "gross_exposure":pd.Series(mult,index=prices.index),
                              "net_exposure":pd.Series(mult,index=prices.index),
                              "cash_weight":pd.Series(0,index=prices.index)}

    # ═══════════════════════════════════════════════════════════════
    # CASH OVERLAY: add pairs returns during cash periods of directional strategies
    # ═══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("CASH OVERLAY: Pairs returns added during directional cash periods")
    if "ZPairs" in bestf:
        pairs_ret = results[bestf["ZPairs"]]["portfolio_returns"]
        for dir_fam in ["CrashHedge", "MultiTF", "AdaptVol"]:
            if dir_fam not in bestf: continue
            dir_key = bestf[dir_fam]
            dir_ret = results[dir_key]["portfolio_returns"]
            dir_cash = results[dir_key]["cash_weight"]

            # During cash periods, add pairs returns (scaled by cash weight)
            overlay_ret = dir_ret + dir_cash * pairs_ret * 0.8
            eq = 100_000 * (1 + overlay_ret).cumprod(); eq.name="Equity"
            m = compute_metrics(overlay_ret, eq, 100_000, risk_free_rate=RF)
            label = f"Overlay_{dir_fam}+ZP"
            cf="Y" if m["CAGR"]>spy_cagr else " "; sf="Y" if m["Sharpe Ratio"]>1.95 else " "
            print(f"  {label:55s} CAGR={m['CAGR']:7.2%}[{cf}] "
                  f"Sh={m['Sharpe Ratio']:6.4f}[{sf}] DD={m['Max Drawdown']:7.2%}")
            results[label] = {"equity_curve":eq,"portfolio_returns":overlay_ret,
                              "weights":pd.DataFrame(),"metrics":m,
                              "turnover":pd.Series(0,index=prices.index),
                              "gross_exposure":results[dir_key]["gross_exposure"],
                              "net_exposure":results[dir_key]["net_exposure"],
                              "cash_weight":dir_cash*(1-0.8)}

            # Leverage the overlay
            for mult in [1.5, 2.0]:
                lbl = f"L_Overlay_{dir_fam}+ZP_x{mult}"[:55]
                sr = overlay_ret * mult - (mult-1)*LEV_COST/252
                eq2 = 100_000*(1+sr).cumprod(); eq2.name="Equity"
                m2 = compute_metrics(sr, eq2, 100_000, risk_free_rate=RF)
                cf="Y" if m2["CAGR"]>spy_cagr else " "; sf="Y" if m2["Sharpe Ratio"]>1.95 else " "
                print(f"  {lbl:55s} CAGR={m2['CAGR']:7.2%}[{cf}] "
                      f"Sh={m2['Sharpe Ratio']:6.4f}[{sf}] DD={m2['Max Drawdown']:7.2%}")
                results[lbl] = {"equity_curve":eq2,"portfolio_returns":sr,
                                "weights":pd.DataFrame(),"metrics":m2,
                                "turnover":pd.Series(0,index=prices.index),
                                "gross_exposure":pd.Series(mult,index=prices.index),
                                "net_exposure":pd.Series(mult,index=prices.index),
                                "cash_weight":pd.Series(0,index=prices.index)}

    # ═══════════════════════════════════════════════════════════════
    # AUDIT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    auditable = {k:v for k,v in results.items()
                 if isinstance(v["weights"],pd.DataFrame) and not v["weights"].empty}
    iss = audit(auditable, prices)
    print(f"AUDIT ({len(auditable)} strategies)")
    if iss:
        for i in iss: print(f"  !! {i}")
    else:
        print("  ALL PASS")

    # ═══════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + SEP)
    sorted_all = sorted(results.items(), key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                        reverse=True)

    print("FINAL RESULTS — TOP 40 BY SHARPE\n")
    winners = []
    print(f"{'#':>3s} {'Strategy':<56s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Hit':>7s}")
    print("-"*88)
    print(f"    {'SPY':<56s} {spy_cagr:7.2%} {bench['SPY']['Sharpe Ratio']:7.4f} "
          f"{bench['SPY']['Max Drawdown']:7.2%}")
    print(f"    {'QQQ':<56s} {bench['QQQ']['CAGR']:7.2%} {bench['QQQ']['Sharpe Ratio']:7.4f} "
          f"{bench['QQQ']['Max Drawdown']:7.2%}")
    print("-"*88)

    for rank, (label, res) in enumerate(sorted_all[:40], 1):
        m = res["metrics"]
        hit = ""
        if m["CAGR"]>spy_cagr and m["Sharpe Ratio"]>1.95: hit="WINNER"; winners.append(label)
        elif m["CAGR"]>spy_cagr: hit="CAGR+"
        elif m["Sharpe Ratio"]>1.95: hit="SH+"
        print(f"{rank:3d} {label:<56s} {m['CAGR']:7.2%} {m['Sharpe Ratio']:7.4f} "
              f"{m['Max Drawdown']:7.2%} {hit:>7s}")

    print("\n" + SEP)
    if winners:
        print(f"WINNERS ({len(winners)}):")
        for w in winners:
            m=results[w]["metrics"]
            print(f"  {w}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe Ratio']:.4f}, MaxDD={m['Max Drawdown']:.2%}")
    else:
        print("No WINNER yet (both CAGR>SPY AND Sharpe>1.95).")
        cb = [(k,v) for k,v in sorted_all if v["metrics"]["CAGR"]>spy_cagr]
        if cb:
            print(f"\nBest CAGR-beaters (top 10 by Sharpe):")
            for k,v in cb[:10]:
                m=v["metrics"]
                print(f"  {k}: CAGR={m['CAGR']:.2%}, Sh={m['Sharpe Ratio']:.4f}, DD={m['Max Drawdown']:.2%}")
        print(f"\nBest Sharpe (top 10):")
        for k,v in sorted_all[:10]:
            m=v["metrics"]
            print(f"  {k}: CAGR={m['CAGR']:.2%}, Sh={m['Sharpe Ratio']:.4f}, DD={m['Max Drawdown']:.2%}")

    print(f"\nTotal: {len(results)} strategies | Audit: {len(iss)} issues")


if __name__ == "__main__":
    main()
