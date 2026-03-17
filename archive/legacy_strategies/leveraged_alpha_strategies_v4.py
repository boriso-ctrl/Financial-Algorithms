"""
Leveraged Alpha v4 — Long/Short + Trade Management + Sector Diversification
============================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95

Key innovations over v3 (best was Sharpe 1.02):
  1. ACTIVE SHORTS during bear markets (turns negative periods → positive)
  2. Trade management with STOP-LOSSES (truncates left tail → big Sharpe boost)
  3. SECTOR ETF expansion (11 sectors + bonds + gold = more diversification)
  4. Multi-timeframe signal atoms with low correlation
  5. Multi-atom ensemble for maximum diversification benefit

Approach: Many independent strategy "atoms" combined at low correlation
  - Each atom targets Sharpe ~0.5-1.0
  - 10+ atoms at pairwise ρ ~0.2-0.4 → ensemble Sharpe ~1.5-2.0+
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

# ── Config ──
SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE"]
BROAD = ["SPY", "QQQ", "IWM"]
HEDGE = ["TLT", "IEF", "GLD"]
ALL_TICKERS = SECTORS + BROAD + HEDGE
START = "2010-01-01"
END = "2025-03-01"
TX_BPS = 5
LEV_COST = 0.015
SHORT_COST = 0.005
RF = 0.0


def load_data():
    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    p = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return p.dropna(how="all").ffill().bfill()


def run_bt(prices, weights, rebal=5, cap=100_000.0):
    common = prices.columns.intersection(weights.columns)
    p, w = prices[common], weights[common].reindex(prices.index).fillna(0)
    mask = pd.Series(False, index=w.index)
    mask.iloc[::rebal] = True
    w = w.where(mask).ffill().fillna(0)
    w = w.shift(1).fillna(0)
    ret = p.pct_change().fillna(0)
    gross = (w * ret).sum(axis=1)
    turn = w.diff().fillna(0).abs().sum(axis=1)
    tx = turn * TX_BPS / 10_000
    g_exp = w.abs().sum(axis=1)
    lc = (g_exp - 1).clip(lower=0) * LEV_COST / 252
    sc = w.clip(upper=0).abs().sum(axis=1) * SHORT_COST / 252
    net = gross - tx - lc - sc
    eq = cap * (1 + net).cumprod()
    eq.name = "Equity"
    m = compute_metrics(net, eq, cap, risk_free_rate=RF, turnover=turn, gross_exposure=g_exp)
    return {"equity_curve": eq, "portfolio_returns": net, "weights": w,
            "metrics": m, "gross_exposure": g_exp}


# ── Signal primitives ──
def sma(s, n): return s.rolling(n, min_periods=n).mean()
def rvol(s, n=21): return s.pct_change().rolling(n, min_periods=10).std() * np.sqrt(252)
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    lo = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / lo.clip(lower=1e-10))
def mom(s, n=252, skip=21): return s.shift(skip).pct_change(n - skip)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY ATOMS — Small, independent signal-asset combinations
# ═══════════════════════════════════════════════════════════════════════════

def atom_trend_ls(prices, ticker, fast=50, slow=200, weight=0.1):
    """Long/short single asset based on MA trend. Short in downtrend."""
    s = prices[ticker]
    f, sl = sma(s, fast), sma(s, slow)
    sig = np.where(f > sl, weight, -weight * 0.5)  # long: full, short: half
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = sig
    return w

def atom_momentum_ls(prices, ticker, lookback=252, weight=0.1):
    """Long if positive 12m momentum, short if negative."""
    m = mom(prices[ticker], lookback, 21)
    sig = np.where(m > 0, weight, np.where(m < -0.05, -weight * 0.5, 0))
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = sig
    return w

def atom_mean_reversion(prices, ticker, rsi_period=5, weight=0.1):
    """Short-term RSI mean reversion. Buy oversold, sell overbought."""
    r = rsi(prices[ticker], rsi_period)
    trend = (prices[ticker] > sma(prices[ticker], 200)).astype(float)
    # Only buy oversold in uptrend, only short overbought in downtrend
    sig = np.where((r < 25) & (trend > 0), weight * 1.5,
          np.where((r < 35) & (trend > 0), weight * 0.8,
          np.where((r > 80) & (trend < 1), -weight * 0.8,
          np.where((r > 70) & (trend < 1), -weight * 0.5, 0.0))))
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = sig
    return w

def atom_vol_managed(prices, ticker, target_vol=0.10, max_w=0.3):
    """Vol-managed single asset (always long, vol-scaled)."""
    v = rvol(prices[ticker], 21)
    scale = (target_vol / v.clip(lower=0.03)).clip(upper=max_w / 0.1)
    scale = scale.rolling(5, min_periods=1).mean()
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = (0.1 * scale).clip(upper=max_w)
    return w

def atom_cross_sectional_momentum(prices, n_long=3, n_short=2, weight=0.15):
    """Cross-sectional momentum: long top N, short bottom N."""
    # Use equity + sector tickers
    eq_tickers = [t for t in SECTORS + BROAD if t in prices.columns]
    if len(eq_tickers) < n_long + n_short:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    m12 = mom(prices[eq_tickers], 252, 21)
    ranks = m12.rank(axis=1, pct=True)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    n = len(eq_tickers)
    top_thresh = 1 - n_long / n
    bot_thresh = n_short / n

    for t in eq_tickers:
        w[t] = np.where(ranks[t] >= top_thresh, weight,
               np.where(ranks[t] <= bot_thresh, -weight * 0.6, 0.0))
    return w

def atom_sector_rotation(prices, n_top=3, weight=0.2):
    """Rotate into top-momentum sectors, avoid bottom."""
    sec = [t for t in SECTORS if t in prices.columns]
    if len(sec) < 5:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    m3 = prices[sec].pct_change(63)
    m6 = prices[sec].shift(21).pct_change(105)
    combined = m3 * 0.5 + m6 * 0.5
    ranks = combined.rank(axis=1, pct=True)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    thresh = 1 - n_top / len(sec)
    for t in sec:
        w[t] = np.where(ranks[t] >= thresh, weight / n_top,
               np.where(ranks[t] <= 0.2, -weight * 0.3 / max(int(len(sec) * 0.2), 1), 0.0))
    return w

def atom_crisis_hedge(prices, hedge_weight=0.15):
    """Long GLD+IEF when SPY in downtrend. Hedge generates POSITIVE returns in crises."""
    spy = prices["SPY"]
    ma200 = sma(spy, 200)
    ma50 = sma(spy, 50)
    bear = ((spy < ma200) & (ma50 < ma200)).astype(float)
    mild_bear = ((spy < ma200) & (ma50 >= ma200)).astype(float) * 0.5

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "GLD" in w.columns:
        w["GLD"] = (bear + mild_bear) * hedge_weight * 0.6 + 0.03  # always hold tiny
    if "IEF" in w.columns:
        w["IEF"] = (bear + mild_bear) * hedge_weight * 0.4 + 0.02
    return w


# ═══════════════════════════════════════════════════════════════════════════
# EVENT-DRIVEN TRADE MANAGER — Adds stop-losses and profit targets
# Operates on top of weight-based strategy, truncating losing trades
# ═══════════════════════════════════════════════════════════════════════════

def apply_trade_management(prices, raw_weights, stop_pct=0.03, trail_pct=0.04,
                           profit_target=0.06):
    """
    Post-process weights to simulate stop-losses and trailing stops.
    For each asset: track entry price, apply stop-loss and trailing stop.
    This truncates the left tail of returns.
    """
    managed = raw_weights.copy()
    ret = prices.pct_change().fillna(0)

    for col in managed.columns:
        if managed[col].abs().sum() < 0.01:
            continue

        position = 0.0  # current position direction: 0, +1, -1
        entry_price = 0.0
        highest_since_entry = 0.0
        lowest_since_entry = np.inf

        sig = managed[col].values
        px = prices[col].values if col in prices.columns else None
        if px is None:
            continue

        for i in range(1, len(sig)):
            new_sig = sig[i]

            if position == 0:
                # New entry
                if abs(new_sig) > 0.01:
                    position = np.sign(new_sig)
                    entry_price = px[i - 1]  # use previous close as entry (shift-1)
                    highest_since_entry = entry_price
                    lowest_since_entry = entry_price
                continue

            # Update tracking
            current = px[i - 1]
            if current > highest_since_entry:
                highest_since_entry = current
            if current < lowest_since_entry:
                lowest_since_entry = current

            # Check stop-loss and trailing stop
            if position > 0:  # long position
                pnl_from_entry = (current - entry_price) / entry_price
                pnl_from_high = (current - highest_since_entry) / highest_since_entry

                # Stop-loss: exit if down stop_pct from entry
                if pnl_from_entry < -stop_pct:
                    sig[i] = 0.0
                    position = 0
                    continue

                # Trailing stop: exit if down trail_pct from high
                if pnl_from_entry > 0.01 and pnl_from_high < -trail_pct:
                    sig[i] = 0.0
                    position = 0
                    continue

                # Profit target
                if pnl_from_entry > profit_target:
                    sig[i] = new_sig * 0.5  # take half profits
                    highest_since_entry = current  # reset for trailing

            elif position < 0:  # short position
                pnl_from_entry = (entry_price - current) / entry_price
                pnl_from_low = (lowest_since_entry - current) / lowest_since_entry

                if pnl_from_entry < -stop_pct:
                    sig[i] = 0.0
                    position = 0
                    continue

                if pnl_from_entry > 0.01 and pnl_from_low < -trail_pct:
                    sig[i] = 0.0
                    position = 0
                    continue

            # Signal changed direction
            if new_sig * position < 0:
                position = np.sign(new_sig)
                entry_price = current
                highest_since_entry = current
                lowest_since_entry = current
            elif abs(new_sig) < 0.01:
                position = 0

        managed[col] = sig
    return managed


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def strategy_long_short_scoring(prices, max_lev=1.5, short_scale=0.5):
    """Enhanced S2_Scoring from v3 with active shorts in bear markets."""
    spy = prices["SPY"]

    c1 = (spy > sma(spy, 200)).astype(float)
    c2 = (sma(spy, 50) > sma(spy, 200)).astype(float)
    c3 = (spy.pct_change(252) > 0).astype(float)
    vol = rvol(spy, 21)
    c4 = (vol < vol.rolling(126, min_periods=30).mean() * 1.1).astype(float)
    r = rsi(spy, 14)
    c5 = ((r > 30) & (r < 75)).astype(float)
    c6 = (spy > sma(spy, 20)).astype(float)

    score = c1 + c2 + c3 + c4 + c5 + c6

    # Long leverage from score
    long_lev = pd.Series(0.0, index=prices.index)
    long_lev[score >= 5] = max_lev
    long_lev[(score >= 3) & (score < 5)] = max_lev * 0.5
    long_lev[(score >= 1) & (score < 3)] = 0.0

    # SHORT when few conditions met
    short_lev = pd.Series(0.0, index=prices.index)
    short_lev[score <= 1] = short_scale * max_lev
    short_lev[score == 2] = short_scale * max_lev * 0.3

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    eq = [c for c in ["SPY", "QQQ"] if c in prices.columns]
    for c in eq:
        per = 1.0 / max(len(eq), 1)
        w[c] = long_lev * per - short_lev * per  # long - short

    # Hedge
    hedge = ((6 - score) / 6).clip(lower=0.05, upper=0.5)
    if "GLD" in w.columns:
        w["GLD"] = hedge * 0.4
    if "IEF" in w.columns:
        w["IEF"] = hedge * 0.3

    return w


def strategy_multi_atom_ensemble(prices, with_trade_mgmt=False):
    """
    Combine many independent strategy atoms for maximum diversification.
    Each atom operates on a different signal/asset/timeframe combo.
    """
    atoms = []
    atom_names = []

    # Trend atoms across different assets and timeframes
    for ticker in ["SPY", "QQQ", "IWM"]:
        if ticker in prices.columns:
            atoms.append(atom_trend_ls(prices, ticker, 50, 200, 0.08))
            atom_names.append(f"trend_50_200_{ticker}")
            atoms.append(atom_trend_ls(prices, ticker, 20, 100, 0.06))
            atom_names.append(f"trend_20_100_{ticker}")

    # Momentum atoms
    for ticker in ["SPY", "QQQ", "XLK", "GLD"]:
        if ticker in prices.columns:
            atoms.append(atom_momentum_ls(prices, ticker, 252, 0.06))
            atom_names.append(f"mom_12m_{ticker}")
            atoms.append(atom_momentum_ls(prices, ticker, 126, 0.05))
            atom_names.append(f"mom_6m_{ticker}")

    # Mean-reversion atoms
    for ticker in ["SPY", "QQQ", "IWM"]:
        if ticker in prices.columns:
            atoms.append(atom_mean_reversion(prices, ticker, 5, 0.07))
            atom_names.append(f"mr_rsi5_{ticker}")
            atoms.append(atom_mean_reversion(prices, ticker, 14, 0.05))
            atom_names.append(f"mr_rsi14_{ticker}")

    # Vol-managed atoms
    for ticker in ["SPY", "QQQ"]:
        if ticker in prices.columns:
            atoms.append(atom_vol_managed(prices, ticker, 0.10, 0.25))
            atom_names.append(f"volmgd_{ticker}")

    # Cross-sectional momentum
    atoms.append(atom_cross_sectional_momentum(prices, 3, 2, 0.10))
    atom_names.append("xsec_mom")

    # Sector rotation
    atoms.append(atom_sector_rotation(prices, 3, 0.15))
    atom_names.append("sec_rot")

    # Crisis hedge
    atoms.append(atom_crisis_hedge(prices, 0.12))
    atom_names.append("crisis_hedge")

    # Combine all atoms
    all_cols = set()
    for a in atoms:
        all_cols.update(a.columns)
    common = sorted(all_cols.intersection(prices.columns))

    combined = pd.DataFrame(0.0, index=prices.index, columns=common)
    for a in atoms:
        for c in common:
            if c in a.columns:
                combined[c] += a[c].reindex(prices.index, fill_value=0)

    # Apply trade management if requested
    if with_trade_mgmt:
        combined = apply_trade_management(prices, combined,
                                          stop_pct=0.03, trail_pct=0.04,
                                          profit_target=0.08)

    return combined, atom_names


def strategy_concentrated_ls_with_stops(prices, leverage=1.5):
    """
    Concentrated long/short on SPY+QQQ with trade management.
    Uses scoring + vol management + stops.
    """
    spy = prices["SPY"]
    qqq = prices.get("QQQ", spy)

    # Multi-condition score
    c1 = (spy > sma(spy, 200)).astype(float)
    c2 = (sma(spy, 50) > sma(spy, 200)).astype(float)
    c3 = (spy > sma(spy, 50)).astype(float)
    c4 = (spy > sma(spy, 20)).astype(float)
    vol = rvol(spy, 21)
    c5 = (vol < vol.rolling(126, min_periods=30).mean() * 1.0).astype(float)
    r = rsi(spy, 14)
    c6 = ((r > 30) & (r < 72)).astype(float)
    c7 = (spy.pct_change(252) > 0).astype(float)
    c8 = (spy.pct_change(63) > 0).astype(float)

    score = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

    # Equity sizing
    eq_w = pd.Series(0.0, index=prices.index)
    eq_w[score >= 7] = leverage
    eq_w[(score >= 5) & (score < 7)] = leverage * 0.6
    eq_w[(score >= 3) & (score < 5)] = leverage * 0.2
    eq_w[(score >= 1) & (score < 3)] = -leverage * 0.3  # SHORT
    eq_w[score == 0] = -leverage * 0.5  # HEAVY SHORT

    # Vol management
    port_ret = 0.55 * spy.pct_change() + 0.45 * qqq.pct_change()
    pvol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_scale = (0.12 / pvol.clip(lower=0.03)).clip(upper=2.0)
    vol_scale = vol_scale.rolling(5, min_periods=1).mean()

    eq_w = eq_w * vol_scale

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in w.columns:
        w["SPY"] = eq_w * 0.55
    if "QQQ" in w.columns:
        w["QQQ"] = eq_w * 0.45

    # Dynamic hedge
    hedge = ((8 - score) / 8).clip(lower=0.03, upper=0.5)
    if "GLD" in w.columns:
        w["GLD"] = hedge * 0.4
    if "IEF" in w.columns:
        w["IEF"] = hedge * 0.3

    # Apply trade management (stops)
    w = apply_trade_management(prices, w, stop_pct=0.025, trail_pct=0.035,
                               profit_target=0.06)
    return w


# ── Audit ──
def audit(name, weights, prices, result):
    issues = 0
    ret_df = prices.pct_change()
    for col in weights.columns:
        if col in ret_df.columns and weights[col].abs().sum() > 0:
            c = weights[col].corr(ret_df[col])
            if abs(c) > 0.25:
                print(f"    ⚠️  {name}/{col}: lookahead corr={c:.4f}")
                issues += 1

    r = result["portfolio_returns"]
    excess = r - RF / 252
    sc = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    if abs(sc - result["metrics"]["Sharpe Ratio"]) > 0.05:
        print(f"    ⚠️  {name}: Sharpe mismatch ({sc:.4f} vs {result['metrics']['Sharpe Ratio']:.4f})")
        issues += 1

    eq = result["equity_curve"]
    yrs = len(r) / 252
    tr = eq.iloc[-1] / 100_000 - 1
    cc = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
    if abs(cc - result["metrics"]["CAGR"]) > 0.005:
        print(f"    ⚠️  {name}: CAGR mismatch ({cc*100:.2f}% vs {result['metrics']['CAGR']*100:.2f}%)")
        issues += 1

    m = result["metrics"]
    if m["CAGR"] > 0.80:
        print(f"    ⚠️  CAGR {m['CAGR']*100:.1f}% unrealistic")
        issues += 1
    if m["Sharpe Ratio"] > 4.5:
        print(f"    ⚠️  Sharpe {m['Sharpe Ratio']:.2f} unrealistic")
        issues += 1
    return issues


def main():
    print("=" * 90)
    print("LEVERAGED ALPHA v4 — L/S + STOPS + SECTOR DIVERSIFICATION + MULTI-ATOM ENSEMBLE")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  {START} to {END}")
    print("=" * 90)

    prices = load_data()
    available = [t for t in ALL_TICKERS if t in prices.columns]
    print(f"\n📊 {len(available)} tickers loaded: {available}")
    print(f"   {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(prices)/252:.1f}yr)")

    # Note which tickers have full history
    for t in available:
        first_valid = prices[t].first_valid_index()
        if first_valid and first_valid > prices.index[60]:
            print(f"   ⚠️  {t} starts at {first_valid.date()} (not full history)")

    # SPY benchmark
    spy_r = prices["SPY"].pct_change().dropna()
    spy_eq = 100_000 * (1 + spy_r).cumprod()
    spy_m = compute_metrics(spy_r, spy_eq, 100_000, risk_free_rate=RF)
    print(f"\n📈 SPY: CAGR={spy_m['CAGR']*100:.2f}%, Sharpe={spy_m['Sharpe Ratio']:.4f}, "
          f"MaxDD={spy_m['Max Drawdown']*100:.2f}%")

    # ── Run strategies ──
    results = {}

    # 1. Long/Short Scoring (build on v3's best)
    print(f"\n{'─'*70}")
    print("🔄 Strategy: Long/Short Scoring")
    for max_lev, short_scale in [(1.5, 0.3), (1.5, 0.5), (1.8, 0.5), (2.0, 0.5), (2.0, 0.7)]:
        label = f"LS_Scoring(lev={max_lev}, short={short_scale})"
        w = strategy_long_short_scoring(prices, max_lev, short_scale)
        r = run_bt(prices, w)
        results[label] = (w, r)
        m = r["metrics"]
        bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
        bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
        star = " ⭐" if bc == "✓" and bs == "✓" else ""
        print(f"   {label:<50} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
              f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] "
              f"MaxDD={m['Max Drawdown']*100:>7.2f}% AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

    # 2. Multi-Atom Ensemble (without trade management)
    print(f"\n{'─'*70}")
    print("🔄 Strategy: Multi-Atom Ensemble")
    w_ens, atom_names = strategy_multi_atom_ensemble(prices, with_trade_mgmt=False)
    r_ens = run_bt(prices, w_ens)
    results["MultiAtom_NoStops"] = (w_ens, r_ens)
    m = r_ens["metrics"]
    print(f"   {'NoStops':<50} CAGR={m['CAGR']*100:>7.2f}% "
          f"Sharpe={m['Sharpe Ratio']:>7.4f} MaxDD={m['Max Drawdown']*100:>7.2f}% "
          f"AvgLev={m['Avg Gross Leverage']:>5.2f}x")
    print(f"   ({len(atom_names)} atoms: {', '.join(atom_names[:8])}...)")

    # 2b. With trade management
    w_ens_tm, _ = strategy_multi_atom_ensemble(prices, with_trade_mgmt=True)
    r_ens_tm = run_bt(prices, w_ens_tm)
    results["MultiAtom_WithStops"] = (w_ens_tm, r_ens_tm)
    m = r_ens_tm["metrics"]
    print(f"   {'WithStops':<50} CAGR={m['CAGR']*100:>7.2f}% "
          f"Sharpe={m['Sharpe Ratio']:>7.4f} MaxDD={m['Max Drawdown']*100:>7.2f}% "
          f"AvgLev={m['Avg Gross Leverage']:>5.2f}x")

    # 2c. Scaled versions of multi-atom
    for scale in [1.3, 1.5, 2.0]:
        w_scaled = w_ens * scale
        r_scaled = run_bt(prices, w_scaled)
        label = f"MultiAtom_x{scale}"
        results[label] = (w_scaled, r_scaled)
        m = r_scaled["metrics"]
        bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
        bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
        star = " ⭐" if bc == "✓" and bs == "✓" else ""
        print(f"   {label:<50} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
              f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] MaxDD={m['Max Drawdown']*100:>7.2f}% "
              f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

    # 3. Concentrated L/S with Stops
    print(f"\n{'─'*70}")
    print("🔄 Strategy: Concentrated L/S with Trade Management")
    for lev in [1.0, 1.3, 1.5, 1.8, 2.0]:
        label = f"ConcentratedLS_Stops(lev={lev})"
        w = strategy_concentrated_ls_with_stops(prices, leverage=lev)
        r = run_bt(prices, w)
        results[label] = (w, r)
        m = r["metrics"]
        bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
        bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
        star = " ⭐" if bc == "✓" and bs == "✓" else ""
        print(f"   {label:<50} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
              f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] MaxDD={m['Max Drawdown']*100:>7.2f}% "
              f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

    # 4. Hybrid: LS_Scoring + MultiAtom ensemble
    print(f"\n{'─'*70}")
    print("🔄 Strategy: Hybrid ensembles")
    w_ls = strategy_long_short_scoring(prices, 1.5, 0.5)
    w_conc = strategy_concentrated_ls_with_stops(prices, 1.5)
    common = w_ens.columns.intersection(w_ls.columns).intersection(w_conc.columns)

    for name, blend in [
        ("LSScoring+MultiAtom", (w_ls[common] * 0.5 + w_ens[common] * 0.5)),
        ("LSScoring+MultiAtom+Conc", (w_ls[common] * 0.33 + w_ens[common] * 0.34 + w_conc[common] * 0.33)),
        ("LSScoring+MultiAtom_x1.5", (w_ls[common] * 0.5 + w_ens[common] * 0.5) * 1.5),
        ("All3_x1.5", (w_ls[common] * 0.33 + w_ens[common] * 0.34 + w_conc[common] * 0.33) * 1.5),
    ]:
        r = run_bt(prices, blend)
        results[name] = (blend, r)
        m = r["metrics"]
        bc = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
        bs = "✓" if m["Sharpe Ratio"] > 1.95 else " "
        star = " ⭐" if bc == "✓" and bs == "✓" else ""
        print(f"   {name:<50} CAGR={m['CAGR']*100:>7.2f}%[{bc}] "
              f"Sharpe={m['Sharpe Ratio']:>7.4f}[{bs}] MaxDD={m['Max Drawdown']*100:>7.2f}% "
              f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

    # ── Audit ──
    print(f"\n{'='*90}")
    print("🔍 AUDIT")
    total_issues = 0
    for name, (w, r) in results.items():
        total_issues += audit(name, w, prices, r)
    print(f"   Total issues: {total_issues}")
    if total_issues == 0:
        print("   ✅ All audits pass")

    # ── Strategy correlation (of top results) ──
    print(f"\n{'='*90}")
    print("📊 Top strategy correlations:")
    sorted_r = sorted(results.items(), key=lambda x: x[1][1]["metrics"]["Sharpe Ratio"], reverse=True)
    top_names = [n for n, _ in sorted_r[:8]]
    corr_data = {n: results[n][1]["portfolio_returns"] for n in top_names}
    corr = pd.DataFrame(corr_data).corr()
    # Abbreviate names for display
    short_names = {n: n[:20] for n in top_names}
    corr.columns = [short_names[n] for n in top_names]
    corr.index = [short_names[n] for n in top_names]
    print(corr.round(3).to_string())

    # ── Sub-period analysis ──
    print(f"\n{'='*90}")
    print("📅 SUB-PERIOD ANALYSIS (top 5 strategies)")
    top5 = [(n, results[n]) for n in [n for n, _ in sorted_r[:5]]]

    for pname, s, e in [("Full", None, None), ("2015-25", "2015-01-01", None),
                         ("2018-25", "2018-01-01", None), ("2020-25", "2020-01-01", None)]:
        sp = prices.copy()
        if s: sp = sp[sp.index >= s]
        if e: sp = sp[sp.index <= e]
        sr = sp["SPY"].pct_change().dropna()
        se_eq = 100_000 * (1 + sr).cumprod()
        sm = compute_metrics(sr, se_eq, 100_000, risk_free_rate=RF)

        print(f"\n  {pname}: SPY CAGR={sm['CAGR']*100:.2f}%, Sharpe={sm['Sharpe Ratio']:.4f}")
        for name, (orig_w, _) in top5:
            # Re-generate weights for sub-period is complex for some strategies.
            # Instead, just slice the original returns.
            orig_ret = results[name][1]["portfolio_returns"]
            sub_ret = orig_ret[orig_ret.index.isin(sp.index)]
            if len(sub_ret) < 100:
                continue
            sub_eq = 100_000 * (1 + sub_ret).cumprod()
            sub_m = compute_metrics(sub_ret, sub_eq, 100_000, risk_free_rate=RF)
            flags = []
            if sub_m["CAGR"] > sm["CAGR"]: flags.append("CAGR✓")
            if sub_m["Sharpe Ratio"] > 1.95: flags.append("Sharpe✓")
            if len(flags) == 2: flags.append("⭐")
            print(f"  {name[:35]:<37} CAGR={sub_m['CAGR']*100:>7.2f}% "
                  f"Sharpe={sub_m['Sharpe Ratio']:>7.4f} MaxDD={sub_m['Max Drawdown']*100:>7.2f}% "
                  f"{' '.join(flags)}")

    # ── Final summary ──
    print(f"\n{'='*90}")
    print("📊 FINAL RESULTS (sorted by Sharpe)")
    print(f"{'Label':<55} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'AvgLev':>6}")
    print("─" * 82)
    print(f"{'SPY':<55} {spy_m['CAGR']*100:>6.2f}% {spy_m['Sharpe Ratio']:>7.4f} "
          f"{spy_m['Max Drawdown']*100:>6.2f}% {'1.00x':>6}")
    print("─" * 82)

    winners = []
    for label, (_, res) in sorted_r:
        m = res["metrics"]
        star = " ⭐" if m["CAGR"] > spy_m["CAGR"] and m["Sharpe Ratio"] > 1.95 else ""
        if star:
            winners.append((label, m))
        print(f"{label:<55} {m['CAGR']*100:>6.2f}% {m['Sharpe Ratio']:>7.4f} "
              f"{m['Max Drawdown']*100:>6.2f}% {m['Avg Gross Leverage']:>5.2f}x{star}")

    if winners:
        print(f"\n🏆 WINNERS:")
        for l, m in winners:
            print(f"   ⭐ {l}: CAGR={m['CAGR']*100:.2f}%, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']*100:.2f}%")
    else:
        print(f"\n⚠️  No full-period winners yet.")
        sorted_by_score = sorted(sorted_r, key=lambda x: (
            min(x[1][1]["metrics"]["CAGR"] / spy_m["CAGR"], 1.5) +
            min(x[1][1]["metrics"]["Sharpe Ratio"] / 1.95, 1.5)
        ), reverse=True)
        print("   Closest to both targets:")
        for label, (_, res) in sorted_by_score[:5]:
            m = res["metrics"]
            print(f"   {label}: CAGR={m['CAGR']*100:.2f}%, Sharpe={m['Sharpe Ratio']:.4f}")

    print(f"\nAudit: {total_issues} issues")
    print("=" * 90)
    return results, spy_m, prices


if __name__ == "__main__":
    main()
