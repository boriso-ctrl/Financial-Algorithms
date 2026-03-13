"""
V9 Optimizer — builds on V8 per-asset configs, grid-searches new V9 params.

V9 new levers (from research doc + strategy audit):
  di_filter        — require DI+ > DI- before buying (directional confirmation)
  obv_filter       — require 5-day OBV slope positive (volume confirms price)
  enable_stoch_rsi — add StochRSI < 0.20 buy signal (more sensitive MR entry)
  enable_bb_signal — add BB lower-band bounce buy signal (more MR entries)
  partial_qty_pct  — fraction of position taken at partial TP (0.33/0.50/0.67)
  vol_regime_scale — multiply vol_target by this in confirmed uptrend
  min_vol_ratio    — minimum volume ratio required for a new entry
  reentry_cooldown — bars to wait after stop-out before next entry

Scoring:  sharpe * 0.55 + (cagr/30) * 0.45  × trade-frequency ratio
          HARD FILTER: both sharpe AND cagr must beat V8 baseline.

Usage:
    python intraday/strategies/v9_optimizer.py
"""

import sys
import os
import json
import itertools
import time
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from aggressive_hybrid_v6_10yr import AggressiveHybridV6

# ---------------------------------------------------------------------------
# V8 baseline configs (must beat these on BOTH sharpe AND cagr)
# ---------------------------------------------------------------------------
V8_CONFIGS = {
    'QQQ': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=3.0, partial_tp_mult=1.5,
        dd_reduce=0.12, dd_halt=0.20,
        rsi_period=9,  rsi_oversold=38, rsi_overbought=65,
        atr_period=14, ema_trend=30, adx_thresh=18,
        min_strength_up=0.30, min_strength_bear=0.35,
        trail_cushion=1.5, post_partial_mult=1.5,
        macd_fast=8, macd_slow=38, macd_sig=9,
        max_hold_trend=60, max_hold_mr=25,
    ),
    'GBTC': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=4.5, partial_tp_mult=1.0,
        dd_reduce=0.12, dd_halt=0.20,
        rsi_period=9,  rsi_oversold=33, rsi_overbought=65,
        atr_period=14, ema_trend=50, adx_thresh=27,
        min_strength_up=0.30, min_strength_bear=0.35,
        trail_cushion=0.5, post_partial_mult=2.5,
        macd_fast=8, macd_slow=38, macd_sig=9,
        max_hold_trend=30, max_hold_mr=25,
    ),
    'XLK': dict(
        trail_atr=3.0, vol_target=0.22, tp_mult=4.5, partial_tp_mult=1.5,
        dd_reduce=0.12, dd_halt=0.20,
        rsi_period=14, rsi_oversold=38, rsi_overbought=65,
        atr_period=14, ema_trend=80, adx_thresh=22,
        min_strength_up=0.20, min_strength_bear=0.35,
        trail_cushion=1.0, post_partial_mult=2.0,
        macd_fast=8, macd_slow=26, macd_sig=9,
        max_hold_trend=80, max_hold_mr=25,
    ),
    'NVDA': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=3.0, partial_tp_mult=1.5,
        dd_reduce=0.12, dd_halt=0.20,
        rsi_period=9,  rsi_oversold=38, rsi_overbought=65,
        atr_period=20, ema_trend=50, adx_thresh=27,
        min_strength_up=0.25, min_strength_bear=0.35,
        trail_cushion=2.0, post_partial_mult=2.0,
        macd_fast=12, macd_slow=26, macd_sig=9,
        max_hold_trend=60, max_hold_mr=25,
    ),
}

# V8 results baseline — optimizer must beat BOTH sharpe AND cagr
V8_BASELINE = {
    'QQQ':  {'sharpe': 1.63, 'cagr': 14.5, 'tpy': 110},
    'GBTC': {'sharpe': 1.32, 'cagr':  6.4, 'tpy':  98},
    'XLK':  {'sharpe': 1.92, 'cagr': 20.0, 'tpy': 117},
    'NVDA': {'sharpe': 2.09, 'cagr': 12.4, 'tpy':  92},
}

# ---------------------------------------------------------------------------
# V9 parameter grid — layered on top of V8 configs
# ---------------------------------------------------------------------------
V9_GRID = {
    'di_filter':        [False, True],
    'obv_filter':       [False, True],
    'enable_stoch_rsi': [False, True],
    'enable_bb_signal': [False, True],
    'partial_qty_pct':  [0.33, 0.50, 0.67],
    'vol_regime_scale': [1.0, 1.05, 1.10],
    'min_vol_ratio':    [0.0, 0.80, 1.00],
    'reentry_cooldown': [0, 2, 3],
}

START = '2015-01-01'
END   = '2025-12-31'

# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------
def score_result(result, ticker):
    """
    Score formula rewards Sharpe + CAGR proportionally, multiplied by
    trade frequency ratio vs V8 baseline (bonus for MORE trades, penalty <V8).
    Returns -999 if either Sharpe or CAGR doesn't beat V8.
    """
    if 'error' in result:
        return -999.0

    sharpe = result.get('sharpe', 0)
    cagr   = result.get('cagr',   0)
    trades = result.get('trades', 0)
    years  = result.get('years',  1)

    v8 = V8_BASELINE[ticker]

    # Hard requirement: must beat V8 on both Sharpe AND CAGR
    if sharpe < v8['sharpe'] or cagr < v8['cagr']:
        return -999.0

    tpy       = trades / max(years, 0.5)
    tpy_ratio = max(0.5, min(1.5, tpy / max(v8['tpy'], 1)))  # bonus for more trades

    base_score = sharpe * 0.55 + (cagr / 30.0) * 0.45
    return base_score * tpy_ratio


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------
def run_backtest(args):
    ticker, v8_cfg, v9_params = args
    try:
        full_cfg = {**v8_cfg, **v9_params}
        trader = AggressiveHybridV6(
            ticker=ticker,
            start=START,
            end=END,
            **full_cfg,
        )
        if not trader.fetch_data():
            return None, v9_params, -999.0
        result = trader.backtest()
        sc = score_result(result, ticker)
        return result, v9_params, sc
    except Exception as e:
        return None, v9_params, -999.0


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------
def optimize_ticker(ticker):
    v8_cfg = V8_CONFIGS[ticker]
    v8_bl  = V8_BASELINE[ticker]

    keys   = list(V9_GRID.keys())
    combos = list(itertools.product(*[V9_GRID[k] for k in keys]))
    total  = len(combos)

    param_list = [dict(zip(keys, c)) for c in combos]
    tasks      = [(ticker, v8_cfg, p) for p in param_list]

    print(f"\n{'='*60}")
    print(f"{ticker}: {total} combinations | V8 baseline → "
          f"Sharpe {v8_bl['sharpe']:.2f}  CAGR {v8_bl['cagr']:.1f}%  "
          f"{v8_bl['tpy']:.0f} trades/yr")
    print(f"{'='*60}")

    t0      = time.time()
    results = []

    n_proc = min(cpu_count(), 6)
    with Pool(processes=n_proc) as pool:
        for i, (res, params, sc) in enumerate(pool.imap_unordered(run_backtest, tasks)):
            if sc > 0:
                results.append((sc, res, params))
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                print(f"  {i+1}/{total}  ({rate:.1f}/s)  "
                      f"valid so far: {len(results)}", flush=True)

    elapsed = time.time() - t0
    print(f"  Finished {total} runs in {elapsed:.0f}s. Valid: {len(results)}")

    if not results:
        print(f"  ❌ No valid configs found that beat V8 baseline for {ticker}")
        return None, None

    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_result, best_params = results[0]

    print(f"\n  ✅ Best V9 for {ticker}:")
    print(f"     Score:    {best_score:.4f}")
    print(f"     Sharpe:   {best_result['sharpe']:.2f}  (V8: {v8_bl['sharpe']:.2f})")
    print(f"     CAGR:     {best_result['cagr']:.2f}%  (V8: {v8_bl['cagr']:.1f}%)")
    print(f"     Trades/yr:{best_result['trades']/best_result['years']:.0f}  "
          f"(V8: {v8_bl['tpy']:.0f})")
    print(f"     MaxDD:    {best_result['max_dd']:.1f}%")
    print(f"     Win rate: {best_result['win_rate']:.1f}%")
    print(f"     Params:   {best_params}")

    # Top-5 summary
    print(f"\n  Top-5 valid configs for {ticker}:")
    for rank, (sc, res, params) in enumerate(results[:5], 1):
        tpy = res['trades'] / res['years']
        print(f"    #{rank}  score={sc:.4f}  sharpe={res['sharpe']:.2f}  "
              f"cagr={res['cagr']:.1f}%  trades/yr={tpy:.0f}  "
              f"dd={res['max_dd']:.1f}%")

    return best_result, best_params


def main():
    tickers = ['QQQ', 'GBTC', 'XLK', 'NVDA']

    all_results = {}
    all_params  = {}

    for ticker in tickers:
        best_result, best_params = optimize_ticker(ticker)
        if best_result and best_params:
            all_results[ticker] = best_result
            all_params[ticker]  = {**V8_CONFIGS[ticker], **best_params}

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("V9 OPTIMIZATION COMPLETE — FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ticker':<6}  {'Sharpe':>6}  {'CAGR':>6}  {'Trades/yr':>9}  {'MaxDD':>6}  {'WinRate':>7}  Beat V8?")
    print("-" * 60)
    for ticker in tickers:
        if ticker in all_results:
            r   = all_results[ticker]
            v8  = V8_BASELINE[ticker]
            tpy = r['trades'] / r['years']
            ok  = "✅" if r['sharpe'] > v8['sharpe'] and r['cagr'] > v8['cagr'] else "❌"
            print(f"{ticker:<6}  {r['sharpe']:>6.2f}  {r['cagr']:>5.1f}%  "
                  f"{tpy:>9.0f}  {r['max_dd']:>5.1f}%  {r['win_rate']:>6.1f}%  {ok}")
        else:
            v8 = V8_BASELINE[ticker]
            print(f"{ticker:<6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>9}  {'N/A':>6}  {'N/A':>7}  ❌ No improvement")

    print()
    print("V8 baseline for comparison:")
    print(f"{'Ticker':<6}  {'Sharpe':>6}  {'CAGR':>6}  {'Trades/yr':>9}")
    print("-" * 35)
    for ticker in tickers:
        v8 = V8_BASELINE[ticker]
        print(f"{ticker:<6}  {v8['sharpe']:>6.2f}  {v8['cagr']:>5.1f}%  {v8['tpy']:>9.0f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    out = {
        'best_params':  {t: {k: v for k, v in p.items()} for t, p in all_params.items()},
        'best_results': {t: r for t, r in all_results.items()},
        'v8_baseline':  V8_BASELINE,
    }
    out_path = os.path.join(out_dir, 'v9_optimization_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")

    # Print paper_trader ASSET_CONFIGS snippet
    print("\n" + "="*60)
    print("ASSET_CONFIGS snippet for paper_trader.py:")
    print("="*60)
    for ticker in tickers:
        if ticker in all_params:
            p = all_params[ticker]
            print(f"\n'{ticker}': dict(")
            for k, v in p.items():
                print(f"    {k}={repr(v)},")
            print("),")


if __name__ == '__main__':
    main()
