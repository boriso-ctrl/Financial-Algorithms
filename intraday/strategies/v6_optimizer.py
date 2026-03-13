"""
V6 Focused Optimizer — QQQ, GBTC, XLK, NVDA
=============================================
Grid-searches key parameters to maximise Sharpe over 10 years.

Parameters tuned per asset:
  trail_atr      : trailing stop width (ATR multiplier)
  vol_target     : annualised volatility target for position sizing
  tp_mult        : take-profit distance (ATR multiplier)
  partial_tp_mult: first partial TP level (ATR multiplier)

Score = Sharpe + CAGR/25
  Balances risk-adjusted return with raw return.
  Assets with >200 trades/yr are preferred (penalise thin-edge setups).

Grid size: 4 x 4 x 4 x 2 = 128 combos, x 4 assets = 512 runs (~20 min)
"""

import sys
import os
import json
import itertools
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

logging.basicConfig(level=logging.WARNING)

START = '2015-01-01'
END   = '2025-12-31'

ASSETS = {
    'QQQ':  'NASDAQ 100',
    'GBTC': 'Bitcoin Trust (10yr)',
    'XLK':  'Technology ETF',
    'NVDA': 'NVIDIA',
}

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
GRID = {
    'trail_atr':       [2.0, 2.5, 3.0, 3.5],
    'vol_target':      [0.12, 0.15, 0.18, 0.22],
    'tp_mult':         [3.0, 3.5, 4.0, 4.5],
    'partial_tp_mult': [1.0, 1.5],
}


def score(r):
    """Composite scoring: Sharpe + CAGR/25, penalise <50 trades/yr."""
    if 'error' in r:
        return -99
    s     = r.get('sharpe', 0)
    c     = r.get('cagr', 0)
    tpy   = r.get('trades', 0) / max(r.get('years', 1), 0.5)
    # Penalise very thin edges (too few trades = over-fitted to a few events)
    trade_pen = min(tpy / 50, 1.0)   # ramps up to 1.0 at 50+ trades/yr
    return (s + c / 25) * trade_pen


def run_config(ticker, trail_atr, vol_target, tp_mult, partial_tp_mult, data_cache):
    """Run one backtest config; reuse pre-fetched data if available."""
    try:
        trader = AggressiveHybridV6(
            ticker=ticker, start=START, end=END,
            trail_atr=trail_atr, vol_target=vol_target,
            tp_mult=tp_mult, partial_tp_mult=partial_tp_mult,
        )
        # Reuse cached data to skip re-downloading
        if ticker in data_cache:
            trader.data = data_cache[ticker]['data'].copy()
            trader.vix  = data_cache[ticker]['vix'].copy()
        else:
            if not trader.fetch_data():
                return {'error': 'fetch failed'}
            data_cache[ticker] = {'data': trader.data.copy(), 'vix': trader.vix.copy()}

        results = trader.backtest()
        if results is None:
            return {'error': 'no result'}
        results['config'] = {
            'trail_atr': trail_atr, 'vol_target': vol_target,
            'tp_mult': tp_mult, 'partial_tp_mult': partial_tp_mult,
        }
        return results
    except Exception as e:
        return {'error': str(e)}


# ---------------------------------------------------------------------------
def optimize_asset(ticker, label, data_cache):
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))

    print(f"\n{'=' * 60}")
    print(f"  Optimizing {ticker} ({label}) — {len(combos)} configurations")
    print(f"{'=' * 60}")

    best_score  = -999
    best_result = None
    best_config = None

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        r = run_config(ticker, **params, data_cache=data_cache)
        sc = score(r)
        if sc > best_score:
            best_score  = sc
            best_result = r
            best_config = params

        if i % 32 == 0:
            prog = f"{i}/{len(combos)}"
            if best_result and 'error' not in best_result:
                print(f"  [{prog}] Best so far: Sharpe {best_result['sharpe']:.2f} "
                      f"CAGR {best_result['cagr']:.1f}%  "
                      f"config={best_config}", flush=True)

    return best_result, best_config, best_score


# ---------------------------------------------------------------------------
def print_result(ticker, label, result, config):
    if not result or 'error' in result:
        print(f"  {ticker}: FAILED — {result.get('error','?')}")
        return
    print(f"\n{'=' * 60}")
    print(f"  BEST CONFIG: {ticker} ({label})")
    print(f"{'=' * 60}")
    print(f"  trail_atr       = {config['trail_atr']}")
    print(f"  vol_target      = {config['vol_target']}")
    print(f"  tp_mult         = {config['tp_mult']}")
    print(f"  partial_tp_mult = {config['partial_tp_mult']}")
    print(f"  ---")
    print(f"  Sharpe:   {result['sharpe']:.2f}  |  Sortino: {result['sortino']:.2f}")
    print(f"  CAGR:     {result['cagr']:.2f}%  |  Return: {result['return_pct']:.1f}%")
    print(f"  Max DD:   {result['max_dd']:.2f}%  |  Calmar: {result['calmar_ratio']:.2f}")
    print(f"  Win Rate: {result['win_rate']:.1f}%  |  PF: {result['profit_factor']:.2f}")
    tpy = result['trades'] / max(result['years'], 0.5)
    print(f"  Trades:   {result['trades']} ({tpy:.0f}/yr)  |  Avg Hold: {result['avg_hold_days']:.1f}d")
    print(f"  Long WR:  {result.get('long_wr', 0):.1f}%   Short WR: {result.get('short_wr', 0):.1f}%")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print(f"  V6 OPTIMIZER  |  QQQ + GBTC + XLK + NVDA  |  10yr")
    print(f"  Grid: trail_atr={GRID['trail_atr']}")
    print(f"        vol_target={GRID['vol_target']}")
    print(f"        tp_mult={GRID['tp_mult']}")
    print(f"        partial_tp_mult={GRID['partial_tp_mult']}")
    total_runs = len(list(itertools.product(*GRID.values()))) * len(ASSETS)
    print(f"  Total runs: {total_runs}")
    print(f"{'=' * 60}")

    data_cache   = {}
    final_results = {}

    for ticker, label in ASSETS.items():
        result, config, sc = optimize_asset(ticker, label, data_cache)
        final_results[ticker] = {'result': result, 'config': config, 'score': sc}
        print_result(ticker, label, result, config)

    # -----------------------------------------------------------------------
    # Summary comparison
    print(f"\n\n{'=' * 72}")
    print(f"  OPTIMIZED RESULTS SUMMARY  (V6, 10yr, long-only)")
    print(f"{'=' * 72}")
    print(f"  {'Ticker':<6}  {'CAGR%':>6}  {'Sharpe':>7}  {'Sortino':>7}  "
          f"{'MaxDD%':>6}  {'WR%':>5}  {'Trades/yr':>10}  {'Calmar':>7}")
    print(f"  {'-'*66}")
    for ticker in ASSETS:
        r = final_results[ticker]['result']
        if r and 'error' not in r:
            tpy = r['trades'] / max(r['years'], 0.5)
            print(f"  {ticker:<6}  {r['cagr']:>6.1f}  {r['sharpe']:>7.2f}  "
                  f"{r['sortino']:>7.2f}  {r['max_dd']:>6.1f}  {r['win_rate']:>5.1f}  "
                  f"{tpy:>10.0f}  {r['calmar_ratio']:>7.2f}")
    print(f"  {'-'*66}")

    # -----------------------------------------------------------------------
    # Save best configs
    save = {
        ticker: {
            'config':  final_results[ticker]['config'],
            'metrics': final_results[ticker]['result'],
        }
        for ticker in ASSETS
        if final_results[ticker]['result'] and 'error' not in final_results[ticker]['result']
    }
    out = 'intraday/results/v6_optimized_configs.json'
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nBest configs saved -> {out}")
    print("\nINSTRUCTIONS: Copy the best config for each asset into your")
    print("live runner. Each asset should use its own tuned parameters.")
