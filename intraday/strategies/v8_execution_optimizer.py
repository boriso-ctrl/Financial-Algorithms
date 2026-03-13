"""
V8 Per-Asset Execution Structure Optimizer
==========================================
Optimises the trade management / SL-TP structure for each asset individually,
building on top of the V6 numeric params + V7 signal params already found.

Parameters tuned per asset:
  trail_cushion    : ATR gain needed before trailing stop activates
                     (0.5=tight, 1.0=standard, 1.5=give it room, 2.0=wide)
  post_partial_mult: after partial TP hits, where to retarget the remainder
                     (1.5=quick, 2.0=standard, 2.5=default, 3.0=let it run)
  macd_fast        : MACD fast EMA period (8=reactive, 12=standard, 19=slow)
  macd_slow        : MACD slow EMA period (17, 26, 38)
  max_hold_trend   : max days for trend signals (30, 45, 60, 80)
  max_hold_mr      : max days for mean-reversion signals (10, 15, 25, 35)

Grid: 4 x 4 x 3 x 3 x 4 x 4 = 2304 combos per asset × 4 = 9216 total
      (~40-55 min total, all data reused from cache)

Asset-specific expectations:
  QQQ  : moderate everything — ETF mean-reversion works best with standard MACDs
  GBTC : fast MACD (8/17), tight trail cushion — crypto moves explosively
  XLK  : slow MACD (19/38), long max hold — tech ETF trends last months
  NVDA : medium MACD, low trail cushion — high-beta single stock, take profit fast

Outputs: intraday/results/v8_execution_configs.json
         Contains V6 + V7 + V8 params as a single 'full_config' per asset.
"""

import sys
import os
import json
import itertools
import logging

sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

logging.basicConfig(level=logging.WARNING)

START = '2015-01-01'
END   = '2025-12-31'

ASSETS = {
    'QQQ':  'NASDAQ 100 ETF',
    'GBTC': 'Bitcoin Trust',
    'XLK':  'Technology ETF',
    'NVDA': 'NVIDIA',
}

# ---------------------------------------------------------------------------
# Load V7 best configs (V6 numeric + V7 signal) — these are FIXED
# ---------------------------------------------------------------------------
V7_CONFIGS_PATH = 'intraday/results/v7_signal_configs.json'

# Hardcoded fallback in case file is missing
DEFAULT_V7_PARAMS = {
    'QQQ': {
        'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 3.0, 'partial_tp_mult': 1.5,
        'rsi_period': 9, 'rsi_oversold': 38, 'atr_period': 14,
        'ema_trend': 30, 'adx_thresh': 18, 'min_strength_up': 0.30,
    },
    'GBTC': {
        'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 4.5, 'partial_tp_mult': 1.0,
        'rsi_period': 9, 'rsi_oversold': 33, 'atr_period': 14,
        'ema_trend': 50, 'adx_thresh': 27, 'min_strength_up': 0.30,
    },
    'XLK': {
        'trail_atr': 3.0, 'vol_target': 0.22, 'tp_mult': 4.5, 'partial_tp_mult': 1.5,
        'rsi_period': 14, 'rsi_oversold': 38, 'atr_period': 14,
        'ema_trend': 80, 'adx_thresh': 22, 'min_strength_up': 0.20,
    },
    'NVDA': {
        'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 3.0, 'partial_tp_mult': 1.5,
        'rsi_period': 9, 'rsi_oversold': 38, 'atr_period': 20,
        'ema_trend': 50, 'adx_thresh': 27, 'min_strength_up': 0.25,
    },
}


def load_v7_params():
    try:
        with open(V7_CONFIGS_PATH) as f:
            data = json.load(f)
        params = {}
        for ticker, entry in data.items():
            fc = entry.get('full_config', {})
            if fc:
                params[ticker] = fc
            else:
                params[ticker] = DEFAULT_V7_PARAMS.get(ticker, {})
        for t, v in DEFAULT_V7_PARAMS.items():
            if t not in params:
                params[t] = v
        return params
    except Exception:
        return DEFAULT_V7_PARAMS.copy()


# ---------------------------------------------------------------------------
# V8 execution structure grid
# ---------------------------------------------------------------------------
EXEC_GRID = {
    # Trail cushion: how much gain (×ATR) before trailing stop activates
    # Smaller = faster trail activation (lock in profit quickly)
    # Larger  = give price more room before trailing kicks in
    'trail_cushion':    [0.5, 1.0, 1.5, 2.0],

    # After partial TP, re-target the remaining half position at:
    #   entry + trail_atr × atr × post_partial_mult
    # Smaller = take profit quickly on remainder
    # Larger  = let runner run further
    'post_partial_mult': [1.5, 2.0, 2.5, 3.0],

    # MACD fast / slow — independent pairs to keep signal logic coherent
    # Tested as (fast, slow) pairs:
    'macd_fast': [8, 12, 19],
    'macd_slow': [17, 26, 38],

    # Max hold by signal type
    'max_hold_trend':   [30, 45, 60, 80],
    'max_hold_mr':      [10, 15, 25, 35],
}

# Total: 4 × 4 × 3 × 3 × 4 × 4 = 2304 per asset


def score(r):
    """
    Composite score — same formula as V7 with added Calmar weight.
    Calmar is particularly important for V8 since we're tuning the
    exit structure which directly controls drawdown.
    """
    if 'error' in r:
        return -99
    sharpe = r.get('sharpe', 0)
    cagr   = r.get('cagr', 0)
    calmar = r.get('calmar_ratio', 0)
    trades = r.get('trades', 0)
    years  = max(r.get('years', 1), 0.5)
    tpy    = trades / years
    trade_pen = min(tpy / 50, 1.0)
    # Slightly higher Calmar weight vs V7 — exit structure directly controls DD
    return (sharpe + cagr / 25 + calmar * 0.15) * trade_pen


def run_config(ticker, v7_params, exec_params, data_cache):
    """Run one backtest, reusing pre-fetched data."""
    try:
        # Guard: macd_fast must be < macd_slow
        if exec_params['macd_fast'] >= exec_params['macd_slow']:
            return {'error': 'invalid_macd_pair'}

        trader = AggressiveHybridV6(
            ticker=ticker, start=START, end=END,
            # V6 numeric params
            trail_atr=v7_params.get('trail_atr', 2.5),
            vol_target=v7_params.get('vol_target', 0.15),
            tp_mult=v7_params.get('tp_mult', 3.5),
            partial_tp_mult=v7_params.get('partial_tp_mult', 1.5),
            # V7 signal params
            rsi_period=v7_params.get('rsi_period', 14),
            rsi_oversold=v7_params.get('rsi_oversold', 35),
            atr_period=v7_params.get('atr_period', 14),
            ema_trend=v7_params.get('ema_trend', 50),
            adx_thresh=v7_params.get('adx_thresh', 25),
            min_strength_up=v7_params.get('min_strength_up', 0.25),
            # V8 execution structure params
            trail_cushion=exec_params['trail_cushion'],
            post_partial_mult=exec_params['post_partial_mult'],
            macd_fast=exec_params['macd_fast'],
            macd_slow=exec_params['macd_slow'],
            max_hold_trend=exec_params['max_hold_trend'],
            max_hold_mr=exec_params['max_hold_mr'],
        )
        # Reuse cached raw data
        if ticker in data_cache:
            trader.data = data_cache[ticker]['data'].copy()
            trader.vix  = data_cache[ticker]['vix'].copy()
        else:
            if not trader.fetch_data():
                return {'error': 'fetch failed'}
            data_cache[ticker] = {
                'data': trader.data.copy(),
                'vix':  trader.vix.copy(),
            }

        result = trader.backtest()
        return result if result else {'error': 'no result'}
    except Exception as e:
        return {'error': str(e)}


# ---------------------------------------------------------------------------
def optimize_asset(ticker, label, v7_params, data_cache):
    keys   = list(EXEC_GRID.keys())
    values = list(EXEC_GRID.values())
    combos = list(itertools.product(*values))

    print(f"\n{'=' * 64}")
    print(f"  Optimizing {ticker} ({label})")
    print(f"  V7 base: trail={v7_params.get('trail_atr')} ema={v7_params.get('ema_trend')} "
          f"adx={v7_params.get('adx_thresh')} rsi={v7_params.get('rsi_period')}")
    print(f"  Exec grid: {len(combos)} combos  (invalid macd pairs skipped)")
    print(f"{'=' * 64}")

    best_score  = -999
    best_result = None
    best_exec   = None
    run_count   = 0

    for i, combo in enumerate(combos, 1):
        exec_params = dict(zip(keys, combo))
        r   = run_config(ticker, v7_params, exec_params, data_cache)
        if 'error' in r and r['error'] == 'invalid_macd_pair':
            continue
        run_count += 1
        sc = score(r)
        if sc > best_score:
            best_score  = sc
            best_result = r
            best_exec   = exec_params.copy()

        if i % 300 == 0 or i == len(combos):
            prog = f"{i}/{len(combos)}"
            if best_result and 'error' not in best_result:
                print(f"  [{prog}] Best: Sharpe {best_result['sharpe']:.2f}  "
                      f"CAGR {best_result['cagr']:.1f}%  "
                      f"DD {best_result['max_dd']:.1f}%  "
                      f"Calmar {best_result['calmar_ratio']:.2f}  "
                      f"config={best_exec}", flush=True)

    print(f"  Valid runs: {run_count}/{len(combos)}")
    return best_result, best_exec, best_score


# ---------------------------------------------------------------------------
def print_result(ticker, label, result, v7_params, exec_params):
    if not result or 'error' in result:
        print(f"  {ticker}: FAILED")
        return
    print(f"\n{'=' * 64}")
    print(f"  BEST EXEC CONFIG: {ticker} ({label})")
    print(f"{'=' * 64}")
    print(f"  -- Numeric (V6) --")
    print(f"  trail_atr={v7_params.get('trail_atr')}  vol_target={v7_params.get('vol_target')}  "
          f"tp_mult={v7_params.get('tp_mult')}  partial_tp_mult={v7_params.get('partial_tp_mult')}")
    print(f"  -- Signal (V7) --")
    print(f"  rsi={v7_params.get('rsi_period')}  rsi_os={v7_params.get('rsi_oversold')}  "
          f"ema_trend={v7_params.get('ema_trend')}  adx={v7_params.get('adx_thresh')}  "
          f"min_str={v7_params.get('min_strength_up')}")
    print(f"  -- Execution (V8) --")
    print(f"  trail_cushion    = {exec_params['trail_cushion']}")
    print(f"  post_partial_mult= {exec_params['post_partial_mult']}")
    print(f"  macd             = {exec_params['macd_fast']}/{exec_params['macd_slow']}/9")
    print(f"  max_hold_trend   = {exec_params['max_hold_trend']}d")
    print(f"  max_hold_mr      = {exec_params['max_hold_mr']}d")
    print(f"  ---")
    print(f"  Sharpe:   {result['sharpe']:.2f}  |  Sortino: {result['sortino']:.2f}")
    print(f"  CAGR:     {result['cagr']:.2f}%  |  Return:   {result['return_pct']:.1f}%")
    print(f"  Max DD:   {result['max_dd']:.2f}%  |  Calmar:   {result['calmar_ratio']:.2f}")
    print(f"  Win Rate: {result['win_rate']:.1f}%  |  PF:       {result['profit_factor']:.2f}")
    tpy = result['trades'] / max(result['years'], 0.5)
    print(f"  Trades:   {result['trades']} ({tpy:.0f}/yr)  |  Hold: {result['avg_hold_days']:.1f}d")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    v7_all = load_v7_params()

    total_combos = len(list(itertools.product(*EXEC_GRID.values())))
    total_runs   = total_combos * len(ASSETS)
    print(f"\n{'=' * 64}")
    print(f"  V8 EXECUTION OPTIMIZER  |  QQQ + GBTC + XLK + NVDA  |  10yr")
    print(f"  Tuning: trail_cushion, post_partial_mult, MACD periods,")
    print(f"          max_hold_trend, max_hold_mr")
    print(f"  Grid per asset: {total_combos}  (invalid MACD pairs auto-skipped)")
    print(f"  Total runs (max): {total_runs}")
    print(f"{'=' * 64}")

    data_cache    = {}
    final_results = {}

    for ticker, label in ASSETS.items():
        v7p = v7_all.get(ticker, DEFAULT_V7_PARAMS[ticker])
        result, exec_params, sc = optimize_asset(ticker, label, v7p, data_cache)
        final_results[ticker] = {
            'v7_params':   v7p,
            'exec_params': exec_params,
            'result':      result,
            'score':       sc,
        }
        print_result(ticker, label, result, v7p, exec_params)

    # -----------------------------------------------------------------------
    # Summary table
    print(f"\n\n{'=' * 80}")
    print(f"  V8 FINAL RESULTS  (V6 numeric + V7 signal + V8 execution, 10yr long-only)")
    print(f"{'=' * 80}")
    hdr = (f"  {'Ticker':<6}  {'CAGR%':>6}  {'Sharpe':>7}  {'Sortino':>7}  "
           f"{'MaxDD%':>6}  {'WR%':>5}  {'Trd/yr':>7}  {'Calmar':>7}")
    print(hdr)
    print(f"  {'-' * 74}")
    for ticker in ASSETS:
        r = final_results[ticker]['result']
        if r and 'error' not in r:
            tpy = r['trades'] / max(r['years'], 0.5)
            print(f"  {ticker:<6}  {r['cagr']:>6.1f}  {r['sharpe']:>7.2f}  "
                  f"{r['sortino']:>7.2f}  {r['max_dd']:>6.1f}  {r['win_rate']:>5.1f}  "
                  f"{tpy:>7.0f}  {r['calmar_ratio']:>7.2f}")
    print(f"  {'-' * 74}")

    # -----------------------------------------------------------------------
    # Save — full_config combines all three layers
    save = {}
    for ticker in ASSETS:
        entry = final_results[ticker]
        if entry['result'] and 'error' not in entry['result']:
            v7p  = entry['v7_params']
            ep   = entry['exec_params']
            save[ticker] = {
                'full_config': {
                    # V6
                    'trail_atr':       v7p.get('trail_atr'),
                    'vol_target':      v7p.get('vol_target'),
                    'tp_mult':         v7p.get('tp_mult'),
                    'partial_tp_mult': v7p.get('partial_tp_mult'),
                    # V7
                    'rsi_period':      v7p.get('rsi_period'),
                    'rsi_oversold':    v7p.get('rsi_oversold'),
                    'atr_period':      v7p.get('atr_period'),
                    'ema_trend':       v7p.get('ema_trend'),
                    'adx_thresh':      v7p.get('adx_thresh'),
                    'min_strength_up': v7p.get('min_strength_up'),
                    # V8
                    'trail_cushion':     ep['trail_cushion'],
                    'post_partial_mult': ep['post_partial_mult'],
                    'macd_fast':         ep['macd_fast'],
                    'macd_slow':         ep['macd_slow'],
                    'max_hold_trend':    ep['max_hold_trend'],
                    'max_hold_mr':       ep['max_hold_mr'],
                },
                'metrics': {k: v for k, v in entry['result'].items()
                            if k != 'config'},
            }

    out = 'intraday/results/v8_execution_configs.json'
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nFull configs saved -> {out}")
    print("\nINSTRUCTIONS: Pass 'full_config' kwargs to AggressiveHybridV6(**full_config)")
    print("All V6 + V7 + V8 params are pre-merged per asset.")
