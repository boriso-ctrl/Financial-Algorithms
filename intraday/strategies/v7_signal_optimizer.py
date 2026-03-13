"""
V7 Per-Asset Signal Optimizer
==============================
Optimises indicator and signal parameters for each asset individually.

WHY THIS MATTERS — different assets need different signal tuning:
  QQQ  : smooth, ETF-level mean-reversion; RSI works well; 14-20 period ATR
  GBTC : extreme crypto volatility; needs faster EMA + higher ADX gate
  XLK  : tech trend-follower; EMA alignment signals dominant
  NVDA : high-beta single stock; needs tighter ADX, faster RSI response

Parameters tuned per asset (on top of V6 best numeric params):
  rsi_period      : RSI lookback (9, 14, 21)
  rsi_oversold    : oversold trigger level (28, 33, 38)
  atr_period      : ATR / ADX computation period (10, 14, 20)
  ema_trend       : medium-term trend EMA span (30, 50, 80)
  adx_thresh      : ADX gate for "strong trend" (18, 22, 27)
  min_strength_up : signal quality gate in uptrend (0.20, 0.25, 0.30)

Grid: 3 x 3 x 3 x 3 x 3 x 3 = 729 combos per asset x 4 assets = 2916 runs
      (estimated ~25-35 min total)

Outputs: intraday/results/v7_signal_configs.json
         Contains both V6 numeric params + V7 signal params per asset.
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
# V6 best numeric params (from v6_optimized_configs.json)
# These are FIXED — we only grid-search the signal params on top.
# ---------------------------------------------------------------------------
V6_CONFIGS_PATH = 'intraday/results/v6_optimized_configs.json'
DEFAULT_V6_PARAMS = {
    'QQQ':  {'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 3.0, 'partial_tp_mult': 1.5},
    'GBTC': {'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 4.5, 'partial_tp_mult': 1.0},
    'XLK':  {'trail_atr': 3.0, 'vol_target': 0.22, 'tp_mult': 4.5, 'partial_tp_mult': 1.5},
    'NVDA': {'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 3.0, 'partial_tp_mult': 1.5},
}


def load_v6_params():
    """Load best V6 params from JSON, falling back to hardcoded defaults."""
    try:
        with open(V6_CONFIGS_PATH) as f:
            data = json.load(f)
        params = {}
        for ticker, entry in data.items():
            cfg = entry.get('config', {})
            if cfg and 'trail_atr' in cfg:
                params[ticker] = cfg
            else:
                params[ticker] = DEFAULT_V6_PARAMS.get(ticker, {})
        # Fill any missing tickers
        for t, v in DEFAULT_V6_PARAMS.items():
            if t not in params:
                params[t] = v
        return params
    except Exception:
        return DEFAULT_V6_PARAMS.copy()


# ---------------------------------------------------------------------------
# Signal parameter grid — asset-characteristic-aware ranges
# ---------------------------------------------------------------------------
SIGNAL_GRID = {
    # RSI lookback: shorter = more sensitive (good for volatile assets like GBTC/NVDA)
    'rsi_period':      [9, 14, 21],

    # Oversold trigger: lower = only deepest dips (better for trend-heavy assets)
    'rsi_oversold':    [28, 33, 38],

    # ATR/ADX period: shorter = more reactive, longer = smoother
    'atr_period':      [10, 14, 20],

    # Medium-term trend EMA span: 30=faster (crypto), 50=standard, 80=slower (ETFs)
    'ema_trend':       [30, 50, 80],

    # ADX gate: higher = only trade strong trends (GBTC/NVDA need higher)
    'adx_thresh':      [18, 22, 27],

    # Signal quality gate in uptrend: lower = more trades, higher = higher quality
    'min_strength_up': [0.20, 0.25, 0.30],
}

# Total: 3^6 = 729 per asset


def score(r):
    """
    Composite score: Sharpe + CAGR/25 × trade-frequency penalty.
    - Penalises < 50 trades/yr (over-fitted to rare events).
    - Rewards high Sharpe AND decent CAGR simultaneously.
    - Calmar bonus: reward low-DD strategies (+0.1 per Calmar point).
    """
    if 'error' in r:
        return -99
    sharpe  = r.get('sharpe', 0)
    cagr    = r.get('cagr', 0)
    calmar  = r.get('calmar_ratio', 0)
    trades  = r.get('trades', 0)
    years   = max(r.get('years', 1), 0.5)
    tpy     = trades / years
    trade_pen = min(tpy / 50, 1.0)
    return (sharpe + cagr / 25 + calmar * 0.10) * trade_pen


def run_config(ticker, v6_params, signal_params, data_cache):
    """Run one backtest reusing pre-fetched raw data."""
    try:
        trader = AggressiveHybridV6(
            ticker=ticker, start=START, end=END,
            # V6 numeric params (fixed per asset)
            trail_atr=v6_params['trail_atr'],
            vol_target=v6_params['vol_target'],
            tp_mult=v6_params['tp_mult'],
            partial_tp_mult=v6_params['partial_tp_mult'],
            # V7 signal/indicator params (grid-searched)
            rsi_period=signal_params['rsi_period'],
            rsi_oversold=signal_params['rsi_oversold'],
            atr_period=signal_params['atr_period'],
            ema_trend=signal_params['ema_trend'],
            adx_thresh=signal_params['adx_thresh'],
            min_strength_up=signal_params['min_strength_up'],
        )
        # Reuse cached raw data to skip re-downloading
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

        results = trader.backtest()
        if results is None:
            return {'error': 'no result'}
        return results
    except Exception as e:
        return {'error': str(e)}


# ---------------------------------------------------------------------------
def optimize_asset(ticker, label, v6_params, data_cache):
    keys   = list(SIGNAL_GRID.keys())
    values = list(SIGNAL_GRID.values())
    combos = list(itertools.product(*values))

    print(f"\n{'=' * 62}")
    print(f"  Optimizing {ticker} ({label})")
    print(f"  V6 base: trail={v6_params['trail_atr']} vol={v6_params['vol_target']} "
          f"tp={v6_params['tp_mult']} ptp={v6_params['partial_tp_mult']}")
    print(f"  Signal grid: {len(combos)} combos")
    print(f"{'=' * 62}")

    best_score  = -999
    best_result = None
    best_sig    = None

    for i, combo in enumerate(combos, 1):
        sig_params = dict(zip(keys, combo))
        r  = run_config(ticker, v6_params, sig_params, data_cache)
        sc = score(r)
        if sc > best_score:
            best_score  = sc
            best_result = r
            best_sig    = sig_params.copy()

        if i % 100 == 0 or i == len(combos):
            prog = f"{i}/{len(combos)}"
            if best_result and 'error' not in best_result:
                print(f"  [{prog}] Best: Sharpe {best_result['sharpe']:.2f}  "
                      f"CAGR {best_result['cagr']:.1f}%  "
                      f"DD {best_result['max_dd']:.1f}%  "
                      f"config={best_sig}", flush=True)

    return best_result, best_sig, best_score


# ---------------------------------------------------------------------------
def print_result(ticker, label, result, v6_params, sig_params):
    if not result or 'error' in result:
        print(f"  {ticker}: FAILED")
        return
    print(f"\n{'=' * 62}")
    print(f"  BEST SIGNAL CONFIG: {ticker} ({label})")
    print(f"{'=' * 62}")
    print(f"  -- Numeric (V6, fixed) --")
    print(f"  trail_atr       = {v6_params['trail_atr']}")
    print(f"  vol_target      = {v6_params['vol_target']}")
    print(f"  tp_mult         = {v6_params['tp_mult']}")
    print(f"  partial_tp_mult = {v6_params['partial_tp_mult']}")
    print(f"  -- Signal/indicator (V7, tuned) --")
    print(f"  rsi_period      = {sig_params['rsi_period']}")
    print(f"  rsi_oversold    = {sig_params['rsi_oversold']}")
    print(f"  atr_period      = {sig_params['atr_period']}")
    print(f"  ema_trend       = {sig_params['ema_trend']}")
    print(f"  adx_thresh      = {sig_params['adx_thresh']}")
    print(f"  min_strength_up = {sig_params['min_strength_up']}")
    print(f"  ---")
    print(f"  Sharpe:   {result['sharpe']:.2f}  |  Sortino: {result['sortino']:.2f}")
    print(f"  CAGR:     {result['cagr']:.2f}%  |  Return:   {result['return_pct']:.1f}%")
    print(f"  Max DD:   {result['max_dd']:.2f}%  |  Calmar:   {result['calmar_ratio']:.2f}")
    print(f"  Win Rate: {result['win_rate']:.1f}%  |  PF:       {result['profit_factor']:.2f}")
    tpy = result['trades'] / max(result['years'], 0.5)
    print(f"  Trades:   {result['trades']} ({tpy:.0f}/yr)  |  Hold: {result['avg_hold_days']:.1f}d")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    v6_params_all = load_v6_params()

    print(f"\n{'=' * 62}")
    print(f"  V7 SIGNAL OPTIMIZER  |  QQQ + GBTC + XLK + NVDA  |  10yr")
    print(f"  Tuning indicator params per asset on top of V6 best configs")
    total = len(list(itertools.product(*SIGNAL_GRID.values()))) * len(ASSETS)
    print(f"  Total runs: {total}  ({total//4} per asset)")
    print(f"{'=' * 62}")

    data_cache    = {}
    final_results = {}

    for ticker, label in ASSETS.items():
        v6p = v6_params_all.get(ticker, DEFAULT_V6_PARAMS[ticker])
        result, sig_params, sc = optimize_asset(ticker, label, v6p, data_cache)
        final_results[ticker] = {
            'v6_params':  v6p,
            'sig_params': sig_params,
            'result':     result,
            'score':      sc,
        }
        print_result(ticker, label, result, v6p, sig_params)

    # -----------------------------------------------------------------------
    # Summary table
    print(f"\n\n{'=' * 76}")
    print(f"  V7 OPTIMIZED RESULTS  (V6 numeric + V7 signal params, 10yr long-only)")
    print(f"{'=' * 76}")
    hdr = f"  {'Ticker':<6}  {'CAGR%':>6}  {'Sharpe':>7}  {'Sortino':>7}  " \
          f"{'MaxDD%':>6}  {'WR%':>5}  {'Trd/yr':>7}  {'Calmar':>7}"
    print(hdr)
    print(f"  {'-' * 70}")
    for ticker in ASSETS:
        r = final_results[ticker]['result']
        if r and 'error' not in r:
            tpy = r['trades'] / max(r['years'], 0.5)
            print(f"  {ticker:<6}  {r['cagr']:>6.1f}  {r['sharpe']:>7.2f}  "
                  f"{r['sortino']:>7.2f}  {r['max_dd']:>6.1f}  {r['win_rate']:>5.1f}  "
                  f"{tpy:>7.0f}  {r['calmar_ratio']:>7.2f}")
    print(f"  {'-' * 70}")

    # -----------------------------------------------------------------------
    # Save
    save = {}
    for ticker in ASSETS:
        entry = final_results[ticker]
        if entry['result'] and 'error' not in entry['result']:
            save[ticker] = {
                'v6_numeric_params': entry['v6_params'],
                'v7_signal_params':  entry['sig_params'],
                'full_config': {**entry['v6_params'], **entry['sig_params']},
                'metrics': {k: v for k, v in entry['result'].items()
                            if k != 'config'},
            }

    out = 'intraday/results/v7_signal_configs.json'
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nFull configs saved -> {out}")
    print("\nINSTRUCTIONS: Use 'full_config' for each asset in your runner.")
    print("Both V6 numeric params AND V7 signal params are pre-combined.")
