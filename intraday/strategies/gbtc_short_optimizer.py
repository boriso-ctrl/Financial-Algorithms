"""
GBTC Short-Selling Optimizer — tests whether adding shorts materially improves
GBTC performance by capturing the 2018 (-84%) and 2022 (-76%) crash years.

Starts from round-2 best long-only config (Sharpe 1.43, CAGR 13.55%) and
grids over short-specific parameters only.

Grid (5 x 5 x 3 x 3 = 225 combos):
  max_hold_short: [60, 90, 120, 180, 240]   crypto crashes last months
  tp_mult_short:  [3.0, 4.5, 6.0, 8.0, 10.0]  large crash = large TP
  trail_atr_short:[3.0, 4.0, 5.0]           wider trail for crash vol
  adx_thresh_short:[22, 27, 32]             when downtrend is confirmed enough

Note: tp_mult and trail_atr for longs are kept at round-2 best (separate vars
for shorts achieved by accepting shared params when both = same, but we grid
short-specific overrides via a thin wrapper approach — we temporarily replace
the short-side logic to use short_tp_mult passed in separately).

Simpler approach: use tp_mult (shared) and max_hold_short (new param).
Grid: max_hold_short x tp_mult x trail_atr  [5x5x3 = 75 combos vs all 225]

Baseline: Sharpe 1.43  CAGR 13.55%

Usage:
    python intraday/strategies/gbtc_short_optimizer.py
"""

import sys, os, json, itertools, time, warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

# ---------------------------------------------------------------------------
# Round-2 best GBTC long-only config
# ---------------------------------------------------------------------------
ROUND2_BEST = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    dd_reduce=0.12, dd_halt=0.20,
    rsi_period=9, rsi_oversold=33, rsi_overbought=65,
    atr_period=14, ema_trend=100, adx_thresh=32,
    min_strength_up=0.30, min_strength_bear=0.35,
    trail_cushion=0.5, post_partial_mult=2.5,
    macd_fast=8, macd_slow=38, macd_sig=9,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True,   # KEY: shorts enabled
)

BASELINE_SHARPE = 1.43
BASELINE_CAGR   = 13.55
BASELINE_TPY    = 80

# ---------------------------------------------------------------------------
# Grid — short-specific params on top of round-2 best
# ---------------------------------------------------------------------------
GRID = {
    'max_hold_short': [60, 90, 120, 180, 240],
    'tp_mult':        [3.0, 4.5, 6.0, 8.0, 10.0],   # shared long+short, larger = hold crash longer
    'trail_atr':      [3.0, 4.0, 5.0, 6.0],           # wider trail = let profits run on shorts
    'adx_thresh':     [22, 27, 32],                    # downtrend confirmation threshold
}

START = '2015-01-01'
END   = '2025-12-31'

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(result):
    if not result or 'error' in result:
        return -999.0
    sharpe = result.get('sharpe', 0)
    cagr   = result.get('cagr',   0)
    trades = result.get('trades', 0)
    years  = result.get('years',  1)
    tpy    = trades / max(years, 0.5)
    tpy_ratio = max(0.5, min(2.0, tpy / BASELINE_TPY))  # wider range — shorts add trades
    return (sharpe * 0.55 + (cagr / 30.0) * 0.45) * tpy_ratio


def beats_baseline(result):
    if not result or 'error' in result:
        return False
    return result.get('sharpe', 0) > BASELINE_SHARPE and result.get('cagr', 0) > BASELINE_CAGR


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_backtest(args):
    grid_params = args
    try:
        cfg = {**ROUND2_BEST, **grid_params}
        trader = AggressiveHybridV6(ticker='GBTC', start=START, end=END, **cfg)
        if not trader.fetch_data():
            return None, grid_params, -999.0
        result = trader.backtest()
        sc = score(result)
        return result, grid_params, sc
    except Exception as e:
        return None, grid_params, -999.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)
    tasks  = [dict(zip(keys, c)) for c in combos]

    print(f"\n{'='*60}")
    print(f"GBTC Short Optimizer: {total} combos | allow_shorts=True")
    print(f"Baseline (long-only round-2): Sharpe {BASELINE_SHARPE:.2f}  "
          f"CAGR {BASELINE_CAGR:.1f}%")
    print(f"Target crash years: 2018 (-84%), 2022 (-76%)")
    print(f"{'='*60}")

    t0    = time.time()
    valid = []
    all_scored = []
    done  = 0

    n_proc = min(cpu_count(), 6)

    with Pool(n_proc) as pool:
        for result, params, sc in pool.imap_unordered(run_backtest, tasks, chunksize=4):
            done += 1
            if result and sc > -999.0:
                all_scored.append((sc, result, params))
                if beats_baseline(result):
                    valid.append((sc, result, params))

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / max(elapsed, 1)
                print(f"  {done}/{total}  valid={len(valid)}  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s  |  Valid (beat long-only): {len(valid)}/{total}")
    print(f"{'='*60}")

    valid.sort(key=lambda x: x[0], reverse=True)
    all_scored.sort(key=lambda x: x[0], reverse=True)

    pool_to_show = valid if valid else all_scored
    label = "VALID (beat long-only baseline)" if valid else "BEST OVERALL (shorts did not help)"

    print(f"\n--- Top 5 {label} ---")
    for rank, (sc, res, prm) in enumerate(pool_to_show[:5], 1):
        long_t  = res.get('long_trades', '?')
        short_t = res.get('short_trades', '?')
        print(f"\n#{rank}  score={sc:.4f}")
        print(f"  Sharpe={res.get('sharpe'):.3f}  CAGR={res.get('cagr'):.2f}%  "
              f"Trades={res.get('trades')} (L:{long_t} S:{short_t})  "
              f"WR={res.get('win_rate'):.1f}%  MaxDD={res.get('max_dd'):.1f}%  "
              f"AvgHold={res.get('avg_hold_days'):.1f}d")
        print(f"  Params: {prm}")

    if valid:
        best_sc, best_res, best_prm = valid[0]
        print(f"\n>>> SHORTS HELP! Long-only -> Long+Short uplift: <<<")
        print(f"    Sharpe: {BASELINE_SHARPE:.2f} -> {best_res['sharpe']:.3f}  "
              f"({best_res['sharpe']-BASELINE_SHARPE:+.3f})")
        print(f"    CAGR:   {BASELINE_CAGR:.2f}% -> {best_res['cagr']:.2f}%  "
              f"({best_res['cagr']-BASELINE_CAGR:+.2f}pp)")
        short_t = best_res.get('short_trades', 0)
        long_t  = best_res.get('long_trades', 0)
        print(f"    Trades: {long_t} long + {short_t} short")
        verdict = 'IMPROVED'
    else:
        best_sc, best_res, best_prm = all_scored[0] if all_scored else (0, {}, {})
        print(f"\n>>> SHORTS DO NOT HELP on this strategy architecture <<<")
        if best_res:
            print(f"    Best w/ shorts: Sharpe={best_res.get('sharpe'):.3f}  "
                  f"CAGR={best_res.get('cagr'):.2f}%  "
                  f"(vs required >{BASELINE_SHARPE}/{BASELINE_CAGR})")
        verdict = 'NO_IMPROVEMENT'

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'gbtc_short_optimizer_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump({
            'verdict':     verdict,
            'baseline':    {'sharpe': BASELINE_SHARPE, 'cagr': BASELINE_CAGR},
            'valid_count': len(valid),
            'best': {
                'params': best_prm,
                'result': best_res,
                'score':  best_sc,
            },
            'top5_valid':   [{'score': s, 'result': r, 'params': p}
                             for s, r, p in valid[:5]],
            'top5_overall': [{'score': s, 'result': r, 'params': p}
                             for s, r, p in all_scored[:5]],
        }, f, indent=2)

    print(f"\nResults: {out_path}")
    return verdict, best_res, best_prm


if __name__ == '__main__':
    main()
