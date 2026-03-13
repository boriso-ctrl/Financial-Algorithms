"""
GBTC Full Re-Grid Optimizer — explores ALL params with crypto-appropriate ranges.

Previous optimizers only partially explored:
  Crypto optimizer (round 1): vol_target, max_hold_trend, trail_atr, tp_mult, adx_thresh
  Still frozen at equity-tuned defaults:
    ema_trend=50     -> Bitcoin runs on 150-200 day cycles (never explored)
    trail_cushion=0.5 -> tighter/looser trail interaction with crypto vol
    rsi_oversold=33  -> crypto oversold levels differ from equity
    vol_target capped at 0.60 -> theoretical ceiling still higher

Grid (4x4x3x4x3x3 = 1728 combos):
  vol_target:      [0.40, 0.50, 0.60, 0.70]   <- push ceiling higher
  max_hold_trend:  [90, 120, 180, 240]          <- full Bitcoin cycle
  ema_trend:       [50, 100, 200]               <- UNEXPLORED: 200-day for BTC
  trail_atr:       [3.0, 3.5, 4.0, 5.0]        <- wider trail for crypto vol
  adx_thresh:      [22, 27, 32]                 <- trend strength filter
  trail_cushion:   [0.5, 1.0, 2.0]             <- UNEXPLORED: cushion interaction

Baseline to beat: Sharpe 1.38, CAGR 13.36% (round-1 crypto optimizer best)

Usage:
    python intraday/strategies/gbtc_full_optimizer.py
"""

import sys, os, json, itertools, time, warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

# ---------------------------------------------------------------------------
# Best config from round-1 crypto optimizer — baseline to beat
# ---------------------------------------------------------------------------
ROUND1_BEST = dict(
    trail_atr=3.5, vol_target=0.50, tp_mult=3.0, partial_tp_mult=1.0,
    dd_reduce=0.12, dd_halt=0.20,
    rsi_period=9, rsi_oversold=33, rsi_overbought=65,
    atr_period=14, ema_trend=50, adx_thresh=32,
    min_strength_up=0.30, min_strength_bear=0.35,
    trail_cushion=0.5, post_partial_mult=2.5,
    macd_fast=8, macd_slow=38, macd_sig=9,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
)

BASELINE_SHARPE = 1.38
BASELINE_CAGR   = 13.36
BASELINE_TPY    = 85   # trades/yr from round-1

# ---------------------------------------------------------------------------
# Full grid — only the params we want to re-sweep
# ---------------------------------------------------------------------------
GRID = {
    'vol_target':    [0.40, 0.50, 0.60, 0.70],
    'max_hold_trend':[90, 120, 180, 240],
    'ema_trend':     [50, 100, 200],
    'trail_atr':     [3.0, 3.5, 4.0, 5.0],
    'adx_thresh':    [22, 27, 32],
    'trail_cushion': [0.5, 1.0, 2.0],
}

START = '2015-01-01'
END   = '2025-12-31'

# ---------------------------------------------------------------------------
# Scoring — reward sharpe + cagr, bonus for maintaining trade frequency
# ---------------------------------------------------------------------------
def score(result):
    if not result or 'error' in result:
        return -999.0
    sharpe = result.get('sharpe', 0)
    cagr   = result.get('cagr',   0)
    trades = result.get('trades', 0)
    years  = result.get('years',  1)
    tpy    = trades / max(years, 0.5)
    tpy_ratio = max(0.5, min(1.5, tpy / BASELINE_TPY))
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
        cfg = {**ROUND1_BEST, **grid_params}
        trader = AggressiveHybridV6(ticker='GBTC', start=START, end=END, **cfg)
        if not trader.fetch_data():
            return None, grid_params, -999.0
        result = trader.backtest()
        sc = score(result)
        return result, grid_params, sc
    except Exception:
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
    print(f"GBTC Full Re-Grid: {total} combinations")
    print(f"Baseline (round-1 crypto): Sharpe {BASELINE_SHARPE:.2f}  "
          f"CAGR {BASELINE_CAGR:.1f}%  ~{BASELINE_TPY} trades/yr")
    print(f"New levers: ema_trend [50/100/200], trail_cushion, vol_target up to 0.70")
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

            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / max(elapsed, 1)
                print(f"  {done}/{total}  valid={len(valid)}  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s  |  Valid combos: {len(valid)}/{total}")
    print(f"{'='*60}")

    valid.sort(key=lambda x: x[0], reverse=True)
    all_scored.sort(key=lambda x: x[0], reverse=True)

    pool_to_show = valid if valid else all_scored
    label = "VALID (beat round-1)" if valid else "BEST OVERALL (no combo beat round-1)"

    print(f"\n--- Top 5 {label} ---")
    for rank, (sc, res, prm) in enumerate(pool_to_show[:5], 1):
        print(f"\n#{rank}  score={sc:.4f}")
        print(f"  Sharpe={res.get('sharpe'):.3f}  CAGR={res.get('cagr'):.2f}%  "
              f"Trades={res.get('trades')}  WinRate={res.get('win_rate'):.1f}%  "
              f"MaxDD={res.get('max_dd'):.1f}%  AvgHold={res.get('avg_hold_days'):.1f}d")
        print(f"  Params: {prm}")

    if valid:
        best_sc, best_res, best_prm = valid[0]
        print(f"\n>>> FURTHER IMPROVEMENT FOUND <<<")
        print(f"    Sharpe: {BASELINE_SHARPE:.2f} -> {best_res['sharpe']:.3f}  "
              f"({best_res['sharpe']-BASELINE_SHARPE:+.3f})")
        print(f"    CAGR:   {BASELINE_CAGR:.2f}% -> {best_res['cagr']:.2f}%  "
              f"({best_res['cagr']-BASELINE_CAGR:+.2f}pp)")
        print(f"    Key change:")
        for k in ['vol_target','max_hold_trend','ema_trend','trail_atr','adx_thresh','trail_cushion']:
            old = ROUND1_BEST.get(k)
            new = best_prm.get(k)
            marker = " <-- CHANGED" if old != new else ""
            print(f"      {k}: {old} -> {new}{marker}")
    else:
        best_sc, best_res, best_prm = all_scored[0] if all_scored else (0, {}, {})
        print(f"\n>>> ROUND-1 IS ALREADY OPTIMAL for this param space <<<")
        if best_res:
            print(f"    Best achieved: Sharpe={best_res.get('sharpe'):.3f}  "
                  f"CAGR={best_res.get('cagr'):.2f}%")

    # Save full results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'gbtc_full_optimizer_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump({
            'verdict': 'IMPROVED' if valid else 'PLATEAU',
            'round1_baseline': {'sharpe': BASELINE_SHARPE, 'cagr': BASELINE_CAGR},
            'valid_count': len(valid),
            'best': {
                'params':  best_prm,
                'result':  best_res,
                'score':   best_sc,
            },
            'top5_valid':   [{'score': s, 'result': r, 'params': p}
                             for s, r, p in valid[:5]],
            'top5_overall': [{'score': s, 'result': r, 'params': p}
                             for s, r, p in all_scored[:5]],
        }, f, indent=2)

    print(f"\nResults: {out_path}")
    return valid[0] if valid else None


if __name__ == '__main__':
    main()
