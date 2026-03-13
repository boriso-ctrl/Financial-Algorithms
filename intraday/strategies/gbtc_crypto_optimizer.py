"""
GBTC Crypto Optimizer — targets the two root causes of GBTC underperformance:

  1. vol_target too low (0.22) → position sizes 5.5x smaller than equity assets
     Grid: 0.30, 0.40, 0.50, 0.60
  2. max_hold_trend too short (30 bars) → exits Bitcoin bull runs too early
     Grid: 60, 90, 120, 150, 180

Also re-grids the other key crypto-relevant params:
  trail_atr       → crypto needs wider trail (3.0, 3.5, 4.0, 4.5, 5.0)
  tp_mult         → higher TP for longer holds (3.0, 4.5, 6.0, 8.0)
  adx_thresh      → tune sensitivity (18, 22, 27, 32)
  enable_bb_signal → keep V9 winner

Scoring:
  Hard floor: sharpe > 1.35 AND cagr > 8.47  (current V9 GBTC baseline)
  OR if nothing beats that: report best combo regardless

Usage:
    python intraday/strategies/gbtc_crypto_optimizer.py
"""

import sys, os, json, itertools, time, warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

# ---------------------------------------------------------------------------
# Current V9 GBTC config (baseline to beat)
# ---------------------------------------------------------------------------
V9_GBTC_BASE = dict(
    trail_atr=3.5, vol_target=0.22, tp_mult=4.5, partial_tp_mult=1.0,
    dd_reduce=0.12, dd_halt=0.20,
    rsi_period=9, rsi_oversold=33, rsi_overbought=65,
    atr_period=14, ema_trend=50, adx_thresh=27,
    min_strength_up=0.30, min_strength_bear=0.35,
    trail_cushion=0.5, post_partial_mult=2.5,
    macd_fast=8, macd_slow=38, macd_sig=9,
    max_hold_trend=30, max_hold_mr=25,
    # V9 additions
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
)

# Must beat both of these
BASELINE_SHARPE = 1.35
BASELINE_CAGR   = 8.47
BASELINE_TPY    = 98

# ---------------------------------------------------------------------------
# Crypto-tuned grid  (4 * 5 * 5 * 4 * 2 = 800 combos)
# ---------------------------------------------------------------------------
GRID = {
    'vol_target':    [0.30, 0.40, 0.50, 0.60],
    'max_hold_trend':[60, 90, 120, 150, 180],
    'trail_atr':     [3.0, 3.5, 4.0, 4.5, 5.0],
    'tp_mult':       [3.0, 4.5, 6.0, 8.0],
    'adx_thresh':    [18, 22, 27, 32],
    'enable_bb_signal': [True, False],
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
        cfg = {**V9_GBTC_BASE, **grid_params}
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
    print(f"GBTC Crypto Optimizer: {total} combinations")
    print(f"Baseline: Sharpe {BASELINE_SHARPE:.2f}  CAGR {BASELINE_CAGR:.1f}%  {BASELINE_TPY} trades/yr")
    print(f"Root-cause fix: vol_target 0.22->0.30-0.60  |  max_hold_trend 30->60-180")
    print(f"{'='*60}")

    t0 = time.time()
    valid = []      # beat baseline on both metrics
    all_scored = [] # for fallback reporting

    n_proc = min(cpu_count(), 6)
    done   = 0

    with Pool(n_proc) as pool:
        for result, params, sc in pool.imap_unordered(run_backtest, tasks, chunksize=4):
            done += 1
            if result and sc > -999.0:
                all_scored.append((sc, result, params))
                if beats_baseline(result):
                    valid.append((sc, result, params))

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate    = done / elapsed
                print(f"  {done}/{total}  valid={len(valid)}  "
                      f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s")

    print(f"\n{'='*60}")
    print(f"GBTC Crypto Optimizer DONE  — {time.time()-t0:.0f}s")
    print(f"Valid combos (beat V9 baseline on BOTH metrics): {len(valid)}")
    print(f"{'='*60}")

    # Sort descending by score
    valid.sort(key=lambda x: x[0], reverse=True)
    all_scored.sort(key=lambda x: x[0], reverse=True)

    # Report top-5 valid, or top-5 overall if none valid
    pool_to_show = valid if valid else all_scored
    label = "VALID (beat V9 baseline)" if valid else "BEST OVERALL (no combo beat baseline)"

    print(f"\n--- Top 5 {label} ---")
    for rank, (sc, res, prm) in enumerate(pool_to_show[:5], 1):
        print(f"\n#{rank}  score={sc:.4f}")
        print(f"  Sharpe={res.get('sharpe'):.3f}  CAGR={res.get('cagr'):.2f}%  "
              f"Trades={res.get('trades')}  WinRate={res.get('win_rate'):.1f}%  "
              f"MaxDD={res.get('max_dd'):.1f}%  AvgHold={res.get('avg_hold_days'):.1f}d")
        print(f"  Params: {prm}")

    # Determine recommendation
    if valid:
        best_sc, best_res, best_prm = valid[0]
        print(f"\n>>> OPTION A SUCCEEDED — V9 GBTC upgradeable <<<")
        print(f"    Sharpe: {BASELINE_SHARPE:.2f} -> {best_res['sharpe']:.2f}  "
              f"(+{best_res['sharpe']-BASELINE_SHARPE:.2f})")
        print(f"    CAGR:   {BASELINE_CAGR:.1f}% -> {best_res['cagr']:.2f}%  "
              f"(+{best_res['cagr']-BASELINE_CAGR:.2f}pp)")
        print(f"    Key params: vol_target={best_prm['vol_target']}, "
              f"max_hold_trend={best_prm['max_hold_trend']}, "
              f"trail_atr={best_prm['trail_atr']}, tp_mult={best_prm['tp_mult']}")
        upgrade_verdict = "UPGRADE"
    else:
        best_sc, best_res, best_prm = all_scored[0] if all_scored else (0, {}, {})
        print(f"\n>>> OPTION A FAILED — best combo still below V9 baseline <<<")
        if best_res:
            print(f"    Best achieved: Sharpe={best_res.get('sharpe'):.2f}  "
                  f"CAGR={best_res.get('cagr'):.2f}%  (vs required {BASELINE_SHARPE}/{BASELINE_CAGR})")
        print(f"    → Proceed with Option B: replace GBTC with a better asset")
        upgrade_verdict = "REPLACE"

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gbtc_crypto_optimizer_results.json')

    top5_valid   = [(sc, res, prm) for sc, res, prm in valid[:5]]
    top5_overall = [(sc, res, prm) for sc, res, prm in all_scored[:5]]

    with open(out_path, 'w') as f:
        json.dump({
            'verdict': upgrade_verdict,
            'baseline': {'sharpe': BASELINE_SHARPE, 'cagr': BASELINE_CAGR, 'tpy': BASELINE_TPY},
            'valid_count': len(valid),
            'best_valid': {
                'params': best_prm if valid else None,
                'result': best_res if valid else None,
                'score':  best_sc  if valid else None,
            },
            'top5_valid':   [{'score': s, 'result': r, 'params': p} for s, r, p in top5_valid],
            'top5_overall': [{'score': s, 'result': r, 'params': p} for s, r, p in top5_overall],
        }, f, indent=2)

    print(f"\nResults saved: {out_path}")
    return upgrade_verdict, best_res, best_prm


if __name__ == '__main__':
    main()
