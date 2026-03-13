"""
GBTC On-Chain Optimizer
Tunes MVRV + Fear & Greed thresholds on top of the V10 short baseline.
Fetches on-chain data ONCE then loops through threshold combos cheaply.

Baseline to beat: Sharpe 1.44, CAGR 13.89% (V10 shorts, 930 trades)
Constraint: Both Sharpe AND CAGR must improve, more trades preferred.
"""
import sys, warnings, itertools, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'intraday/strategies')
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

# ── V10 best params (frozen) ──────────────────────────────────────────────────
V10_PARAMS = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=100, adx_thresh=32, min_strength_up=0.30,
    trail_cushion=0.5, post_partial_mult=2.5,
    macd_fast=8, macd_slow=38, macd_sig=9,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=60,
)

BASELINE_SHARPE = 1.44
BASELINE_CAGR   = 13.89
BASELINE_TRADES = 930

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID = {
    'mvrv_long_thresh':  [0.8, 1.0, 1.2, 1.5, 1.8, 2.0],   # below = buy boost
    'mvrv_short_thresh': [2.5, 3.0, 3.5, 4.0, 4.5],         # above = short boost/long suppress
    'fg_fear_thresh':    [15, 20, 25, 30, 35],               # below = extreme fear buy
    'fg_greed_thresh':   [65, 70, 75, 80, 85],              # above = extreme greed short
}
# 6 x 5 x 5 x 5 = 750 combos

# ── Fetch data ONCE ───────────────────────────────────────────────────────────
print("Fetching GBTC + on-chain data (one-time)...")
t0 = time.time()
base = AggressiveHybridV6(
    'GBTC', start='2015-01-01', end='2025-12-31',
    use_onchain=True,
    **V10_PARAMS
)
ok = base.fetch_data()
if not ok:
    print("ERROR: data fetch failed"); sys.exit(1)
print(f"Data fetched in {time.time()-t0:.1f}s  ({len(base.data)} GBTC bars)")
print(f"MVRV loaded: {base._mvrv.min():.2f} - {base._mvrv.max():.2f}")
print(f"F&G loaded:  {base._fg.min():.0f} - {base._fg.max():.0f}")

# ── Grid search ───────────────────────────────────────────────────────────────
keys   = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
total  = len(combos)
print(f"\nRunning {total} combos...\n")

results = []
t_start = time.time()

for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))

    # Skip invalid: long_thresh must be < short_thresh
    if params['mvrv_long_thresh'] >= params['mvrv_short_thresh']:
        continue

    # Update thresholds in-place (no re-fetch)
    base.mvrv_long_thresh  = params['mvrv_long_thresh']
    base.mvrv_short_thresh = params['mvrv_short_thresh']
    base.fg_fear_thresh    = params['fg_fear_thresh']
    base.fg_greed_thresh   = params['fg_greed_thresh']

    try:
        r = base.backtest()
    except Exception as e:
        continue

    if r is None:
        continue

    # Both must improve, penalize fewer trades
    if r['sharpe'] > BASELINE_SHARPE and r['cagr'] > BASELINE_CAGR:
        trade_bonus = (r['trades'] - BASELINE_TRADES) * 0.0005
        score = r['sharpe'] + (r['cagr'] / 100) + trade_bonus
        results.append({**params, **r, 'score': score})

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_start
        rate    = (i + 1) / elapsed
        eta     = (total - i - 1) / rate
        print(f"  {i+1}/{total}  valid so far: {len(results)}  ETA: {eta:.0f}s")

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s  |  {len(results)} combos beat baseline\n")

# ── Results ───────────────────────────────────────────────────────────────────
if not results:
    print("No combo beat BOTH Sharpe AND CAGR baseline.")
    print("\nTop 5 by Sharpe (showing partial improvements):")
    # Re-run to find best single-metric improvements
    all_r = []
    for vals in combos:
        params = dict(zip(keys, vals))
        if params['mvrv_long_thresh'] >= params['mvrv_short_thresh']:
            continue
        base.mvrv_long_thresh  = params['mvrv_long_thresh']
        base.mvrv_short_thresh = params['mvrv_short_thresh']
        base.fg_fear_thresh    = params['fg_fear_thresh']
        base.fg_greed_thresh   = params['fg_greed_thresh']
        try:
            r = base.backtest()
            if r:
                all_r.append({**params, **r})
        except Exception:
            pass
    all_r.sort(key=lambda x: x['sharpe'], reverse=True)
    for row in all_r[:5]:
        print(f"  Sharpe={row['sharpe']:.3f} CAGR={row['cagr']:.2f}% "
              f"Trades={row['trades']} | "
              f"mvrv_L={row['mvrv_long_thresh']} mvrv_S={row['mvrv_short_thresh']} "
              f"fg_F={row['fg_fear_thresh']} fg_G={row['fg_greed_thresh']}")
else:
    results.sort(key=lambda x: x['score'], reverse=True)
    print("=" * 90)
    print(f"{'Rank':<5} {'Sharpe':>7} {'CAGR%':>8} {'Trades':>7} {'WR%':>6} {'MaxDD%':>7} "
          f"{'mvrv_L':>7} {'mvrv_S':>7} {'fg_F':>5} {'fg_G':>5}")
    print("=" * 90)
    for rank, row in enumerate(results[:20], 1):
        flag = ''
        if row['sharpe'] == max(r['sharpe'] for r in results): flag += '*S'
        if row['cagr']   == max(r['cagr']   for r in results): flag += '*C'
        if row['trades'] == max(r['trades']  for r in results): flag += '*T'
        print(f"{rank:<5} {row['sharpe']:>7.3f} {row['cagr']:>8.2f} {row['trades']:>7} "
              f"{row['win_rate']:>6.1f} {row['max_dd']:>7.1f} "
              f"{row['mvrv_long_thresh']:>7.1f} {row['mvrv_short_thresh']:>7.1f} "
              f"{row['fg_fear_thresh']:>5} {row['fg_greed_thresh']:>5}  {flag}")
    print("=" * 90)
    print(f"\nBaseline: Sharpe={BASELINE_SHARPE} CAGR={BASELINE_CAGR}% Trades={BASELINE_TRADES}")

    best = results[0]
    print(f"\n>>> BEST PARAMS (score={best['score']:.4f}):")
    print(f"  use_onchain=True")
    print(f"  mvrv_long_thresh={best['mvrv_long_thresh']}")
    print(f"  mvrv_short_thresh={best['mvrv_short_thresh']}")
    print(f"  fg_fear_thresh={best['fg_fear_thresh']}")
    print(f"  fg_greed_thresh={best['fg_greed_thresh']}")
    print(f"  => Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.2f}%  "
          f"Trades={best['trades']} (L={best['long_trades']} S={best['short_trades']})")
    print(f"  => vs baseline: dSharpe={best['sharpe']-BASELINE_SHARPE:+.3f}  "
          f"dCAGR={best['cagr']-BASELINE_CAGR:+.2f}pp  "
          f"dTrades={best['trades']-BASELINE_TRADES:+d}")
