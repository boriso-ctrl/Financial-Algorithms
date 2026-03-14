"""
BTC-USD Optimizer
Params tuned on GBTC (weekday-only) are miscalibrated for 24/7 BTC spot data.
Scale factor: 365/252 ~= 1.45x (all time-based params need to be larger).

Baseline: same strategy on GBTC with V10 params -> Sharpe 1.47, CAGR 14.19%
BTC-USD raw (GBTC params): Sharpe 0.87, CAGR 7.3%

Grid focuses on calendar-time alignment + crypto vol scaling.
"""
import sys, warnings, itertools, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'intraday/strategies')
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

TICKER = 'BTC-USD'

# Baseline: naive port of GBTC params
BASELINE_SHARPE = 0.87
BASELINE_CAGR   = 7.30
BASELINE_TRADES = 815

# Fixed params (less time-sensitive)
FIXED = dict(
    tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33,
    min_strength_up=0.30, trail_cushion=0.5, post_partial_mult=2.5,
    macd_sig=9,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
)

# Grid — scale time params up for 24/7 data
GRID = {
    'ema_trend':      [100, 130, 150, 180],  # GBTC=100 -> 24/7 equiv ~145
    'max_hold_trend': [90, 120, 150, 180],   # GBTC=90  -> 24/7 equiv ~130
    'max_hold_short': [60, 80, 100],         # GBTC=60  -> 24/7 equiv ~87
    'max_hold_mr':    [25, 35, 45],          # GBTC=25  -> 24/7 equiv ~36
    'vol_target':     [0.40, 0.50, 0.60, 0.70],  # crypto vol may need retuning
    'trail_atr':      [3.0, 3.5, 4.0, 4.5],
    'adx_thresh':     [25, 28, 32, 35],
    'atr_period':     [14, 21],
    'macd_fast':      [8, 12],
    'macd_slow':      [26, 38],
}
# 4x4x3x3x4x4x4x2x2x2 = 36,864 -- too many, reduce

# Round 1: focus on the highest-impact params first
GRID_R1 = {
    'ema_trend':      [100, 130, 150, 180],
    'max_hold_trend': [90, 120, 150],
    'vol_target':     [0.40, 0.50, 0.60, 0.70],
    'trail_atr':      [3.0, 3.5, 4.0, 4.5],
    'adx_thresh':     [25, 28, 32, 35],
}
# 4x3x4x4x4 = 768 combos

# Fetch data once
print(f"Fetching {TICKER} data (one-time)...")
t0 = time.time()
base = AggressiveHybridV6(TICKER, start='2015-01-01', end='2025-12-31',
    ema_trend=100, max_hold_trend=90, vol_target=0.60,
    trail_atr=4.0, adx_thresh=32,
    max_hold_short=60, max_hold_mr=25,
    macd_fast=8, macd_slow=38, atr_period=14,
    **FIXED)
ok = base.fetch_data()
if not ok:
    print("FETCH FAILED"); sys.exit(1)
print(f"Fetched in {time.time()-t0:.1f}s  ({len(base.data)} bars)")

keys   = list(GRID_R1.keys())
combos = list(itertools.product(*[GRID_R1[k] for k in keys]))
total  = len(combos)
print(f"Running {total} combos...\n")

results = []
t_start = time.time()

for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))
    for k, v in params.items():
        setattr(base, k, v)

    try:
        r = base.backtest()
    except Exception:
        continue
    if r is None or 'error' in r:
        continue

    if r['sharpe'] > BASELINE_SHARPE and r['cagr'] > BASELINE_CAGR:
        trade_bonus = (r['trades'] - BASELINE_TRADES) * 0.0005
        score = r['sharpe'] + (r['cagr'] / 100) + trade_bonus
        results.append({**params, **r, 'score': score})

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta  = (total - i - 1) / max(rate, 0.001)
        print(f"  {i+1}/{total}  valid: {len(results)}  ETA: {eta:.0f}s")

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s  |  {len(results)} combos beat baseline\n")

if not results:
    print("No combo beat baseline. Best by Sharpe:")
    all_r = []
    for vals in combos:
        params = dict(zip(keys, vals))
        for k, v in params.items():
            setattr(base, k, v)
        try:
            r = base.backtest()
            if r and 'error' not in r:
                all_r.append({**params, **r})
        except Exception:
            pass
    all_r.sort(key=lambda x: x['sharpe'], reverse=True)
    for row in all_r[:10]:
        print(f"  Sharpe={row['sharpe']:.3f} CAGR={row['cagr']:.2f}% "
              f"Trades={row['trades']} | "
              f"ema={row['ema_trend']} mht={row['max_hold_trend']} "
              f"vol={row['vol_target']} atr={row['trail_atr']} adx={row['adx_thresh']}")
else:
    results.sort(key=lambda x: x['score'], reverse=True)
    print("=" * 95)
    print(f"{'Rank':<5} {'Sharpe':>7} {'CAGR%':>8} {'Trades':>7} {'WR%':>6} {'MaxDD%':>7} "
          f"{'ema':>5} {'mht':>5} {'vol':>5} {'atr':>5} {'adx':>5}")
    print("=" * 95)
    for rank, row in enumerate(results[:20], 1):
        print(f"{rank:<5} {row['sharpe']:>7.3f} {row['cagr']:>8.2f} {row['trades']:>7} "
              f"{row['win_rate']:>6.1f} {row['max_dd']:>7.1f} "
              f"{row['ema_trend']:>5} {row['max_hold_trend']:>5} "
              f"{row['vol_target']:>5.2f} {row['trail_atr']:>5.1f} {row['adx_thresh']:>5}")
    print("=" * 95)
    print(f"\nBaseline (GBTC params on BTC-USD): Sharpe={BASELINE_SHARPE} CAGR={BASELINE_CAGR}% Trades={BASELINE_TRADES}")

    best = results[0]
    print(f"\n>>> BEST BTC-USD PARAMS:")
    print(f"  ema_trend={best['ema_trend']}  max_hold_trend={best['max_hold_trend']}")
    print(f"  vol_target={best['vol_target']}  trail_atr={best['trail_atr']}  adx_thresh={best['adx_thresh']}")
    print(f"  => Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.2f}%  "
          f"Trades={best['trades']} (L={best['long_trades']} S={best['short_trades']})")
    print(f"  => vs baseline: dSharpe={best['sharpe']-BASELINE_SHARPE:+.3f}  "
          f"dCAGR={best['cagr']-BASELINE_CAGR:+.2f}pp  "
          f"dTrades={best['trades']-BASELINE_TRADES:+d}")
