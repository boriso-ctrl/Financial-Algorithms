"""
Weekend-only BTC-USD optimizer.
Entry days: Friday (4) and/or Saturday (5) — positions open on Sat/Sun.
max_hold kept short (1-3 days) to ensure positions close before Monday GBTC overlap.
"""
import sys, warnings, itertools
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from intraday.strategies.aggressive_hybrid_v6_10yr import AggressiveHybridV6

# Pre-fetch data once; reuse across all combos
print("Fetching BTC-USD data once...")
_base = AggressiveHybridV6('BTC-USD', start='2015-01-01', end='2025-01-01',
    trail_atr=4.0, vol_target=0.70, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=60, adx_thresh=20, min_strength_up=0.25,
    trail_cushion=0.5, post_partial_mult=2.0,
    macd_fast=8, macd_slow=38,
    max_hold_trend=2, max_hold_mr=2,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=2,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
    entry_days={4, 5},  # Fri+Sat
)
_base.fetch_data()
print(f"Loaded {len(_base.data)} BTC-USD bars\n")

# Grid
GRID = dict(
    trail_atr    = [2.5, 3.5, 4.5],
    tp_mult      = [1.5, 2.5, 3.5],
    vol_target   = [0.60, 0.80],
    ema_trend    = [20, 40, 60, 100],
    max_hold     = [1, 2, 3],
    entry_days   = [{4, 5}, {5, 6}],   # Fri+Sat or Sat+Sun
)

keys   = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
print(f"Testing {len(combos)} combos...")

results = []
for vals in combos:
    p = dict(zip(keys, vals))
    try:
        s = AggressiveHybridV6('BTC-USD', start='2015-01-01', end='2025-01-01',
            trail_atr=p['trail_atr'], vol_target=p['vol_target'],
            tp_mult=p['tp_mult'], partial_tp_mult=1.0,
            rsi_period=9, rsi_oversold=33, atr_period=14,
            ema_trend=p['ema_trend'], adx_thresh=20, min_strength_up=0.25,
            trail_cushion=0.5, post_partial_mult=2.0,
            macd_fast=8, macd_slow=38,
            max_hold_trend=p['max_hold'], max_hold_mr=p['max_hold'],
            enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
            allow_shorts=True, max_hold_short=p['max_hold'],
            use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
            fg_fear_thresh=25, fg_greed_thresh=75,
            entry_days=p['entry_days'],
        )
        s.data     = _base.data.copy()
        s.vix      = _base.vix
        s._mvrv    = _base._mvrv
        s._fg      = _base._fg
        s.prepare_indicators()
        m = s.backtest()
        if 'error' in m or m['trades'] < 30:
            continue
        results.append({**p, **m})
    except Exception as e:
        continue

results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\nValid combos: {len(results)}")
print(f"\nTop 15 weekend BTC-USD results:")
print(f"{'entry_days':<12} {'ema':>4} {'trail':>6} {'tp':>5} {'vol':>5} {'hold':>5} "
      f"{'Sharpe':>7} {'CAGR%':>7} {'Trades':>7} {'WR%':>6} {'MaxDD%':>7}")
print('-'*90)
for r in results[:15]:
    ed = 'Fri+Sat' if 4 in r['entry_days'] and 5 in r['entry_days'] and 6 not in r['entry_days'] else \
         'Sat+Sun' if 5 in r['entry_days'] and 6 in r['entry_days'] and 4 not in r['entry_days'] else str(r['entry_days'])
    print(f"{ed:<12} {r['ema_trend']:>4} {r['trail_atr']:>6.1f} {r['tp_mult']:>5.1f} "
          f"{r['vol_target']:>5.2f} {r['max_hold']:>5} "
          f"{r['sharpe']:>7.3f} {r['cagr']:>7.2f} "
          f"{r['trades']:>7d} {r['win_rate']:>6.1f} {r['max_dd']:>7.1f}")

if results:
    best = results[0]
    print(f"\nBEST: entry_days={'Fri+Sat' if 4 in best['entry_days'] else 'Sat+Sun'} "
          f"ema={best['ema_trend']} trail={best['trail_atr']} tp={best['tp_mult']} "
          f"vol={best['vol_target']} hold={best['max_hold']}")
    print(f"      Sharpe={best['sharpe']}  CAGR={best['cagr']}%  "
          f"Trades={best['trades']}  WR={best['win_rate']}%  MaxDD={best['max_dd']}%")
