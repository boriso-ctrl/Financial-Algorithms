# -*- coding: utf-8 -*-
"""
Equity Strategy Finalizer.

Combines best learnings from Phases 1-6 to find the definitively best
config per asset.  Also tests allow_transition_longs.

Priority: Sharpe >= baseline, then Sortino, then CAGR.
Window: 2015-01-01 -> 2025-01-01 (10 years)
"""
import sys, io, warnings, json
# Force UTF-8 output regardless of Windows cp1252 console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from intraday.strategies.aggressive_hybrid_v6_10yr import AggressiveHybridV6

START = '2015-01-01'
END   = '2025-01-01'

# ---------------------------------------------------------------------------
# Base configs (current production)
# ---------------------------------------------------------------------------
BASE_CONFIGS = {
    'QQQ': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=3.0, partial_tp_mult=1.5,
        rsi_period=9,  rsi_oversold=38, atr_period=14,
        ema_trend=30,  adx_thresh=18,  min_strength_up=0.30,
        trail_cushion=1.5, post_partial_mult=1.5,
        macd_fast=8, macd_slow=38,
        max_hold_trend=60, max_hold_mr=25,
        allow_shorts=False, enable_stoch_rsi=False, enable_bb_signal=False,
        partial_qty_pct=0.50, vol_regime_scale=1.1,
    ),
    'XLK': dict(
        trail_atr=3.0, vol_target=0.22, tp_mult=4.5, partial_tp_mult=1.5,
        rsi_period=14, rsi_oversold=38, atr_period=14,
        ema_trend=80,  adx_thresh=22,  min_strength_up=0.20,
        trail_cushion=1.0, post_partial_mult=2.0,
        macd_fast=8, macd_slow=26,
        max_hold_trend=80, max_hold_mr=25,
        allow_shorts=False, enable_stoch_rsi=False, enable_bb_signal=False,
        partial_qty_pct=0.50, vol_regime_scale=1.1,
    ),
    'NVDA': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=3.0, partial_tp_mult=1.5,
        rsi_period=9,  rsi_oversold=38, atr_period=20,
        ema_trend=50,  adx_thresh=27,  min_strength_up=0.25,
        trail_cushion=2.0, post_partial_mult=2.0,
        macd_fast=12, macd_slow=26,
        max_hold_trend=60, max_hold_mr=25,
        allow_shorts=False, enable_stoch_rsi=False, enable_bb_signal=True,
        partial_qty_pct=0.67, vol_regime_scale=1.1,
    ),
}

# ---------------------------------------------------------------------------
# Candidates per asset: best combos from Phase 1-6 + untested combos
# Each entry = (label, override_dict)
# ---------------------------------------------------------------------------
CANDIDATES = {
    'QQQ': [
        # --- BASELINES ---
        ('BASE',
         {}),
        # Phase 6 winner (Sharpe 1.52, Sortino 2.23, CAGR 13.44%)
        ('P6-best: adx=14 str=0.20 sr+bb ema=25',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=14, min_strength_up=0.20, vol_target=0.22,
              trail_atr=3.5, tp_mult=3.5, post_partial_mult=2.0, partial_qty_pct=0.50,
              ema_trend=25)),
        # Try Ph6 winner + wider exits
        ('P6+wider-exit: adx=14 str=0.20 tp=4.0 pq=0.33',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=14, min_strength_up=0.20, vol_target=0.22,
              trail_atr=3.5, tp_mult=4.0, post_partial_mult=2.0, partial_qty_pct=0.33,
              ema_trend=25)),
        # Phase 6 winner + transition longs
        ('P6-best + transition_longs',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=14, min_strength_up=0.20, vol_target=0.22,
              trail_atr=3.5, tp_mult=3.5, post_partial_mult=2.0, partial_qty_pct=0.50,
              ema_trend=25, allow_transition_longs=True)),
        # Base + transition longs
        ('BASE + transition_longs',
         dict(allow_transition_longs=True)),
        # Lower ADX, same signals as Phase 6 best, original exits
        ('adx=14 str=0.20 sr+bb ema=30 tp=3.0',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=14, min_strength_up=0.20, vol_target=0.22,
              trail_atr=3.5, tp_mult=3.0, post_partial_mult=1.5, partial_qty_pct=0.50,
              ema_trend=30)),
        # Just StochRSI added (no BB, lower adx)
        ('adx=14 str=0.20 sr=Y bb=N ema=30',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=False,
              adx_thresh=14, min_strength_up=0.20, vol_target=0.22,
              trail_atr=3.5, tp_mult=3.0, post_partial_mult=1.5, partial_qty_pct=0.50,
              ema_trend=30)),
    ],

    'XLK': [
        # --- BASELINES ---
        ('BASE',
         {}),
        # Phase 3 winner: adx=22, str=0.15 (Sharpe 1.95)
        ('P3-best: adx=22 str=0.15',
         dict(adx_thresh=22, min_strength_up=0.15)),
        # P3-best + Phase 4 exits (Sortino 3.60, CAGR 21.17%)
        ('P3+P4: adx=22 str=0.15 tp=5.0 ppm=2.5 pq=0.33',
         dict(adx_thresh=22, min_strength_up=0.15,
              trail_atr=3.0, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # P3+P4 + StochRSI+BB
        ('P3+P4+sig: adx=22 str=0.15 tp=5.0 sr+bb',
         dict(adx_thresh=22, min_strength_up=0.15,
              trail_atr=3.0, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.33,
              enable_stoch_rsi=True, enable_bb_signal=True)),
        # P3+P4 + vol=0.28
        ('P3+P4+vt=0.28: adx=22 str=0.15 tp=5.0',
         dict(adx_thresh=22, min_strength_up=0.15,
              trail_atr=3.0, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.33,
              vol_target=0.28)),
        # Phase 6 best (no shorts): adx=18 str=0.15 sr+bb ema=60 tp=5.0 pq=0.33 (Sortino 3.39)
        ('P6-noshor: adx=18 str=0.15 sr+bb ema=60 tp=5.0',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=18, min_strength_up=0.15, vol_target=0.22,
              trail_atr=3.0, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.33,
              ema_trend=60)),
        # P3+P4 + transition longs
        ('P3+P4 + transition_longs',
         dict(adx_thresh=22, min_strength_up=0.15,
              trail_atr=3.0, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.33,
              allow_transition_longs=True)),
        # BASE + transition longs
        ('BASE + transition_longs',
         dict(allow_transition_longs=True)),
        # Wider ema to capture longer trends
        ('P3-best + ema=100',
         dict(adx_thresh=22, min_strength_up=0.15, ema_trend=100)),
        # adx=20 str=0.15 (between 18 and 22)
        ('adx=20 str=0.15 tp=4.5 ppm=2.0 pq=0.50',
         dict(adx_thresh=20, min_strength_up=0.15)),
    ],

    'NVDA': [
        # --- BASELINES ---
        ('BASE',
         {}),
        # Phase 1 winner: just shorts=OFF (already base)
        # Phase 3 winner: vol=0.28 (CAGR 15.44%, Sharpe ~2.12)
        ('P3: vol=0.28',
         dict(vol_target=0.28)),
        # Phase 4 winners
        ('P4a: trail=4.0 tp=3.5 ppm=2.5 pq=0.33  [Sortino 6.66]',
         dict(trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33)),
        ('P4b: trail=3.5 tp=4.0 ppm=2.5 pq=0.33  [Sortino 6.33 CAGR15%]',
         dict(trail_atr=3.5, tp_mult=4.0, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # Phase 5 winner: ema=30 (best Sortino 5.20, lowest DD 4%)
        ('P5: ema=30',
         dict(ema_trend=30)),
        # Combine P3+P4a (vol=0.28 + best exits)
        ('P3+P4a: vol=0.28 trail=4.0 tp=3.5 pq=0.33',
         dict(vol_target=0.28,
              trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # Combine P3+P4b
        ('P3+P4b: vol=0.28 trail=3.5 tp=4.0 pq=0.33',
         dict(vol_target=0.28,
              trail_atr=3.5, tp_mult=4.0, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # Combine P3+P4a+P5
        ('P3+P4a+P5: vol=0.28 trail=4.0 tp=3.5 ema=30',
         dict(vol_target=0.28, ema_trend=30,
              trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # Phase 6 best CAGR (sh=N, adx=25, sr=Y, bb=Y, vt=0.28): CAGR 16.18%
        ('P6-cagr: adx=25 sr+bb vt=0.28 trail=3.5 tp=3.5 ema=50',
         dict(allow_shorts=False, enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=25, min_strength_up=0.20, vol_target=0.28,
              trail_atr=3.5, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.50,
              ema_trend=50)),
        # Phase 6 best Sortino (sh=Y, adx=27, tr=4.0, tp=4.0): Sortino 6.51
        ('P6-sortino: adx=27 tr=4.0 tp=4.0 ema=50 (shorts)',
         dict(allow_shorts=True, max_hold_short=45,
              enable_stoch_rsi=True, enable_bb_signal=True,
              adx_thresh=27, min_strength_up=0.20, vol_target=0.22,
              trail_atr=4.0, tp_mult=4.0, post_partial_mult=2.5, partial_qty_pct=0.33,
              ema_trend=50)),
        # Best overall combo no shorts (P3+P4a+StochRSI)
        ('P3+P4a+sr: vol=0.28 trail=4.0 tp=3.5 pq=0.33 sr=Y',
         dict(vol_target=0.28, enable_stoch_rsi=True,
              trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # Full combo: P3+P4b+ema=30
        ('BEST-GUESS: vol=0.28 trail=4.0 tp=3.5 ema=30 sr=Y',
         dict(vol_target=0.28, ema_trend=30, enable_stoch_rsi=True,
              trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33)),
        # Transition longs on base
        ('BASE + transition_longs',
         dict(allow_transition_longs=True)),
        # Transition longs on P3+P4a
        ('P3+P4a + transition_longs',
         dict(vol_target=0.28,
              trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33,
              allow_transition_longs=True)),
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_sep(char='=', width=148):
    print(char * width)

def print_header():
    print(f"  {'Tick':<5}  {'Config':<55}  {'Sharpe':>6}  {'Sortino':>7}  "
          f"{'CAGR':>7}  {'DD':>5}  {'T':>5}  {'WR':>6}  {'Final$':>11}")
    print("  " + "-" * 119)

def print_row(r):
    print(f"  {r['ticker']:<5}  {r['label']:<55}  "
          f"{r['sharpe']:>6.3f}  {r['sortino']:>7.3f}  "
          f"{r['cagr']:>6.2f}%  {r['max_dd']:>4.1f}%  "
          f"{r['trades']:>5d}  {r['win_rate']:>5.1f}%  ${r['final']:>10,.0f}")

# ---------------------------------------------------------------------------
# Pre-fetch data once per ticker
# ---------------------------------------------------------------------------
print("Pre-fetching data...")
_cache = {}
for ticker, base in BASE_CONFIGS.items():
    s = AggressiveHybridV6(ticker, start=START, end=END, **base)
    s.fetch_data()
    _cache[ticker] = s
    print(f"  {ticker}: {len(s.data)} bars")

def run_cached(ticker, cfg, label):
    ref = _cache[ticker]
    t = AggressiveHybridV6(ticker, start=START, end=END, **cfg)
    t.data = ref.data.copy()
    t.vix  = ref.vix
    if hasattr(ref, '_mvrv'): t._mvrv = ref._mvrv
    if hasattr(ref, '_fg'):   t._fg   = ref._fg
    t.prepare_indicators()
    r = t.backtest()
    return {
        'label':    label[:55],
        'ticker':   ticker,
        'sharpe':   round(r['sharpe'],  3),
        'sortino':  round(r['sortino'], 3),
        'cagr':     round(r['cagr'],    2),
        'max_dd':   round(r['max_dd'],  1),
        'trades':   r['trades'],
        'win_rate': round(r['win_rate'], 1),
        'pf':       round(r['profit_factor'], 2),
        'final':    round(100_000 * (1 + r['cagr'] / 100) ** 10),
        'cfg':      {k: v for k, v in cfg.items() if not callable(v)},
    }

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
all_results = {t: [] for t in BASE_CONFIGS}

print_sep()
print("EQUITY FINALIZER — Combined best-of-phase sweep + Transition Longs")
print_sep()
print_header()

for ticker, candidates in CANDIDATES.items():
    base = BASE_CONFIGS[ticker]
    for label, overrides in candidates:
        cfg = {**base, **overrides}
        r = run_cached(ticker, cfg, label)
        if r:
            all_results[ticker].append(r)
            print_row(r)

# ---------------------------------------------------------------------------
# Final ranking per asset
# ---------------------------------------------------------------------------
print()
print_sep()
print("FINAL RANKING -- Top 12 per asset (Sharpe desc -> Sortino desc -> CAGR desc)")
print_sep()

best_configs = {}
for ticker in BASE_CONFIGS:
    results = all_results[ticker]
    if not results:
        continue
    seen, deduped = set(), []
    for r in results:
        key = (r['sharpe'], r['sortino'], r['cagr'], r['trades'])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    ranked = sorted(deduped, key=lambda x: (-x['sharpe'], -x['sortino'], -x['cagr']))
    print(f"\n  --- {ticker} ---")
    print_header()
    for r in ranked[:12]:
        print_row(r)
    best_configs[ticker] = ranked[0]
    # Also print best Sortino
    sort_ranked = sorted(deduped, key=lambda x: (-x['sortino'], -x['sharpe']))
    if sort_ranked[0] != ranked[0]:
        print(f"  ** Best Sortino: ", end='')
        print_row(sort_ranked[0])

# ---------------------------------------------------------------------------
# Best configs output (paper_trader.py format)
# ---------------------------------------------------------------------------
print()
print_sep()
print("BEST CONFIGS FOR paper_trader.py")
print_sep()
for ticker, r in best_configs.items():
    print(f"\n# {ticker}  Sharpe={r['sharpe']}  Sortino={r['sortino']}  "
          f"CAGR={r['cagr']}%  MaxDD={r['max_dd']}%  Trades={r['trades']}")
    print(f"'{ticker}': {{")
    for k, v in sorted(r['cfg'].items()):
        if v is not None:
            print(f"    {k!r}: {v!r},")
    print("}")

# Save JSON
output = {ticker: {'metrics': {k: v for k, v in r.items() if k not in ('cfg',)},
                   'cfg': r['cfg']}
          for ticker, r in best_configs.items()}
with open('intraday/strategies/equity_finalize_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: intraday/strategies/equity_finalize_results.json")
