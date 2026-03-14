"""
Equity Strategy Optimizer — Autonomous sweep for QQQ, XLK, NVDA.

Tests:
  1. Shorts on/off
  2. StochRSI + BB signals on/off
  3. ADX threshold variations (catch more waves)
  4. min_strength_up variations (more entry frequency)
  5. vol_target variations (position sizing)
  6. partial_qty_pct and post_partial_mult variations
  7. ema_trend variations (faster trend detection)
  8. tp_mult variations (wider / narrower exits)

Outputs: ranked table per asset sorted by Sharpe, then Sortino.
Also outputs best config per asset for paper_trader.py.

Start: 2015-01-01  End: 2025-01-01  (10 years, clean OOS boundary)
"""
import sys, warnings, itertools, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from intraday.strategies.aggressive_hybrid_v6_10yr import AggressiveHybridV6

START = '2015-01-01'
END   = '2025-01-01'

# ── Base (current production) configs for each equity ─────────────────────────
BASE_CONFIGS = {
    'QQQ': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=3.0, partial_tp_mult=1.5,
        rsi_period=9, rsi_oversold=38, atr_period=14,
        ema_trend=30, adx_thresh=18, min_strength_up=0.30,
        trail_cushion=1.5, post_partial_mult=1.5,
        macd_fast=8, macd_slow=38,
        max_hold_trend=60, max_hold_mr=25,
        allow_shorts=False, enable_stoch_rsi=False, enable_bb_signal=False,
        partial_qty_pct=0.50, vol_regime_scale=1.1,
    ),
    'XLK': dict(
        trail_atr=3.0, vol_target=0.22, tp_mult=4.5, partial_tp_mult=1.5,
        rsi_period=14, rsi_oversold=38, atr_period=14,
        ema_trend=80, adx_thresh=22, min_strength_up=0.20,
        trail_cushion=1.0, post_partial_mult=2.0,
        macd_fast=8, macd_slow=26,
        max_hold_trend=80, max_hold_mr=25,
        allow_shorts=False, enable_stoch_rsi=False, enable_bb_signal=False,
        partial_qty_pct=0.50, vol_regime_scale=1.1,
    ),
    'NVDA': dict(
        trail_atr=3.5, vol_target=0.22, tp_mult=3.0, partial_tp_mult=1.5,
        rsi_period=9, rsi_oversold=38, atr_period=20,
        ema_trend=50, adx_thresh=27, min_strength_up=0.25,
        trail_cushion=2.0, post_partial_mult=2.0,
        macd_fast=12, macd_slow=26,
        max_hold_trend=60, max_hold_mr=25,
        allow_shorts=False, enable_stoch_rsi=False, enable_bb_signal=True,
        partial_qty_pct=0.67, vol_regime_scale=1.1,
    ),
}

# ── Sweep dimensions ────────────────────────────────────────────────────────
# Each key maps to a list of values to try. The cartesian product would be too
# large, so we use a structured approach: sweep one or two dims at a time around
# the base, keeping the rest at base. This avoids overfitting while finding
# locally optimal configs.

# Phase 1: Shorts on/off
# Phase 2: Signal extras (StochRSI + BB)
# Phase 3: ADX / min_strength / vol_target
# Phase 4: EMA / TP / trail

PHASE1_SHORTS = [False, True]

PHASE2_SIGNALS = [
    dict(enable_stoch_rsi=False, enable_bb_signal=False),  # base
    dict(enable_stoch_rsi=True,  enable_bb_signal=False),
    dict(enable_stoch_rsi=False, enable_bb_signal=True),
    dict(enable_stoch_rsi=True,  enable_bb_signal=True),
]

# Per-asset dims (wider search space)
PHASE3_PER_ASSET = {
    'QQQ': [
        dict(adx_thresh=14, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=14, min_strength_up=0.25, vol_target=0.22),
        dict(adx_thresh=18, min_strength_up=0.20, vol_target=0.22),  # base
        dict(adx_thresh=18, min_strength_up=0.25, vol_target=0.22),
        dict(adx_thresh=18, min_strength_up=0.30, vol_target=0.22),
        dict(adx_thresh=18, min_strength_up=0.20, vol_target=0.28),
        dict(adx_thresh=20, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=20, min_strength_up=0.20, vol_target=0.28),
    ],
    'XLK': [
        dict(adx_thresh=18, min_strength_up=0.15, vol_target=0.22),
        dict(adx_thresh=18, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=20, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=22, min_strength_up=0.20, vol_target=0.22),  # base
        dict(adx_thresh=22, min_strength_up=0.15, vol_target=0.22),
        dict(adx_thresh=22, min_strength_up=0.20, vol_target=0.28),
        dict(adx_thresh=25, min_strength_up=0.20, vol_target=0.22),
    ],
    'NVDA': [
        dict(adx_thresh=22, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=25, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=27, min_strength_up=0.25, vol_target=0.22),  # base
        dict(adx_thresh=27, min_strength_up=0.20, vol_target=0.22),
        dict(adx_thresh=27, min_strength_up=0.25, vol_target=0.28),
        dict(adx_thresh=30, min_strength_up=0.25, vol_target=0.22),
    ],
}

PHASE4_EXITS = {
    'QQQ': [
        dict(trail_atr=3.0, tp_mult=3.0, post_partial_mult=1.5, partial_qty_pct=0.33),
        dict(trail_atr=3.5, tp_mult=3.0, post_partial_mult=1.5, partial_qty_pct=0.50),  # base
        dict(trail_atr=3.5, tp_mult=4.0, post_partial_mult=2.0, partial_qty_pct=0.50),
        dict(trail_atr=4.0, tp_mult=3.0, post_partial_mult=2.0, partial_qty_pct=0.33),
        dict(trail_atr=4.0, tp_mult=4.0, post_partial_mult=2.0, partial_qty_pct=0.50),
        dict(trail_atr=3.5, tp_mult=3.5, post_partial_mult=2.0, partial_qty_pct=0.50),
    ],
    'XLK': [
        dict(trail_atr=2.5, tp_mult=4.5, post_partial_mult=2.0, partial_qty_pct=0.33),
        dict(trail_atr=3.0, tp_mult=4.5, post_partial_mult=2.0, partial_qty_pct=0.50),  # base
        dict(trail_atr=3.0, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.33),
        dict(trail_atr=3.5, tp_mult=4.5, post_partial_mult=2.0, partial_qty_pct=0.33),
        dict(trail_atr=3.5, tp_mult=5.0, post_partial_mult=2.5, partial_qty_pct=0.50),
        dict(trail_atr=3.0, tp_mult=4.0, post_partial_mult=2.0, partial_qty_pct=0.50),
    ],
    'NVDA': [
        dict(trail_atr=3.0, tp_mult=3.0, post_partial_mult=2.0, partial_qty_pct=0.50),
        dict(trail_atr=3.5, tp_mult=3.0, post_partial_mult=2.0, partial_qty_pct=0.67),  # base
        dict(trail_atr=3.5, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.50),
        dict(trail_atr=4.0, tp_mult=3.0, post_partial_mult=2.0, partial_qty_pct=0.50),
        dict(trail_atr=4.0, tp_mult=3.5, post_partial_mult=2.5, partial_qty_pct=0.33),
        dict(trail_atr=3.5, tp_mult=4.0, post_partial_mult=2.5, partial_qty_pct=0.33),
    ],
}

PHASE5_EMA = {
    'QQQ': [20, 25, 30, 40, 50],
    'XLK': [50, 60, 80, 100],
    'NVDA': [30, 40, 50, 60, 80],
}


def run_one(ticker, cfg, label='') -> dict:
    t = AggressiveHybridV6(ticker, start=START, end=END, **cfg)
    if not t.fetch_data():
        return {}
    r = t.backtest()
    return {
        'label':    label,
        'ticker':   ticker,
        'sharpe':   round(r['sharpe'], 3),
        'sortino':  round(r['sortino'], 3),
        'cagr':     round(r['cagr'], 2),
        'max_dd':   round(r['max_dd'], 1),
        'trades':   r['trades'],
        'win_rate': round(r['win_rate'], 1),
        'pf':       round(r['profit_factor'], 2),
        'final':    round(100_000 * (1 + r['cagr'] / 100) ** 10),
        'cfg':      cfg,
    }


def print_row(r):
    print(f"  {r['ticker']:<5}  {r['label']:<52}  "
          f"Sh={r['sharpe']:>6.3f}  So={r['sortino']:>6.3f}  "
          f"CAGR={r['cagr']:>6.2f}%  DD={r['max_dd']:>5.1f}%  "
          f"T={r['trades']:>4}  WR={r['win_rate']:>5.1f}%  ${r['final']:>10,.0f}")


def print_header():
    print(f"  {'Tick':<5}  {'Config':<52}  {'Sharpe':>7}  {'Sortino':>7}  "
          f"{'CAGR':>7}  {'  DD':>6}  {'T':>5}  {'WinR':>6}  {'Final $':>12}")
    print("  " + "-" * 130)


# ── Pre-fetch data once per ticker ─────────────────────────────────────────
print("Pre-fetching data...")
_data_cache = {}
for ticker in BASE_CONFIGS:
    base = BASE_CONFIGS[ticker]
    s = AggressiveHybridV6(ticker, start=START, end=END, **base)
    s.fetch_data()
    _data_cache[ticker] = s
    print(f"  {ticker}: {len(s.data)} bars")

def run_cached(ticker, cfg, label='') -> dict:
    """Run backtest reusing cached data from the base strategy instance."""
    ref = _data_cache[ticker]
    t = AggressiveHybridV6(ticker, start=START, end=END, **cfg)
    t.data  = ref.data.copy()
    t.vix   = ref.vix
    if hasattr(ref, '_mvrv'): t._mvrv = ref._mvrv
    if hasattr(ref, '_fg'):   t._fg   = ref._fg
    t.prepare_indicators()
    r = t.backtest()
    return {
        'label':    label,
        'ticker':   ticker,
        'sharpe':   round(r['sharpe'], 3),
        'sortino':  round(r['sortino'], 3),
        'cagr':     round(r['cagr'], 2),
        'max_dd':   round(r['max_dd'], 1),
        'trades':   r['trades'],
        'win_rate': round(r['win_rate'], 1),
        'pf':       round(r['profit_factor'], 2),
        'final':    round(100_000 * (1 + r['cagr'] / 100) ** 10),
        'cfg':      {k: v for k, v in cfg.items() if not callable(v)},
    }

all_results = {t: [] for t in BASE_CONFIGS}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Baseline + Shorts
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 1 — Shorts ON vs OFF")
print("=" * 140)
print_header()

for ticker, base in BASE_CONFIGS.items():
    for shorts in PHASE1_SHORTS:
        cfg = {**base, 'allow_shorts': shorts,
               'max_hold_short': 45 if shorts else None}
        label = f"shorts={'ON ' if shorts else 'OFF'}"
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Signal extras (StochRSI / BB)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 2 — Signal Extras (StochRSI + BB)")
print("=" * 140)
print_header()

for ticker, base in BASE_CONFIGS.items():
    for sig in PHASE2_SIGNALS:
        cfg = {**base, **sig}
        label = f"stoch={'Y' if sig['enable_stoch_rsi'] else 'N'} bb={'Y' if sig['enable_bb_signal'] else 'N'}"
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — ADX / min_strength / vol_target
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 3 — ADX / min_strength / vol_target")
print("=" * 140)
print_header()

for ticker, variants in PHASE3_PER_ASSET.items():
    base = BASE_CONFIGS[ticker]
    for v in variants:
        cfg = {**base, **v}
        label = f"adx={v['adx_thresh']} str={v['min_strength_up']:.2f} vol={v['vol_target']:.2f}"
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Exit mechanics (trail / TP / partial)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 4 — Exit Mechanics (trail / TP / partial)")
print("=" * 140)
print_header()

for ticker, variants in PHASE4_EXITS.items():
    base = BASE_CONFIGS[ticker]
    for v in variants:
        cfg = {**base, **v}
        label = (f"trail={v['trail_atr']} tp={v['tp_mult']} "
                 f"ppm={v['post_partial_mult']} pq={v['partial_qty_pct']:.2f}")
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — EMA period (trend regime speed)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 5 — EMA Trend Period")
print("=" * 140)
print_header()

for ticker, emas in PHASE5_EMA.items():
    base = BASE_CONFIGS[ticker]
    for ema in emas:
        cfg = {**base, 'ema_trend': ema}
        label = f"ema_trend={ema}"
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Best combo sweep (top combinations from above phases)
# Take best shorts setting + best signals + best adx/str per asset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 6 — Best Combo Sweep (cross-phase combinations)")
print("=" * 140)
print_header()

COMBO_GRID = {
    'QQQ': [
        # (shorts, stoch_rsi, bb, adx, min_str, vol, trail, tp, ppm, pq, ema)
        (True,  True,  True,  14, 0.20, 0.22, 3.5, 3.5, 2.0, 0.50, 25),
        (True,  True,  True,  14, 0.20, 0.28, 3.5, 3.5, 2.0, 0.50, 25),
        (True,  True,  True,  14, 0.25, 0.22, 3.5, 4.0, 2.0, 0.50, 30),
        (False, True,  True,  14, 0.20, 0.22, 3.5, 3.5, 2.0, 0.50, 25),
        (False, True,  True,  14, 0.20, 0.28, 3.5, 4.0, 2.0, 0.33, 30),
        (True,  False, True,  14, 0.20, 0.22, 4.0, 4.0, 2.0, 0.50, 25),
        (True,  True,  False, 18, 0.20, 0.22, 3.5, 3.5, 2.0, 0.50, 30),
        (True,  True,  True,  18, 0.20, 0.28, 4.0, 3.5, 2.0, 0.33, 25),
    ],
    'XLK': [
        (True,  True,  True,  18, 0.15, 0.22, 3.0, 5.0, 2.5, 0.33, 60),
        (True,  True,  True,  18, 0.20, 0.22, 3.0, 5.0, 2.5, 0.33, 60),
        (True,  True,  True,  20, 0.15, 0.22, 3.0, 4.5, 2.0, 0.33, 80),
        (False, True,  True,  18, 0.15, 0.22, 3.0, 5.0, 2.5, 0.33, 60),
        (False, True,  True,  18, 0.15, 0.28, 3.0, 5.0, 2.5, 0.33, 60),
        (True,  True,  False, 18, 0.15, 0.22, 3.0, 5.0, 2.5, 0.50, 60),
        (True,  False, True,  18, 0.20, 0.22, 3.5, 5.0, 2.5, 0.33, 60),
        (True,  True,  True,  22, 0.15, 0.28, 3.0, 5.0, 2.5, 0.33, 80),
    ],
    'NVDA': [
        (True,  True,  True,  22, 0.20, 0.22, 3.5, 3.5, 2.5, 0.50, 40),
        (True,  True,  True,  25, 0.20, 0.22, 3.5, 3.5, 2.5, 0.50, 50),
        (True,  True,  True,  25, 0.25, 0.22, 4.0, 3.5, 2.5, 0.33, 50),
        (False, True,  True,  22, 0.20, 0.22, 3.5, 3.5, 2.5, 0.50, 40),
        (False, True,  True,  25, 0.20, 0.28, 3.5, 3.5, 2.5, 0.50, 50),
        (True,  True,  False, 22, 0.20, 0.22, 3.5, 3.5, 2.0, 0.67, 50),
        (True,  True,  True,  27, 0.20, 0.22, 4.0, 4.0, 2.5, 0.33, 50),
        (True,  False, True,  25, 0.20, 0.22, 3.5, 3.5, 2.5, 0.50, 40),
    ],
}

BASE_EMA = {'QQQ': 30, 'XLK': 80, 'NVDA': 50}

for ticker, combos in COMBO_GRID.items():
    base = BASE_CONFIGS[ticker]
    for (sh, sr, bb, adx, ms, vt, ta, tp, ppm, pq, ema) in combos:
        cfg = {
            **base,
            'allow_shorts': sh,   'max_hold_short': 45 if sh else None,
            'enable_stoch_rsi': sr, 'enable_bb_signal': bb,
            'adx_thresh': adx,    'min_strength_up': ms,
            'vol_target': vt,     'trail_atr': ta,
            'tp_mult': tp,        'post_partial_mult': ppm,
            'partial_qty_pct': pq, 'ema_trend': ema,
        }
        label = (f"sh={'Y' if sh else 'N'} sr={'Y' if sr else 'N'} bb={'Y' if bb else 'N'} "
                 f"adx={adx} str={ms:.2f} vt={vt:.2f} tr={ta} tp={tp} ema={ema}")
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL RANKING — Best per asset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("FINAL RANKING — Top 10 per asset (by Sharpe, then Sortino)")
print("=" * 140)

best_configs = {}

for ticker in BASE_CONFIGS:
    results = all_results[ticker]
    if not results:
        continue
    # Deduplicate (same Sharpe/Sortino/CAGR combo → same effective run)
    seen = set()
    deduped = []
    for r in results:
        key = (r['sharpe'], r['sortino'], r['cagr'], r['trades'])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    ranked = sorted(deduped, key=lambda x: (-x['sharpe'], -x['sortino'], -x['cagr']))
    print(f"\n  ── {ticker} ──")
    print_header()
    for r in ranked[:10]:
        print_row(r)
    best_configs[ticker] = ranked[0]

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT BEST CONFIGS AS DICT SNIPPETS (for copy-paste into paper_trader.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("BEST CONFIGS (paper_trader.py format)")
print("=" * 140)
for ticker, r in best_configs.items():
    print(f"\n# {ticker}  Sharpe={r['sharpe']}  Sortino={r['sortino']}  "
          f"CAGR={r['cagr']}%  MaxDD={r['max_dd']}%  Trades={r['trades']}")
    print(f"'{ticker}': {{")
    for k, v in r['cfg'].items():
        if v is not None:
            print(f"    {k!r}: {v!r},")
    print("}")

# Save to JSON for later reference
output = {ticker: {'metrics': {k: v for k, v in r.items() if k != 'cfg'},
                   'cfg': r['cfg']}
          for ticker, r in best_configs.items()}
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Transition longs (allow_transition_longs=True)
# Sweep on top of best combo so far per asset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("PHASE 7 — Transition Longs (allow_transition_longs=True)")
print("=" * 140)
print_header()

for ticker, r_best in best_configs.items():
    base_cfg = r_best['cfg']
    for tl in [False, True]:
        cfg = {**base_cfg, 'allow_transition_longs': tl}
        label = f"transition_longs={'ON ' if tl else 'OFF'} (on top of best combo)"
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)
    # Also sweep transition longs against base config
    for tl in [False, True]:
        cfg = {**BASE_CONFIGS[ticker], 'allow_transition_longs': tl}
        label = f"transition_longs={'ON ' if tl else 'OFF'} (base config)"
        r = run_cached(ticker, cfg, label)
        if r: all_results[ticker].append(r); print_row(r)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL RE-RANKING (includes transition longs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 140)
print("FINAL RANKING incl. Transition Longs — Top 10 per asset")
print("=" * 140)

for ticker in BASE_CONFIGS:
    results = all_results[ticker]
    if not results:
        continue
    seen = set()
    deduped = []
    for r in results:
        key = (r['sharpe'], r['sortino'], r['cagr'], r['trades'])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    ranked = sorted(deduped, key=lambda x: (-x['sharpe'], -x['sortino'], -x['cagr']))
    print(f"\n  ── {ticker} ──")
    print_header()
    for r in ranked[:10]:
        print_row(r)
    best_configs[ticker] = ranked[0]

print("\n" + "=" * 140)
print("BEST CONFIGS (paper_trader.py format)")
print("=" * 140)
for ticker, r in best_configs.items():
    print(f"\n# {ticker}  Sharpe={r['sharpe']}  Sortino={r['sortino']}  "
          f"CAGR={r['cagr']}%  MaxDD={r['max_dd']}%  Trades={r['trades']}")
    print(f"'{ticker}': {{")
    for k, v in r['cfg'].items():
        if v is not None:
            print(f"    {k!r}: {v!r},")
    print("}")

with open('intraday/strategies/equity_optimizer_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print("\nResults saved to intraday/strategies/equity_optimizer_results.json")
