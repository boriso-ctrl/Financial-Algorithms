"""
Overlap MaxDD mitigation test — two approaches + combinations.

Baseline: Config G (ema=145, trail=4.0, tp=3.0, hold=90, no handoff)
Combined MaxDD = -12.4%, CAGR = 25.38%

Problem: when BTC-USD hold extends into weekdays, both legs run simultaneously
on 98%-correlated assets → concurrent drawdowns stack.

Part 1 — Capital allocation splits (same Config G BTC equity curve)
    GBTC:BTC split  90:10 → 50:50

Part 2 — Tighter trail on weekday handoff (full $100k split: 50/50 for now)
    Enter with GBTC params (trail=4.0), but on first weekday tighten to
    trail=3.0 / 2.5 / 2.0 / 1.5 — stops BTC-USD from bleeding into overlap.

Part 3 — Best trail tightening × best allocation split (small grid)
"""
import sys, warnings, io
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
SEP = '-' * 95
SEP65 = '-' * 65
from intraday.strategies.aggressive_hybrid_v6_10yr import AggressiveHybridV6
import numpy as np
import pandas as pd

START = '2015-01-01'
END   = '2025-01-01'

# ── GBTC params ───────────────────────────────────────────────────────────────
GBTC_PARAMS = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=145, adx_thresh=32, min_strength_up=0.30,
    trail_cushion=0.5, post_partial_mult=2.5,
    macd_fast=8, macd_slow=38,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=60,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
    signal_ticker='BTC-USD', signal_ema_period=180,
)

# BTC weekend params — Config G baseline (ema=145, full hold)
BTC_BASE = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=145, adx_thresh=32, min_strength_up=0.25,
    trail_cushion=0.5, post_partial_mult=2.0,
    macd_fast=8, macd_slow=38,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=60,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
    entry_days={4, 5},
)


# ── Portfolio combiner with allocation split ──────────────────────────────────
def combine(eq_a: pd.Series, eq_b: pd.Series, alloc_a: float = 0.5):
    """Combine two equity curves with a capital split (alloc_a = GBTC share)."""
    alloc_b = 1.0 - alloc_a
    ret_a   = eq_a.pct_change().fillna(0)
    ret_b   = eq_b.pct_change().fillna(0)
    comb_r  = ret_a.multiply(alloc_a).add(ret_b.multiply(alloc_b), fill_value=0)
    curve   = (1 + comb_r).cumprod() * 100_000
    rets    = curve.pct_change().dropna()
    years   = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr    = (curve.iloc[-1] / 100_000) ** (1 / years) - 1
    sh      = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    roll    = curve.expanding().max()
    mdd     = ((curve - roll) / roll).min() * 100
    return cagr * 100, sh, float(mdd), float(curve.iloc[-1])


# ── Fetch data ────────────────────────────────────────────────────────────────
print("Fetching GBTC...")
gbtc = AggressiveHybridV6('GBTC', start=START, end=END, **GBTC_PARAMS)
gbtc.fetch_data()
gbtc_r  = gbtc.backtest()
gbtc_eq = pd.Series(gbtc.equity_curve,
                    index=gbtc.data.index[:len(gbtc.equity_curve)])
print(f"GBTC:  Sharpe={gbtc_r['sharpe']:.3f}  CAGR={gbtc_r['cagr']:.2f}%  MaxDD={gbtc_r['max_dd']:.1f}%")

print("\nFetching BTC-USD (once)...")
_ref = AggressiveHybridV6('BTC-USD', start=START, end=END, **BTC_BASE)
_ref.fetch_data()
print(f"BTC-USD: {len(_ref.data)} bars loaded\n")


def run_btc(handoff=None):
    """Run BTC-USD Config-G with optional handoff_params and return equity Series + results."""
    s = AggressiveHybridV6('BTC-USD', start=START, end=END,
                           handoff_params=handoff, **BTC_BASE)
    s.data  = _ref.data.copy()
    s.vix   = _ref.vix
    s._mvrv = _ref._mvrv
    s._fg   = _ref._fg
    s.prepare_indicators()
    r = s.backtest()
    eq = pd.Series(s.equity_curve, index=s.data.index[:len(s.equity_curve)])
    return eq, r


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 95)
print("PART 1 — Capital allocation splits  (Config G, no trail tightening)")
print("=" * 95)
print(f"{'Alloc GBTC:BTC':<18} {'Cmb Sharpe':>10} {'Cmb CAGR':>9} {'Cmb MaxDD':>10} {'Final $':>11}")
print('─' * 65)

# Run Config G BTC once (no handoff)
btc_g_eq, btc_g_r = run_btc(handoff=None)
print(f"  BTC-USD alone:  Sharpe={btc_g_r['sharpe']:.3f}  CAGR={btc_g_r['cagr']:.2f}%  "
      f"MaxDD={btc_g_r['max_dd']:.1f}%  Trades={btc_g_r['trades']}")

for alloc in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]:
    label = f"{int(alloc*100):3d}:{int((1-alloc)*100):2d}"
    if alloc == 1.00:
        # GBTC alone
        cagr = gbtc_r['cagr']; sh = gbtc_r['sharpe']; mdd = gbtc_r['max_dd']
        fin  = 100_000 * ((1+cagr/100)**10)
        print(f"  {label:<16}  {sh:>10.3f} {cagr:>9.2f} {mdd:>10.1f} {fin:>11,.0f}  ← GBTC alone")
        continue
    cagr, sh, mdd, fin = combine(gbtc_eq, btc_g_eq, alloc_a=alloc)
    print(f"  {label:<16}  {sh:>10.3f} {cagr:>9.2f} {mdd:>10.1f} {fin:>11,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 95)
print("PART 2 — Trail tightening on weekday handoff  (50/50 split, full hold=90)")
print("  On the first weekday after a weekend entry, trail_atr is tightened")
print("=" * 95)
print(f"{'Entry trail → Wkday trail':<28} {'BTC Shr':>8} {'BTC CAGR':>9} {'BTC DD':>8} "
      f"{'Cmb Shr':>8} {'Cmb CAGR':>9} {'Cmb DD':>8} {'Final $':>11}")
print('─' * 95)

for entry_trail, wkday_trail in [
    (4.0, None),   # No handoff (baseline)
    (4.0, 3.0),
    (4.0, 2.5),
    (4.0, 2.0),
    (4.0, 1.5),
    (3.5, 2.5),    # Tighter entry trail too
    (3.0, 2.0),
    (3.5, 1.5),
]:
    hp = None
    label = f"trail={entry_trail:.1f} → no handoff    "
    if wkday_trail is not None:
        hp = dict(trail_atr=wkday_trail, tp_mult=3.0,
                  max_hold_trend=90, max_hold_mr=25, max_hold_short=60)
        label = f"trail={entry_trail:.1f} → wkday {wkday_trail:.1f}        "

    # Override trail_atr for entry
    import copy
    params = copy.copy(BTC_BASE)
    params['trail_atr'] = entry_trail
    s = AggressiveHybridV6('BTC-USD', start=START, end=END,
                           handoff_params=hp, **params)
    s.data  = _ref.data.copy()
    s.vix   = _ref.vix
    s._mvrv = _ref._mvrv
    s._fg   = _ref._fg
    s.prepare_indicators()
    r = s.backtest()
    btc_eq = pd.Series(s.equity_curve, index=s.data.index[:len(s.equity_curve)])

    cagr, sh, mdd, fin = combine(gbtc_eq, btc_eq, alloc_a=0.50)
    print(f"  {label:<26} {r['sharpe']:>8.3f} {r['cagr']:>9.2f} {r['max_dd']:>8.1f} "
          f"{sh:>8.3f} {cagr:>9.2f} {mdd:>8.1f} {fin:>11,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 95)
print("PART 3 — Best trail tightening × capital allocation  (small grid)")
print("  entry trail=4.0 → weekday trail=2.5  (based on Part 2 results)")
print("=" * 95)
print(f"{'Alloc GBTC:BTC':<18} {'Cmb Sharpe':>10} {'Cmb CAGR':>9} {'Cmb MaxDD':>10} {'Final $':>11}")
print('─' * 65)

hp_best = dict(trail_atr=2.5, tp_mult=3.0,
               max_hold_trend=90, max_hold_mr=25, max_hold_short=60)
btc_t_eq, btc_t_r = run_btc(handoff=hp_best)
print(f"  BTC-USD (trail→2.5): Sharpe={btc_t_r['sharpe']:.3f}  CAGR={btc_t_r['cagr']:.2f}%"
      f"  MaxDD={btc_t_r['max_dd']:.1f}%  Trades={btc_t_r['trades']}")

for alloc in [0.90, 0.80, 0.70, 0.60, 0.50]:
    label = f"{int(alloc*100):3d}:{int((1-alloc)*100):2d}"
    cagr, sh, mdd, fin = combine(gbtc_eq, btc_t_eq, alloc_a=alloc)
    print(f"  {label:<16}  {sh:>10.3f} {cagr:>9.2f} {mdd:>10.1f} {fin:>11,.0f}")

# Also try trail→1.5
hp_tight = dict(trail_atr=1.5, tp_mult=3.0,
                max_hold_trend=90, max_hold_mr=25, max_hold_short=60)
btc_lt_eq, btc_lt_r = run_btc(handoff=hp_tight)
print(f"\n  BTC-USD (trail→1.5): Sharpe={btc_lt_r['sharpe']:.3f}  CAGR={btc_lt_r['cagr']:.2f}%"
      f"  MaxDD={btc_lt_r['max_dd']:.1f}%  Trades={btc_lt_r['trades']}")
for alloc in [0.80, 0.70, 0.60]:
    label = f"{int(alloc*100):3d}:{int((1-alloc)*100):2d}"
    cagr, sh, mdd, fin = combine(gbtc_eq, btc_lt_eq, alloc_a=alloc)
    print(f"  {label:<16}  {sh:>10.3f} {cagr:>9.2f} {mdd:>10.1f} {fin:>11,.0f}")

print(f"\n  Reference → GBTC alone: Sharpe={gbtc_r['sharpe']:.3f}  CAGR={gbtc_r['cagr']:.2f}%"
      f"  MaxDD={gbtc_r['max_dd']:.1f}%")
