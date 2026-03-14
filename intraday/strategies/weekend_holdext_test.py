"""
Weekend strategy: extended-hold and handoff parameter tests.

Questions tested:
  1. Is max_hold=2 optimal, or should weekend trades run into the week?
  2. Does ema_trend=20 need to be shortened, or does ema=145 (GBTC scale) work better?
  3. When extending hold: keep the weekend exit params, or hand off to GBTC exit params
     on the first weekday?

Configs tested (all entry_days={4,5}, BTC-USD):
  A  Current best    : ema=20,  trail=3.5, tp=3.5, hold=2             [baseline]
  B  Extend hold     : ema=20,  trail=3.5, tp=3.5, hold=7             [same params, longer]
  C  Extend hold     : ema=20,  trail=3.5, tp=3.5, hold=14
  D  Extend hold     : ema=20,  trail=3.5, tp=3.5, hold=30
  E  GBTC ema+exits  : ema=145, trail=4.0, tp=3.0, hold=2             [GBTC params, short hold]
  F  GBTC ema+exits  : ema=145, trail=4.0, tp=3.0, hold=14
  G  GBTC ema full   : ema=145, trail=4.0, tp=3.0, hold=90/25/60      [full GBTC params]
  H  Handoff (ema=20): ema=20,  trail=3.5, tp=3.5, hold=90, but on 1st weekday
       → switch to GBTC trail=4.0, tp=3.0, hold=90/25/60
  I  Handoff (ema=145): same as H but EMA=145 entries too

Each row shows:  standalone BTC-USD metrics  |  combined with GBTC portfolio metrics
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from intraday.strategies.aggressive_hybrid_v6_10yr import AggressiveHybridV6
import numpy as np
import pandas as pd

START = '2015-01-01'
END   = '2025-01-01'

# ── GBTC params (current best) ────────────────────────────────────────────────
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

# Handoff dict: values applied to a position on its first weekday after weekend entry
GBTC_HANDOFF = dict(
    trail_atr=4.0, tp_mult=3.0,
    max_hold_trend=90, max_hold_mr=25, max_hold_short=60,
)

# ── Pre-fetch data once per ticker ───────────────────────────────────────────
print("Fetching GBTC data...")
gbtc_base = AggressiveHybridV6('GBTC', start=START, end=END, **GBTC_PARAMS)
gbtc_base.fetch_data()
gbtc_r = gbtc_base.backtest()

print(f"\nGBTC standalone: Sharpe={gbtc_r['sharpe']:.3f}  CAGR={gbtc_r['cagr']:.2f}%  "
      f"MaxDD={gbtc_r['max_dd']:.1f}%  Trades={gbtc_r['trades']}")
# Save equity curve
gbtc_equity = pd.Series(
    gbtc_base.equity_curve,
    index=gbtc_base.data.index[:len(gbtc_base.equity_curve)]
)

print("\nFetching BTC-USD data (once)...")
_btc_ref = AggressiveHybridV6('BTC-USD', start=START, end=END,
    trail_atr=3.5, vol_target=0.60, tp_mult=3.5, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=20, adx_thresh=20, min_strength_up=0.25,
    trail_cushion=0.5, post_partial_mult=2.0,
    macd_fast=8, macd_slow=38,
    max_hold_trend=2, max_hold_mr=2,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=2,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
    entry_days={4, 5},
)
_btc_ref.fetch_data()
print(f"BTC-USD loaded: {len(_btc_ref.data)} bars\n")


# ── Helper: combine two equity curves ───────────────────────────────────────
def combine_equity(eq_a: pd.Series, eq_b: pd.Series):
    """Combine two equity curves (starting at 100k each) into a single $100k portfolio."""
    ret_a = eq_a.pct_change().fillna(0)
    ret_b = eq_b.pct_change().fillna(0)
    combined_ret = ret_a.add(ret_b, fill_value=0)
    combined     = (1 + combined_ret).cumprod() * 100_000
    returns      = combined.pct_change().dropna()
    years        = (combined.index[-1] - combined.index[0]).days / 365.25
    cagr         = (combined.iloc[-1] / 100_000) ** (1 / years) - 1
    sharpe       = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    roll_max     = combined.expanding().max()
    max_dd       = ((combined - roll_max) / roll_max).min() * 100
    return cagr * 100, sharpe, float(max_dd), float(combined.iloc[-1])


# ── Test configs ─────────────────────────────────────────────────────────────
# (label, ema, trail, tp, mh_trend, mh_mr, mh_short, adx, handoff_params or None)
CONFIGS = [
    # ── Current baseline ──────────────────────────────────────────────────────
    ('A: Current (ema=20, hold=2)',         20,  3.5, 3.5,  2,  2,  2,  20, None),
    # ── Extended hold, same weekend params ───────────────────────────────────
    ('B: Extend hold=7  (ema=20)',          20,  3.5, 3.5,  7,  7,  7,  20, None),
    ('C: Extend hold=14 (ema=20)',          20,  3.5, 3.5, 14, 14, 14,  20, None),
    ('D: Extend hold=30 (ema=20)',          20,  3.5, 3.5, 30, 25, 30,  20, None),
    # ── GBTC-scale ema, GBTC exit params ─────────────────────────────────────
    ('E: GBTC params, hold=2  (ema=145)',  145,  4.0, 3.0,  2,  2,  2,  32, None),
    ('F: GBTC params, hold=14 (ema=145)',  145,  4.0, 3.0, 14, 14, 14,  32, None),
    ('G: GBTC params, hold=90 (ema=145)',  145,  4.0, 3.0, 90, 25, 60,  32, None),
    # ── Handoff: enter with weekend params, switch to GBTC on first weekday ──
    ('H: Handoff (ema=20, hold=90 → GBTC)',20,  3.5, 3.5, 90, 25, 60,  20, GBTC_HANDOFF),
    ('I: Handoff (ema=145, hold=90 → GBTC)',145, 4.0, 3.5, 90, 25, 60, 32, GBTC_HANDOFF),
]

print(f"{'Config':<42} "
      f"{'BTC Shr':>7} {'BTC CAGR':>8} {'BTC DD':>7} {'Trd':>5}  "
      f"{'Cmb Shr':>7} {'Cmb CAGR':>8} {'Cmb DD':>7} {'Final $':>10}")
print('─' * 115)

for (label, ema, trail, tp, mht, mhmr, mhs, adx, handoff) in CONFIGS:
    try:
        s = AggressiveHybridV6('BTC-USD', start=START, end=END,
            trail_atr=trail, vol_target=0.60, tp_mult=tp, partial_tp_mult=1.0,
            rsi_period=9, rsi_oversold=33, atr_period=14,
            ema_trend=ema, adx_thresh=adx, min_strength_up=0.25,
            trail_cushion=0.5, post_partial_mult=2.0,
            macd_fast=8, macd_slow=38,
            max_hold_trend=mht, max_hold_mr=mhmr,
            enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
            allow_shorts=True, max_hold_short=mhs,
            use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
            fg_fear_thresh=25, fg_greed_thresh=75,
            entry_days={4, 5},
            handoff_params=handoff,
        )
        # Reuse fetched data
        s.data  = _btc_ref.data.copy()
        s.vix   = _btc_ref.vix
        s._mvrv = _btc_ref._mvrv
        s._fg   = _btc_ref._fg
        s.prepare_indicators()
        r = s.backtest()

        if r.get('trades', 0) < 10:
            print(f"{label:<42}  <10 trades — skipped")
            continue

        btc_equity = pd.Series(
            s.equity_curve,
            index=s.data.index[:len(s.equity_curve)]
        )

        c_cagr, c_sharpe, c_dd, c_final = combine_equity(gbtc_equity, btc_equity)

        print(f"{label:<42} "
              f"{r['sharpe']:>7.3f} {r['cagr']:>8.2f} {r['max_dd']:>7.1f} {r['trades']:>5}  "
              f"{c_sharpe:>7.3f} {c_cagr:>8.2f} {c_dd:>7.1f} {c_final:>10,.0f}")

    except Exception as e:
        print(f"{label:<42}  ERROR: {e}")

print()
print(f"Reference → GBTC alone:  "
      f"Sharpe={gbtc_r['sharpe']:.3f}  CAGR={gbtc_r['cagr']:.2f}%  "
      f"MaxDD={gbtc_r['max_dd']:.1f}%  Final=${100_000 * (1 + gbtc_r['cagr']/100)**10:,.0f}")
