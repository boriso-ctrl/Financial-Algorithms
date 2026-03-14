"""
EMA Lead Comparison — True vs Adjusted EMA for GBTC / BTC-USD
==============================================================

Problem:
  GBTC trades Mon-Fri only (~252 bars/year).
  BTC-USD trades 24/7   (~365 bars/year).

  EMA(100) on GBTC covers 100 trading days  ≈ 145 calendar days (~4.8 months).
  EMA(100) on BTC-USD covers 100 calendar days ≈ 3.3 months — a shorter look-back.

  Since GBTC is a derivative of BTC-USD, the 24/7 BTC-USD chart is the
  *leading* chart.  Using BTC-USD's EMA for GBTC's regime classification
  should produce better-timed trend gates.

Calendar-equivalent period:
  BTC_EMA_period = GBTC_EMA_period × (365 / 252) ≈ period × 1.448

Variants tested:
  1. GBTC  — native EMA(100)           [baseline]
  2. GBTC  — BTC-lead EMA(100)         [same period, 24/7 data]
  3. GBTC  — BTC-lead EMA(145)         [calendar-adjusted, same calendar window]
  4. BTC-USD — native EMA(100)         [baseline]
  5. BTC-USD — native EMA(145)         [calendar-adjusted to match GBTC EMA(100)]
"""

import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'intraday/strategies')

import yfinance as yf
import numpy as np
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

# ── V10 best params (frozen) ──────────────────────────────────────────────────
BASE_PARAMS = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=100, adx_thresh=32, min_strength_up=0.30,
    trail_cushion=0.5, post_partial_mult=2.5,
    macd_fast=8, macd_slow=38, macd_sig=9,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=60,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
)

START = '2015-01-01'
END   = '2025-12-31'

# Calendar-equivalent BTC period for GBTC EMA(100)
# GBTC 100 trading days ≈ 100 × (365/252) ≈ 145 calendar days
GBTC_EMA   = BASE_PARAMS['ema_trend']          # 100 trading days
CAL_ADJ_PERIOD = round(GBTC_EMA * 365 / 252)   # ≈ 145 calendar days


# ── Fetch BTC-USD 24/7 data once ─────────────────────────────────────────────
print(f"Fetching BTC-USD (24/7 lead data) {START} → {END} ...")
t0 = time.time()
_raw_btc = yf.download('BTC-USD', start=START, end=END, progress=False, auto_adjust=True)
if isinstance(_raw_btc.columns, __import__('pandas').MultiIndex):
    _raw_btc.columns = _raw_btc.columns.get_level_values(0)
btc_df = _raw_btc
print(f"BTC-USD: {len(btc_df)} bars  (fetch {time.time()-t0:.1f}s)")
print(f"GBTC EMA({GBTC_EMA}) trading days ≈ {CAL_ADJ_PERIOD} calendar days")
print(f"Calendar-adjusted BTC period: EMA({CAL_ADJ_PERIOD})\n")


def run_variant(label, ticker, ema_period, lead_df=None, lead_period=None):
    """Run one backtest variant and return (label, metrics_dict)."""
    params = {**BASE_PARAMS, 'ema_trend': ema_period}
    strat = AggressiveHybridV6(ticker, start=START, end=END, **params)
    ok = strat.fetch_data()
    if not ok:
        print(f"  {label}: DATA FETCH FAILED")
        return label, None
    if lead_df is not None:
        strat.lead_ema_df     = lead_df
        strat.lead_ema_period = lead_period   # period on lead source
    r = strat.backtest()
    n_bars = len(strat.data)
    r['_bars']  = n_bars
    r['_label'] = label
    return label, r


# ── Run all variants ──────────────────────────────────────────────────────────
variants = [
    # (label,                      ticker,     ema_period,       lead_df, lead_period)
    ("GBTC  | native  EMA(100)",  'GBTC',     GBTC_EMA,         None,    None),
    ("GBTC  | BTC-lead EMA(100)", 'GBTC',     GBTC_EMA,         btc_df,  GBTC_EMA),
    (f"GBTC  | BTC-lead EMA({CAL_ADJ_PERIOD})", 'GBTC', GBTC_EMA, btc_df, CAL_ADJ_PERIOD),
    ("BTCUSD| native  EMA(100)",  'BTC-USD',  GBTC_EMA,         None,    None),
    (f"BTCUSD| native  EMA({CAL_ADJ_PERIOD})", 'BTC-USD', CAL_ADJ_PERIOD, None, None),
]

results = []
for label, ticker, ema_period, lead_df, lead_period in variants:
    print(f"Running: {label} ...")
    t0 = time.time()
    lbl, r = run_variant(label, ticker, ema_period, lead_df, lead_period)
    if r:
        print(f"  → Sharpe {r['sharpe']:.3f}  CAGR {r['cagr']:.1f}%  Trades {r['trades']}"
              f"  WinRate {r['win_rate']:.0f}%  MaxDD {r['max_dd']:.1f}%  ({time.time()-t0:.0f}s)")
        results.append(r)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 95)
print(f"{'Variant':<38} {'Bars':>5} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} "
      f"{'Trades':>7} {'WinR%':>6} {'MaxDD%':>7} {'PF':>5}")
print("=" * 95)
for r in results:
    print(f"{r['_label']:<38} {r['_bars']:>5} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
          f"{r['cagr']:>7.1f} {r['trades']:>7} {r['win_rate']:>6.1f} "
          f"{r['max_dd']:>7.1f} {r['profit_factor']:>5.2f}")
print("=" * 95)

# ── Interpretation guide ──────────────────────────────────────────────────────
print(f"""
Key questions answered:
  • GBTC native vs BTC-lead EMA(100):  same period, does 24/7 data improve regime timing?
  • GBTC native vs BTC-lead EMA({CAL_ADJ_PERIOD}): calendar-equivalent window — expected best for GBTC.
  • BTC-USD EMA(100) vs EMA({CAL_ADJ_PERIOD}):  does the longer look-back improve BTC-USD standalone?

Bar-count context:
  GBTC  ≈ 252 bars/yr (Mon-Fri only)
  BTC-USD ≈ 365 bars/yr (24/7)
  GBTC EMA({GBTC_EMA}) ≈ BTC-USD EMA({CAL_ADJ_PERIOD}) in calendar time
""")
