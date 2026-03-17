"""
V11 Alpha Signal Comparison
============================
Tests each new alpha signal extension individually and combined vs the V10 baseline.

New signals tested:
  1. Williams %R (exhaustion oscillator)
  2. RSI/Price Divergence (bullish/bearish divergence detection)
  3. VWAP Z-score (mean-reversion from rolling VWAP)
  4. CCI Reversals (ranging-market signal when ADX < 25)
  5. Multi-lookback Momentum Confirmation (63/126/252-day trend agreement)
  6. ALL combined

Baseline: GBTC | BTC-lead EMA(100) with V10 best params + on-chain.
"""

import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'intraday/strategies')

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
    # BTC-lead regime
    signal_ticker='BTC-USD', signal_ema_period=100,
)

START = '2015-01-01'
END   = '2025-12-31'

# ── Variant definitions ──────────────────────────────────────────────────────
VARIANTS = [
    ('V10 Baseline',          {}),
    ('+ Williams %R',         dict(enable_williams_r=True)),
    ('+ RSI Divergence',      dict(enable_divergence=True)),
    ('+ VWAP Z-score',        dict(enable_vwap_zscore=True)),
    ('+ CCI Reversals',       dict(enable_cci_mr=True)),
    ('+ Momentum Confirm',    dict(enable_momentum_confirm=True)),
    ('+ ALL Combined',        dict(
        enable_williams_r=True,
        enable_divergence=True,
        enable_vwap_zscore=True,
        enable_cci_mr=True,
        enable_momentum_confirm=True,
    )),
]


def run_variant(label, extra_params):
    """Run one backtest variant and return (label, metrics)."""
    params = {**BASE_PARAMS, **extra_params}
    strat = AggressiveHybridV6('GBTC', start=START, end=END, **params)
    ok = strat.fetch_data()
    if not ok:
        return label, {'error': 'fetch failed'}
    metrics = strat.backtest()
    return label, metrics


def main():
    print('=' * 90)
    print('V11 ALPHA SIGNAL COMPARISON — GBTC | BTC-lead EMA(100)')
    print('=' * 90)
    print()

    results = []
    for label, extra in VARIANTS:
        t0 = time.time()
        print(f'Running: {label:30s} ...', end=' ', flush=True)
        lbl, m = run_variant(label, extra)
        elapsed = time.time() - t0
        results.append((lbl, m))
        if 'error' in m:
            print(f'ERROR: {m["error"]}')
        else:
            print(f'Sharpe {m["sharpe"]:6.2f}  CAGR {m["cagr"]:6.1f}%  '
                  f'MaxDD {m["max_dd"]:5.1f}%  Trades {m["trades"]:4d}  '
                  f'WR {m["win_rate"]:5.1f}%  ({elapsed:.1f}s)')

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print('=' * 90)
    print(f'{"Variant":<30s} {"Sharpe":>7s} {"Sortino":>8s} {"CAGR":>7s} '
          f'{"MaxDD":>6s} {"Trades":>7s} {"WR":>6s} {"PF":>5s} {"Calmar":>7s}')
    print('-' * 90)

    baseline_sharpe = None
    for label, m in results:
        if 'error' in m:
            print(f'{label:<30s}  ERROR')
            continue
        if baseline_sharpe is None:
            baseline_sharpe = m['sharpe']
        delta = m['sharpe'] - baseline_sharpe
        delta_str = f'({"+" if delta >= 0 else ""}{delta:.2f})' if label != 'V10 Baseline' else ''
        print(f'{label:<30s} {m["sharpe"]:7.2f} {m["sortino"]:8.2f} {m["cagr"]:6.1f}% '
              f'{m["max_dd"]:5.1f}% {m["trades"]:7d} {m["win_rate"]:5.1f}% '
              f'{m["profit_factor"]:5.2f} {m["calmar"]:7.2f}  {delta_str}')

    print('=' * 90)
    print()

    # ── Best variant ──────────────────────────────────────────────────────────
    valid = [(l, m) for l, m in results if 'error' not in m]
    if valid:
        best = max(valid, key=lambda x: x[1]['sharpe'])
        print(f'Best by Sharpe: {best[0]}  (Sharpe={best[1]["sharpe"]:.2f}, '
              f'CAGR={best[1]["cagr"]:.1f}%, MaxDD={best[1]["max_dd"]:.1f}%)')

        # Show long/short breakdown for best
        bm = best[1]
        print(f'  Long trades: {bm["long_trades"]}  (WR {bm["long_wr"]:.1f}%)')
        print(f'  Short trades: {bm["short_trades"]}  (WR {bm["short_wr"]:.1f}%)')
        print(f'  Avg hold: {bm["avg_hold_days"]:.1f} days')
        print(f'  Best trade: ${bm["best_trade"]:,.0f}  |  Worst: ${bm["worst_trade"]:,.0f}')


if __name__ == '__main__':
    main()
