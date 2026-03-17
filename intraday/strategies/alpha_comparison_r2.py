"""
V11 Alpha Round 2 — Momentum-based alpha extraction
=====================================================
Round 1 showed Momentum Confirm as the best individual contributor:
  CAGR +0.5%, MaxDD -0.1%, PF +0.20, Sortino +0.06 vs baseline

This round tests more aggressive momentum-based ideas:
  1. Momentum Confirm (Round 1 winner, baseline for Round 2)
  2. Momentum FILTER: suppress longs when Mom_Score < 0 (don't enter fading trends)
  3. Momentum + Divergence combo
  4. Momentum risk scaling: scale risk by Mom_Score magnitude
  5. Short filter: only short when Mom_Score < -0.33
  6. Kitchen sink: best combo from above
"""

import sys, warnings, time, copy
warnings.filterwarnings('ignore')
sys.path.insert(0, 'intraday/strategies')

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
    signal_ticker='BTC-USD', signal_ema_period=100,
)

START = '2015-01-01'
END   = '2025-12-31'


class MomentumFilterV11(AggressiveHybridV6):
    """Extends V6 with momentum-as-filter: suppress longs when Mom_Score < threshold."""

    def __init__(self, *args, mom_long_filter=0.0, mom_short_filter=-0.33,
                 mom_risk_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mom_long_filter  = mom_long_filter   # suppress longs if Mom_Score < this
        self.mom_short_filter = mom_short_filter   # suppress shorts if Mom_Score > this
        self.mom_risk_scale   = mom_risk_scale     # scale risk by abs(Mom_Score)

    def backtest(self):
        """Patched backtest that injects momentum filter into entry conditions."""
        # Run prepare_indicators to get Mom_Score column
        self.prepare_indicators()

        # Extract Mom_Score as numpy array for fast access
        _mom_score = self.data['Mom_Score'].to_numpy(dtype=float)

        # Save original backtest methods
        orig_backtest = AggressiveHybridV6.backtest

        # We need to patch the backtest loop. The cleanest way is to
        # override generate_buy_signals and generate_sell_signals to add
        # momentum gating, plus optionally scale risk.

        _orig_buy  = self._gen_buy_orig
        _orig_sell = self._gen_sell_orig

        # Actually, let's just do a direct backtest with the patches inline.
        # The backtest() method calls prepare_indicators() again, so we need
        # to be smart about this. Let's override the signal generators.
        return super().backtest()

    @property
    def _gen_buy_orig(self):
        return super().generate_buy_signals

    @property
    def _gen_sell_orig(self):
        return super().generate_sell_signals

    def generate_buy_signals(self, idx):
        # Momentum gate: suppress longs in fading momentum
        mom = self.data.iloc[idx]['Mom_Score']
        if mom < self.mom_long_filter:
            return [], 0.0, 'trend'
        signals, strength, sig_type = super().generate_buy_signals(idx)
        # Risk scaling: boost strength when momentum strongly positive
        if self.mom_risk_scale and mom > 0.66:
            strength *= 1.15
        return signals, strength, sig_type

    def generate_sell_signals(self, idx):
        # Momentum gate: suppress shorts unless momentum is negative
        mom = self.data.iloc[idx]['Mom_Score']
        if mom > self.mom_short_filter:
            return [], 0.0, 'trend'
        signals, strength, sig_type = super().generate_sell_signals(idx)
        if self.mom_risk_scale and mom < -0.66:
            strength *= 1.15
        return signals, strength, sig_type


VARIANTS = [
    ('V10 Baseline',            AggressiveHybridV6, {}),
    ('R1: + Momentum Confirm',  AggressiveHybridV6, dict(enable_momentum_confirm=True)),
    ('R2a: Mom Filter (>0)',    MomentumFilterV11,  dict(
        enable_momentum_confirm=True, mom_long_filter=0.0)),
    ('R2b: Mom Filter (>-0.33)', MomentumFilterV11, dict(
        enable_momentum_confirm=True, mom_long_filter=-0.33)),
    ('R2c: Mom + Divergence',   AggressiveHybridV6, dict(
        enable_momentum_confirm=True, enable_divergence=True)),
    ('R2d: Mom + Risk Scale',   MomentumFilterV11,  dict(
        enable_momentum_confirm=True, mom_risk_scale=True, mom_long_filter=-999)),
    ('R2e: Mom Short Filter',   MomentumFilterV11,  dict(
        enable_momentum_confirm=True, mom_short_filter=0.0)),
    ('R2f: Mom Filter + Div + Scale', MomentumFilterV11, dict(
        enable_momentum_confirm=True, enable_divergence=True,
        mom_risk_scale=True, mom_long_filter=0.0, mom_short_filter=0.0)),
]


def run_variant(label, klass, extra_params):
    params = {**BASE_PARAMS, **extra_params}
    strat = klass('GBTC', start=START, end=END, **params)
    ok = strat.fetch_data()
    if not ok:
        return label, {'error': 'fetch failed'}
    metrics = strat.backtest()
    return label, metrics


def main():
    print('=' * 95)
    print('V11 ALPHA ROUND 2 — MOMENTUM-BASED ALPHA EXTRACTION — GBTC | BTC-lead EMA(100)')
    print('=' * 95)
    print()

    results = []
    for label, klass, extra in VARIANTS:
        t0 = time.time()
        print(f'Running: {label:35s} ...', end=' ', flush=True)
        lbl, m = run_variant(label, klass, extra)
        elapsed = time.time() - t0
        results.append((lbl, m))
        if 'error' in m:
            print(f'ERROR: {m["error"]}')
        else:
            print(f'Sharpe {m["sharpe"]:6.2f}  CAGR {m["cagr"]:6.1f}%  '
                  f'MaxDD {m["max_dd"]:5.1f}%  Trades {m["trades"]:4d}  '
                  f'WR {m["win_rate"]:5.1f}%  PF {m["profit_factor"]:.2f}  ({elapsed:.1f}s)')

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 95)
    print(f'{"Variant":<35s} {"Sharpe":>7s} {"Sortino":>8s} {"CAGR":>7s} '
          f'{"MaxDD":>6s} {"Trades":>7s} {"WR":>6s} {"PF":>5s} {"Calmar":>7s}')
    print('-' * 95)

    baseline = None
    for label, m in results:
        if 'error' in m:
            print(f'{label:<35s}  ERROR')
            continue
        if baseline is None:
            baseline = m
        ds = m['sharpe'] - baseline['sharpe']
        dc = m['cagr'] - baseline['cagr']
        delta = f'(S{"+" if ds>=0 else ""}{ds:.2f} C{"+" if dc>=0 else ""}{dc:.1f}%)'
        if label == 'V10 Baseline':
            delta = ''
        print(f'{label:<35s} {m["sharpe"]:7.2f} {m["sortino"]:8.2f} {m["cagr"]:6.1f}% '
              f'{m["max_dd"]:5.1f}% {m["trades"]:7d} {m["win_rate"]:5.1f}% '
              f'{m["profit_factor"]:5.2f} {m["calmar"]:7.2f}  {delta}')

    print('=' * 95)

    valid = [(l, m) for l, m in results if 'error' not in m]
    if valid:
        best_sharpe  = max(valid, key=lambda x: x[1]['sharpe'])
        best_calmar  = max(valid, key=lambda x: x[1]['calmar'])
        best_sortino = max(valid, key=lambda x: x[1]['sortino'])
        print(f'\nBest Sharpe:  {best_sharpe[0]}  ({best_sharpe[1]["sharpe"]:.2f})')
        print(f'Best Calmar:  {best_calmar[0]}  ({best_calmar[1]["calmar"]:.2f})')
        print(f'Best Sortino: {best_sortino[0]}  ({best_sortino[1]["sortino"]:.2f})')

        bm = best_sharpe[1]
        print(f'\nBest details: Sharpe={bm["sharpe"]:.2f}, CAGR={bm["cagr"]:.1f}%, '
              f'MaxDD={bm["max_dd"]:.1f}%, WR={bm["win_rate"]:.1f}%')
        print(f'  Longs: {bm["long_trades"]} (WR {bm["long_wr"]:.1f}%)  '
              f'Shorts: {bm["short_trades"]} (WR {bm["short_wr"]:.1f}%)  '
              f'Hold: {bm["avg_hold_days"]:.1f}d')


if __name__ == '__main__':
    main()
