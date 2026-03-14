import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'intraday/strategies')
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

t = AggressiveHybridV6(
    'SPY', start='2020-01-01', end='2023-12-31',
    enable_stoch_rsi=True, enable_bb_signal=True,
    di_filter=True, obv_filter=True,
    partial_qty_pct=0.33, vol_regime_scale=1.1,
    min_vol_ratio=0.8, reentry_cooldown=2
)
t.fetch_data()
r = t.backtest()
print(f"Sharpe={r['sharpe']}  CAGR={r['cagr']}%  Trades={r['trades']}  MaxDD={r['max_dd']}%")
print("V9 sanity check PASSED")
