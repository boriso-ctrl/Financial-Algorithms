"""Deprecated: moved to scripts/demo_blend.py.

Left as a thin shim for backwards compatibility. Prefer running:
    python scripts/demo_blend.py --tickers AAPL MSFT AMZN --years 3 --report-prefix demo
"""

from scripts.demo_blend import build_price_signal, build_volume_signal, main
"""Example: blend price and volume signals and run backtest.

This script shows how to:
- load prices
- build two signals (price MA-cross, volume VWMA spread)
- blend with weights
- run backtest with frictions and export a report

Usage:
    python scripts/demo_blend.py --tickers AAPL MSFT AMZN --years 3 --report-prefix demo
"""

import os
import sys

# Add repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import pandas as pd

from data_loader import load_daily_prices
from backtest.simple_backtest import run_backtest
from backtest.signal_blender import blend_signals
from indicators.price import ma_cross_strategy
from indicators.volume import vwma_signal


def build_price_signal(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    # Ensure OHLC structure
    if 'Close' not in prices.columns:
        prices = prices.copy()
        prices['Close'] = prices.iloc[:, 0]
    out = ma_cross_strategy(prices.copy(), sl=0.0, n1=fast, n2=slow)
    sig = out['MA_signal'] if 'MA_signal' in out.columns else out
    if sig.ndim == 1:
        sig = sig.to_frame(name='price')
    # Broadcast to all tickers if only one column present
    if sig.shape[1] == 1 and prices.shape[1] > 1:
        sig = pd.concat([sig.iloc[:, 0]] * prices.shape[1], axis=1)
        sig.columns = prices.columns
    return sig


def build_volume_signal(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    # Use close and a proxy volume (if volume not available, fallback zeros)
    close = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
    volume = prices.get('Volume') if 'Volume' in prices.columns else pd.Series(1_000_000, index=prices.index)
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    sig = vwma_signal(close, volume, fast=fast, slow=slow)
    if sig.ndim == 1:
        sig = sig.to_frame(name='volume')
    if sig.shape[1] == 1 and prices.shape[1] > 1:
        sig = pd.concat([sig.iloc[:, 0]] * prices.shape[1], axis=1)
        sig.columns = prices.columns
    return sig


def main():
    parser = argparse.ArgumentParser(description="Demo blended backtest (price + volume)")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'AMZN'], help="Tickers")
    parser.add_argument('--years', type=int, default=3, help="Lookback years")
    parser.add_argument('--fast', type=int, default=20, help="Fast MA/VWMA")
    parser.add_argument('--slow', type=int, default=50, help="Slow MA/VWMA")
    parser.add_argument('--price-weight', type=float, default=2.0, help="Weight for price signal")
    parser.add_argument('--volume-weight', type=float, default=1.0, help="Weight for volume signal")
    parser.add_argument('--capital', type=float, default=100000.0, help="Starting capital")
    parser.add_argument('--cost-bps', type=float, default=1.0)
    parser.add_argument('--slippage-bps', type=float, default=5.0)
    parser.add_argument('--allow-short', action='store_true')
    parser.add_argument('--max-signal-abs', type=float, default=5.0)
    parser.add_argument('--report-prefix', type=str, default='demo')
    args = parser.parse_args()

    prices = load_daily_prices(args.tickers)
    prices = prices.last(f"{args.years}Y") if args.years else prices

    price_sig = build_price_signal(prices, fast=args.fast, slow=args.slow)
    vol_sig = build_volume_signal(prices, fast=args.fast, slow=args.slow)

    blended = blend_signals(
        {'price': price_sig, 'volume': vol_sig},
        {'price': args.price_weight, 'volume': args.volume_weight},
        max_signal_abs=args.max_signal_abs,
    )

    results = run_backtest(
        prices,
        blended,
        initial_capital=args.capital,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        allow_short=args.allow_short,
        max_signal_abs=args.max_signal_abs,
    )

    print("=== Blended Backtest Metrics ===")
    for k, v in results['metrics'].items():
        print(f"{k:<20} {v}")

    # Save report
    metrics_path = f"{args.report_prefix}_metrics.json"
    equity_path = f"{args.report_prefix}_equity.csv"
    results['equity_curve'].to_csv(equity_path, header=True)
    import json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"Saved metrics to {metrics_path} and equity to {equity_path}")


if __name__ == "__main__":
    main()
