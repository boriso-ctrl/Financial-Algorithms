"""
Backtest runner with pluggable strategies.

This script demonstrates:
1. Loading data
2. Selecting an indicator/strategy by name
3. Generating signals with params
4. Running backtest
5. Displaying results
"""

import sys
import os
import argparse
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_daily_prices
from backtest.simple_backtest import run_backtest
from backtest.strategy_registry import registry
import matplotlib.pyplot as plt
import json


def _summarize_prices(prices) -> None:
    print(f"  Data shape: {prices.shape[0]} rows x {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    missing = prices.isna().sum().sum()
    if missing:
        pct = missing / prices.size * 100
        print(f"  Missing values: {missing} ({pct:.2f}% of matrix) [filled forward/backward in engine]")
    dupes = prices.index.duplicated().sum()
    if dupes:
        print(f"  Warning: {dupes} duplicate dates in index; engine will raise if not deduped")


def main():
    args = parse_args()

    print("=" * 60)
    print("Strategy Backtest")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/4] Loading price data...")
    prices = load_daily_prices(args.tickers)
    prices = prices.last(f"{args.years}Y") if args.years else prices
    print(f"  Loaded {len(prices)} days for {len(args.tickers)} tickers")
    _summarize_prices(prices)
    
    # 2. Generate signals
    strat = registry.get(args.strategy)
    if strat is None:
        available = ', '.join(registry.list().keys())
        raise SystemExit(f"Unknown strategy '{args.strategy}'. Available: {available}")

    print(f"\n[2/4] Generating signals with strategy '{args.strategy}'...")
    params = strat['params'].copy()
    # Override with CLI params
    if args.fast:
        params['fast'] = args.fast
    if args.slow:
        params['slow'] = args.slow
    if args.sl is not None:
        params['sl'] = args.sl

    signals = strat['func'](prices, **params)
    # If strategy returns enriched df, extract signal column when possible
    if isinstance(signals, (list, tuple)):
        raise SystemExit("Strategy returned unsupported type; expected DataFrame or Series.")
    if hasattr(signals, 'columns') and signals.shape[1] >= 1 and not signals.equals(prices):
        # keep only columns that look like signals if any; otherwise preserve as-is
        signal_cols = [c for c in signals.columns if 'signal' in c.lower()]
        if signal_cols:
            signals = signals[signal_cols]

    # Ensure signals shape matches prices
    signals = signals.reindex(prices.index).fillna(0)
    print(f"  Signals generated: {len(signals)} rows")
    
    # 3. Run backtest
    print("\n[3/4] Running backtest...")
    initial_capital = args.capital
    results = run_backtest(
        prices,
        signals,
        initial_capital=initial_capital,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        allow_short=args.allow_short,
        risk_free_rate=args.risk_free_rate,
        max_gross_leverage=args.max_gross_leverage,
        max_position_weight=args.max_position_weight,
        max_signal_abs=args.max_signal_abs,
        strict_data=args.strict_data,
    )
    
    # 4. Display results
    print("\n[4/4] Results")
    print("=" * 60)
    print(f"Initial Capital: ${initial_capital:,.2f}\n")
    
    for key, value in results['metrics'].items():
        print(f"{key:.<25} {value}")
    
    # Plot equity curve
    print("\n" + "=" * 60)
    print("Generating equity curve plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Equity curve
    results['equity_curve'].plot(ax=ax1, linewidth=2)
    ax1.set_title(f"{args.strategy} Backtest: Equity Curve", fontsize=14)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(alpha=0.3)
    ax1.axhline(initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.legend()
    
    # Number of positions over time
    results['num_positions'].plot(ax=ax2, linewidth=1, alpha=0.7)
    ax2.set_title("Number of Active Positions", fontsize=14)
    ax2.set_ylabel("# Positions")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{args.report_prefix}_equity.png" if args.report_prefix else "backtest_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Plot saved to: {plot_path}")
    plt.show()

    # Persist metrics/equity if requested
    if args.report_prefix:
        metrics_path = f"{args.report_prefix}_metrics.json"
        equity_path = f"{args.report_prefix}_equity.csv"
        results['equity_curve'].to_csv(equity_path, header=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"✓ Metrics saved to: {metrics_path}")
        print(f"✓ Equity curve saved to: {equity_path}")
    
    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple strategy backtest.")
    parser.add_argument('--strategy', default='sma', help="Strategy name (see registry)")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'AMZN'], help="List of tickers")
    parser.add_argument('--years', type=int, default=3, help="Limit to last N years (0 = all)")
    parser.add_argument('--capital', type=float, default=100000.0, help="Starting capital")
    parser.add_argument('--fast', type=int, help="Fast period (for SMA-like strategies)")
    parser.add_argument('--slow', type=int, help="Slow period (for SMA-like strategies)")
    parser.add_argument('--sl', type=float, help="Stop-loss/threshold param where applicable")
    parser.add_argument('--cost-bps', type=float, default=1.0, help="Commission/fees per turnover in basis points")
    parser.add_argument('--slippage-bps', type=float, default=5.0, help="Slippage per turnover in basis points")
    parser.add_argument('--allow-short', action='store_true', help="Enable short signals (-1) when provided")
    parser.add_argument('--risk-free-rate', type=float, default=0.0, help="Annual risk-free rate (e.g., 0.05 for 5%)")
    parser.add_argument('--max-gross-leverage', type=float, default=1.0, help="Cap on sum of absolute weights")
    parser.add_argument('--max-position-weight', type=float, default=1.0, help="Cap on absolute weight per position")
    parser.add_argument('--max-signal-abs', type=float, default=5.0, help="Clip raw signal magnitudes to this abs value")
    parser.add_argument('--strict-data', action='store_true', help="Raise on duplicate dates or NaNs after fill")
    parser.add_argument('--report-prefix', type=str, help="If set, saves metrics (json), equity (csv), and plot with this prefix")
    return parser.parse_args()


if __name__ == "__main__":
    main()
