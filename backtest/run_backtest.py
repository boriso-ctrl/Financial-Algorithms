"""
Example backtest runner: SMA crossover strategy on 3 tickers.

This script demonstrates:
1. Loading data
2. Generating signals
3. Running backtest
4. Displaying results
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_daily_prices
from signals.sma_signal import sma_signal
from backtest.simple_backtest import run_backtest
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("SMA Crossover Backtest")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/4] Loading price data...")
    tickers = ['AAPL', 'MSFT', 'AMZN']
    prices = load_daily_prices(tickers)
    
    # Filter to last 3 years for faster testing
    prices = prices.last('3Y')
    print(f"  Loaded {len(prices)} days for {len(tickers)} tickers")
    print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # 2. Generate signals
    print("\n[2/4] Generating SMA crossover signals...")
    fast_period = 20
    slow_period = 50
    signals = sma_signal(prices, fast=fast_period, slow=slow_period)
    print(f"  SMA periods: fast={fast_period}, slow={slow_period}")
    
    # Drop NaN rows (before slow SMA is computed)
    valid_idx = signals.notna().all(axis=1)
    prices = prices[valid_idx]
    signals = signals[valid_idx]
    print(f"  Valid signals: {len(signals)} days")
    
    # 3. Run backtest
    print("\n[3/4] Running backtest...")
    initial_capital = 100000
    results = run_backtest(prices, signals, initial_capital=initial_capital)
    
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
    ax1.set_title(f"SMA({fast_period}/{slow_period}) Backtest: Equity Curve", fontsize=14)
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
    plt.savefig('backtest_results.png', dpi=150)
    print(f"✓ Plot saved to: backtest_results.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
