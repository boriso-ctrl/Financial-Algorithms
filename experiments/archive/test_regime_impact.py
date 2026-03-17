"""Test: Compare hybrid WITH and WITHOUT regime filters."""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.financial_algorithms.data.yfinance_loader import load_stock_bars
from src.financial_algorithms.strategies.hybrid_phase3_phase6 import HybridPhase3Phase6


def test_regime_filter_impact():
    """Compare backtest WITH and WITHOUT regime filtering."""
    
    print("\n" + "="*70)
    print("TEST: Regime Filter Impact on Trade Count")
    print("="*70)
    
    # Load data
    print("\nLoading AAPL, MSFT, AMZN (1 year daily)...")
    df = load_stock_bars(['AAPL', 'MSFT', 'AMZN'], interval='1d', period='1y')
    print(f"[OK] Loaded {len(df):,} bars\n")
    
    # Scenario 1: Tight regime (should filter more)
    print("Scenario 1: TIGHT regime (RSI 45-55, Volume 0.95)")
    print("-" * 70)
    hybrid_tight = HybridPhase3Phase6(
        rsi_threshold_oversold=45,
        rsi_threshold_overbought=55,
        min_volume_ma_ratio=0.95,
    )
    metrics_tight = hybrid_tight.backtest_daily(df, initial_capital=100000)
    print(f"  Trades: {metrics_tight['num_trades']}")
    print(f"  Sharpe: {metrics_tight['sharpe']:.2f}")
    
    # Scenario 2: Loose regime (should filter less)
    print("\nScenario 2: LOOSE regime (RSI 10-90, Volume 0.3)")
    print("-" * 70)
    hybrid_loose = HybridPhase3Phase6(
        rsi_threshold_oversold=10,
        rsi_threshold_overbought=90,
        min_volume_ma_ratio=0.3,
    )
    metrics_loose = hybrid_loose.backtest_daily(df, initial_capital=100000)
    print(f"  Trades: {metrics_loose['num_trades']}")
    print(f"  Sharpe: {metrics_loose['sharpe']:.2f}")
    
    # Scenario 3: Super loose (almost no regime filtering)
    print("\nScenario 3: SUPER LOOSE regime (RSI 0-100, Volume 0.1)")
    print("-" * 70)
    hybrid_super_loose = HybridPhase3Phase6(
        rsi_threshold_oversold=0,
        rsi_threshold_overbought=100,
        min_volume_ma_ratio=0.1,
    )
    metrics_super = hybrid_super_loose.backtest_daily(df, initial_capital=100000)
    print(f"  Trades: {metrics_super['num_trades']}")
    print(f"  Sharpe: {metrics_super['sharpe']:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if metrics_tight['num_trades'] == metrics_loose['num_trades'] == metrics_super['num_trades']:
        print("[ERROR] ALL scenarios have identical trade counts!")
        print("        This means regime filtering has NO EFFECT on backtest.")
        print("")
        print("Diagnosis: The base Phase 3 signal is the bottleneck,")
        print("           not the regime filter. The regime filter is")
        print("           applied to days that already have signals.")
        print("")
        print("Solution: Rethink the strategy architecture.")
        print("          Consider:")
        print("          - Using entry signals without regime filter")
        print("          - Using regime filter for EXIT only")
        print("          - Expanding base signal generation")
        print("          - Rolling-based position tracking instead of")
        print("            single-position constraint")
    else:
        print("[OK] Regime filtering IS affecting backtest")
        print(f"     Difference: {metrics_tight['num_trades']} vs {metrics_loose['num_trades']} trades")


if __name__ == "__main__":
    test_regime_filter_impact()
