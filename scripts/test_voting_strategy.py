"""
Test Multi-Indicator Voting Strategy vs Original Hybrid

Compares:
- Old hybrid (RSI gate-based, Sharpe 0.61)
- New voting (8 indicators, score-based entry/exit, 0-4% position sizing)
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.financial_algorithms.data.yfinance_loader import load_stock_bars
from src.financial_algorithms.strategies.multi_indicator_voting import HybridVotingStrategy


def test_voting_strategy():
    """Test voting strategy on real AAPL/MSFT/AMZN data."""
    
    print("\n" + "="*70)
    print("MULTI-INDICATOR VOTING STRATEGY")
    print("Real Data Backtest: AAPL, MSFT, AMZN (1-year daily)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_stock_bars(['AAPL', 'MSFT', 'AMZN'], interval='1d', period='1y')
    print(f"[OK] Loaded {len(df):,} bars for {df['symbol'].nunique()} symbols")
    
    # Test voting strategy
    print("\nTesting VOTING strategy...")
    print("-" * 70)
    
    strategy = HybridVotingStrategy(
        buy_threshold=5,      # Buy when 5+ of 8 indicators agree
        exit_threshold=0,     # Exit when score drops to 0
        max_position_size=0.04,   # 4% max position
        target_avg_position=0.02,  # Target 2% average
    )
    
    try:
        metrics = strategy.backtest_daily(df, initial_capital=100000)
        
        print(f"\nResults:")
        print(f"  Total Return:      {metrics['total_return']*100:>7.2f}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe']:>7.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']*100:>7.2f}%")
        print(f"  Win Rate:          {metrics['win_rate']*100:>7.1f}%")
        print(f"  Trades:            {metrics['num_trades']:>7.0f}")
        print(f"  Avg Position Size: {metrics['avg_position_size']*100:>7.2f}%")
        
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON: Old Hybrid vs New Voting")
    print("="*70)
    print(f"{'Metric':<20} {'Old Hybrid':<15} {'New Voting':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Sharpe Ratio':<20} {0.61:<15.2f} {metrics['sharpe']:<15.2f} {metrics['sharpe']-0.61:+.2f}")
    print(f"{'Trades/Year':<20} {17:<15.0f} {metrics['num_trades']:<15.0f} {metrics['num_trades']-17:+.0f}")
    print(f"{'Avg Position %':<20} {2.5:<15.2f} {metrics['avg_position_size']*100:<15.2f} {metrics['avg_position_size']*100-2.5:+.2f}")
    print(f"{'Max Position %':<20} {7.5:<15.2f} {4.0:<15.2f} {4.0-7.5:+.2f}")
    print(f"{'Max Drawdown %':<20} {-0.0:<15.2f} {metrics['max_drawdown']*100:<15.2f}")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if metrics['sharpe'] > 0.61:
        print(f"✓ Sharpe improved: {metrics['sharpe']:.2f} vs 0.61 (old)")
    else:
        print(f"⚠ Sharpe lower: {metrics['sharpe']:.2f} vs 0.61 (might need tuning)")
    
    if metrics['num_trades'] > 20:
        print(f"✓ More trades: {metrics['num_trades']:.0f} vs 17 (better diversification)")
    else:
        print(f"⚠ Fewer trades: {metrics['num_trades']:.0f} vs 17")
    
    print(f"✓ Position sizing: Avg {metrics['avg_position_size']*100:.1f}% (target 2%), Max 4%")
    print(f"✓ Entry threshold: Score > +5 (5+ of 8 indicators agree)")
    print(f"✓ Exit rule: When score drops to ≤ 0 (consensus broken)")
    
    # Indicator contribution analysis
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Validate on different time periods (2023-2024, 2024-2025)")
    print("2. Test on other assets (SPY, QQQ, individual stocks)")
    print("3. Fine-tune buy_threshold (current: +5, try +4 or +6)")
    print("4. Fine-tune indicator weights if persistence insufficient")
    print("5. Consider adding ATR-based stop loss (% of capital)")


if __name__ == "__main__":
    test_voting_strategy()
