#!/usr/bin/env python
"""
Parameter sweep for Phase 6 system.
Tests different configurations one at a time to see impact on returns.

Tests:
1. Buy threshold sensitivity (+2, +2.5, +3, +3.5, +4)
2. Position sizing impact (2%, 4%, 6%)
3. Profit target adjustments (1.5%/3%, 2%/4%, 2.5%/5%)
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Results will be stored here
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def run_backtest_config(asset, buy_threshold, sell_threshold, tp1_pct, tp2_pct, position_size_min, output_suffix):
    """Run a single backtest with custom parameters."""
    
    config = {
        'asset': asset,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'tp1_pct': tp1_pct,
        'tp2_pct': tp2_pct,
        'position_size_min': position_size_min,
    }
    
    # Create a modified backtest runner with these parameters
    output_file = f'results_{asset}_{output_suffix}.json'
    
    # We'll modify the script inline for testing
    cmd = f"""python -c "
import sys
sys.path.insert(0, 'src')
import json
import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_enhanced_weighted import EnhancedWeightedVotingStrategy
from financial_algorithms.backtest.tiered_exits import TieredExitManager

# Load data
df = yf.download('{asset}', start='2023-01-01', end='2025-12-31', progress=False)
df = df.fillna(method='ffill')
print(f'Loaded {{len(df)}} bars for {asset}')

# Create strategy with custom params
strategy = EnhancedWeightedVotingStrategy(
    risk_pct=2.0,
    tp1_pct={tp1_pct},
    tp2_pct={tp2_pct},
    min_buy_score={buy_threshold},
    max_sell_score={sell_threshold},
)

exit_manager = TieredExitManager(
    risk_pct=2.0,
    tp1_pct={tp1_pct},
    tp2_pct={tp2_pct},
)

# Run backtest
equity = 100000
position = None
trades = []

for idx in range(50, len(df)):
    row = df.iloc[idx]
    date = df.index[idx]
    
    close_hist = df['close'].iloc[:idx+1].values
    high_hist = df['high'].iloc[:idx+1].values
    low_hist = df['low'].iloc[:idx+1].values
    vol_hist = df['volume'].iloc[:idx+1].values
    
    score = strategy.calculate_voting_score(close_hist, high_hist, low_hist, vol_hist)
    
    if position is None:
        if strategy.should_enter(score):
            entry_price = float(row['close'])
            # Dynamic sizing
            if score < {buy_threshold} + 1:
                risk_pct = {position_size_min}
            elif score < {buy_threshold} + 2:
                risk_pct = {position_size_min} + 2
            else:
                risk_pct = {position_size_min} + 4
            
            position_size = equity * risk_pct / 100
            quantity = position_size / entry_price
            
            position = exit_manager.create_trade('{asset}', entry_price, date, quantity)
            trades.append({{'entry_price': entry_price, 'score': score}})
    
    else:
        if exit_manager.check_exit_conditions(position, float(row['close']), date, score):
            exit_cond = exit_manager.check_exit_conditions(position, float(row['close']), date, score)
            pl = exit_cond.get('pl', 0)
            equity += pl
            position = None

# Calculate metrics
if trades:
    win_count = sum(1 for t in trades if t.get('return', 0) > 0)
    avg_return = np.mean([t.get('return', 0) for t in trades])
    sharpe = np.std([t.get('return', 0) for t in trades]) * np.sqrt(252) if np.std([t.get('return', 0) for t in trades]) > 0 else 0
else:
    win_count = 0
    avg_return = 0
    sharpe = 0

result = {{
    'asset': '{asset}',
    'config': {{
        'buy_threshold': {buy_threshold},
        'tp1_pct': {tp1_pct},
        'tp2_pct': {tp2_pct},
        'position_size_min': {position_size_min},
    }},
    'trades': len(trades),
    'equity': equity,
    'return': equity - 100000,
    'return_pct': (equity - 100000) / 100000 * 100,
}}

with open('{output_file}', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
"
    """
    
    os.system(cmd)
    return output_file

def test_buy_thresholds():
    """Test 1: Vary buy threshold"""
    print("\n" + "="*80)
    print("TEST 1: BUY THRESHOLD SENSITIVITY")
    print("="*80)
    
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for asset in ['AAPL']:  # Just test AAPL for now
        print(f"\nTesting {asset}:")
        print(f"{'Threshold':<12} {'Trades':<10} {'Return %':<12}")
        print("-" * 35)
        
        for thresh in thresholds:
            output = f'test_thresh_{thresh}.json'
            try:
                # Simplified inline test
                print(f"{thresh:<12} {'?':<10} {'?':<12} (placeholder)")
            except:
                pass

def test_position_sizing():
    """Test 2: Vary position sizing"""
    print("\n" + "="*80)
    print("TEST 2: POSITION SIZING IMPACT")
    print("="*80)
    
    sizes = ['2%', '4%', '6%', '8%']
    
    for asset in ['AAPL']:
        print(f"\nTesting {asset}:")
        print(f"{'Position Size':<15} {'Trades':<10} {'Return %':<12} {'Sharpe':<10}")
        print("-" * 50)
        
        for size in sizes:
            print(f"{size:<15} {'?':<10} {'?':<12} {'?':<10} (placeholder)")

def test_profit_targets():
    """Test 3: Vary TP1/TP2"""
    print("\n" + "="*80)
    print("TEST 3: PROFIT TARGET ADJUSTMENTS")
    print("="*80)
    
    configs = [
        (1.5, 3.0, "Conservative (1.5%/3%)"),
        (2.0, 4.0, "Moderate (2%/4%)"),
        (2.5, 5.0, "Aggressive (2.5%/5%)"),
    ]
    
    for asset in ['AAPL']:
        print(f"\nTesting {asset}:")
        print(f"{'Config':<30} {'Trades':<10} {'Return %':<12} {'Sharpe':<10}")
        print("-" * 65)
        
        for tp1, tp2, label in configs:
            print(f"{label:<30} {'?':<10} {'?':<12} {'?':<10} (placeholder)")

if __name__ == '__main__':
    print("PHASE 6 PARAMETER SWEEP")
    print(f"Start Time: {datetime.now()}")
    
    test_buy_thresholds()
    test_position_sizing()
    test_profit_targets()
    
    print("\n" + "="*80)
    print("Parameter sweep complete")
