#!/usr/bin/env python
"""
Aggregate results from parallel backtests.

Usage:
    python scripts/aggregate_results.py \
        --files results_aapl.json results_spy.json results_qqq.json results_tsla.json \
        --output FINAL_RESULTS.json

Creates comparison table and summary statistics.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(file_path: str) -> Dict:
    """Load JSON results file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def aggregate_results(files: List[str]) -> Dict:
    """
    Aggregate results from multiple backtest files.
    
    Returns:
        Dict with comparison table and summary stats
    """
    all_results = []
    
    for file_path in files:
        if not Path(file_path).exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            result = load_results(file_path)
            all_results.append(result)
            logger.info(f"Loaded {file_path}: {result['asset']}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No valid results files found")
    
    # Create comparison table
    comparison = []
    for result in all_results:
        comparison.append({
            'asset': result['asset'],
            'sharpe': round(result['sharpe_ratio'], 2),
            'return_pct': round(result['total_return_pct'], 2),
            'trades': result['trade_count'],
            'win_rate': round(result['win_rate_pct'], 1),
            'best_trade': round(result['best_trade_pct'], 2),
            'worst_trade': round(result['worst_trade_pct'], 2),
            'final_equity': round(result['final_equity'], 2),
        })
    
    # Calculate cross-asset statistics
    sharpes = [r['sharpe_ratio'] for r in all_results]
    returns = [r['total_return_pct'] for r in all_results]
    trades = [r['trade_count'] for r in all_results]
    win_rates = [r['win_rate_pct'] for r in all_results]
    
    aggregate = {
        'timestamp': datetime.now().isoformat(),
        'asset_count': len(all_results),
        'comparison_table': comparison,
        'cross_asset_statistics': {
            'avg_sharpe': round(sum(sharpes) / len(sharpes), 2) if sharpes else 0,
            'median_sharpe': round(sorted(sharpes)[len(sharpes)//2], 2) if sharpes else 0,
            'best_sharpe': round(max(sharpes), 2) if sharpes else 0,
            'worst_sharpe': round(min(sharpes), 2) if sharpes else 0,
            'avg_return_pct': round(sum(returns) / len(returns), 2) if returns else 0,
            'total_trades': sum(trades),
            'avg_win_rate': round(sum(win_rates) / len(win_rates), 1) if win_rates else 0,
        },
        'assets_passing_criteria': [
            r['asset'] for r in all_results 
            if r['sharpe_ratio'] >= 0.5 and r['win_rate_pct'] >= 50
        ],
    }
    
    return aggregate


def print_summary_table(aggregate: Dict):
    """Print formatted summary table."""
    print("\n" + "="*90)
    print("PHASE 6 BACKTEST RESULTS - ENHANCED WEIGHTED VOTING SYSTEM")
    print("="*90)
    
    # Comparison table
    print("\n{:<8} {:<10} {:<12} {:<8} {:<10} {:<12} {:<12}".format(
        "ASSET", "SHARPE", "RETURN %", "TRADES", "WIN RATE", "BEST %", "WORST %"
    ))
    print("-"*90)
    
    for row in aggregate['comparison_table']:
        print("{:<8} {:<10.2f} {:<12.2f} {:<8} {:<10.1f} {:<12.2f} {:<12.2f}".format(
            row['asset'],
            row['sharpe'],
            row['return_pct'],
            row['trades'],
            row['win_rate'],
            row['best_trade'],
            row['worst_trade'],
        ))
    
    # Summary stats
    stats = aggregate['cross_asset_statistics']
    print("\n" + "-"*90)
    print("CROSS-ASSET SUMMARY")
    print("-"*90)
    print(f"  Average Sharpe:      {stats['avg_sharpe']:.2f}")
    print(f"  Best Sharpe:         {stats['best_sharpe']:.2f}")
    print(f"  Worst Sharpe:        {stats['worst_sharpe']:.2f}")
    print(f"  Average Return:      {stats['avg_return_pct']:.2f}%")
    print(f"  Total Trades:        {stats['total_trades']}")
    print(f"  Average Win Rate:    {stats['avg_win_rate']:.1f}%")
    
    # Criteria check
    passing = aggregate['assets_passing_criteria']
    print(f"\n  Assets Meeting Criteria (Sharpe ≥ 0.5 & Win Rate ≥ 50%): {len(passing)}/4")
    if passing:
        print(f"    ✓ {', '.join(passing)}")
    else:
        print(f"    ✗ None passing yet - needs refinement")
    
    print("\n" + "="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Aggregate parallel backtest results')
    parser.add_argument('--files', nargs='+', required=True, help='Result JSON files')
    parser.add_argument('--output', default='FINAL_RESULTS.json', help='Output file')
    
    args = parser.parse_args()
    
    logger.info(f"Aggregating {len(args.files)} result files")
    aggregate = aggregate_results(args.files)
    
    # Save JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    logger.info(f"Saved aggregated results to {output_path}")
    
    # Print summary
    print_summary_table(aggregate)
    
    return aggregate


if __name__ == '__main__':
    main()
