"""
Automated parameter search for Hybrid Phase 3 + Phase 6 strategy.

Tests combinations of RSI and volume thresholds to find optimal settings.
Objective: Maximize Sharpe while maintaining > 80 trades/year and < 15% max drawdown.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.financial_algorithms.data.yfinance_loader import load_stock_bars
from src.financial_algorithms.strategies.hybrid_phase3_phase6 import HybridPhase3Phase6


def run_parameter_search(
    df: pd.DataFrame,
    rsi_oversold_values: list = None,
    rsi_overbought_values: list = None,
    volume_ratio_values: list = None,
    initial_capital: float = 100000,
) -> dict:
    """
    Grid search for optimal hybrid strategy parameters.
    
    Args:
        df: OHLCV DataFrame with symbol column
        rsi_oversold_values: List of RSI oversold thresholds to test
        rsi_overbought_values: List of RSI overbought thresholds to test
        volume_ratio_values: List of volume ratio thresholds to test
        initial_capital: Capital for backtest
    
    Returns:
        Dict with all results and best configuration
    """
    if rsi_oversold_values is None:
        rsi_oversold_values = [25, 30, 35, 40, 45]
    if rsi_overbought_values is None:
        rsi_overbought_values = [60, 65, 70, 75, 80]
    if volume_ratio_values is None:
        volume_ratio_values = [0.6, 0.7, 0.8, 0.9]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'rsi_oversold': rsi_oversold_values,
            'rsi_overbought': rsi_overbought_values,
            'volume_ratio': volume_ratio_values,
        },
        'search_results': [],
        'metrics_summary': {},
    }
    
    num_combos = len(rsi_oversold_values) * len(rsi_overbought_values) * len(volume_ratio_values)
    print(f"\n{'='*70}")
    print(f"HYBRID STRATEGY PARAMETER SEARCH")
    print(f"{'='*70}")
    print(f"Testing {num_combos} parameter combinations...")
    print(f"Data: {len(df):,} bars, {df['symbol'].nunique()} symbols")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    combo_idx = 0
    
    for rsi_os, rsi_ob, vol_ratio in product(rsi_oversold_values, rsi_overbought_values, volume_ratio_values):
        combo_idx += 1
        
        # Skip invalid combinations
        if rsi_os >= rsi_ob:
            continue
        
        try:
            # Run backtest with these parameters
            hybrid = HybridPhase3Phase6(
                rsi_threshold_oversold=rsi_os,
                rsi_threshold_overbought=rsi_ob,
                min_volume_ma_ratio=vol_ratio,
                volatility_target=0.02,
            )
            
            metrics = hybrid.backtest_daily(df, initial_capital=initial_capital)
            
            # Store result
            result = {
                'rsi_oversold': rsi_os,
                'rsi_overbought': rsi_ob,
                'volume_ratio': vol_ratio,
                'sharpe': metrics['sharpe'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'num_trades': metrics['num_trades'],
            }
            
            results['search_results'].append(result)
            
            # Print progress
            status = "[OK]" if metrics['num_trades'] > 80 else "[LOW_TRADES]"
            print(f"[{combo_idx:3d}] RSI {rsi_os:2d}-{rsi_ob:2d}, Vol {vol_ratio:.1f} "
                  f"-> Sharpe {metrics['sharpe']:6.2f}, "
                  f"Trades {metrics['num_trades']:3d}, "
                  f"DD {metrics['max_drawdown']:6.1%} {status}")
        
        except Exception as e:
            print(f"[{combo_idx:3d}] RSI {rsi_os:2d}-{rsi_ob:2d}, Vol {vol_ratio:.1f} -> ERROR: {e}")
            continue
    
    # Analyze results
    if results['search_results']:
        df_results = pd.DataFrame(results['search_results'])
        
        # Calculate summary metrics
        results['metrics_summary'] = {
            'best_sharpe': float(df_results['sharpe'].max()),
            'best_sharpe_config': df_results.loc[df_results['sharpe'].idxmax()].to_dict(),
            'mean_sharpe': float(df_results['sharpe'].mean()),
            'std_sharpe': float(df_results['sharpe'].std()),
        }
        
        # Find configurations that meet criteria
        # Criteria: >80 trades, <-15% DD, Sharpe > 0.5
        valid_configs = df_results[
            (df_results['num_trades'] >= 80) &
            (df_results['max_drawdown'] >= -0.15) &
            (df_results['sharpe'] > 0.5)
        ].sort_values('sharpe', ascending=False)
        
        results['valid_configs'] = valid_configs.head(10).to_dict('records')
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total configurations tested: {len(df_results)}")
        print(f"Best Sharpe: {results['metrics_summary']['best_sharpe']:.2f}")
        print(f"Mean Sharpe: {results['metrics_summary']['mean_sharpe']:.2f}")
        print(f"Valid configs (>80 trades, <15% DD, S>0.5): {len(valid_configs)}")
        
        if len(valid_configs) > 0:
            print(f"\n{'TOP 5 CONFIGURATIONS':^70}")
            print("-" * 70)
            for i, row in valid_configs.head(5).iterrows():
                print(f"RSI {row['rsi_oversold']:2.0f}-{row['rsi_overbought']:2.0f}, "
                      f"Vol {row['volume_ratio']:.1f} "
                      f"-> Sharpe {row['sharpe']:6.2f}, "
                      f"Trades {row['num_trades']:3.0f}, "
                      f"DD {row['max_drawdown']:6.1%}, "
                      f"Return {row['total_return']:6.1%}")
            
            # Get best overall
            best = valid_configs.iloc[0]
            results['recommended_config'] = {
                'rsi_oversold': float(best['rsi_oversold']),
                'rsi_overbought': float(best['rsi_overbought']),
                'volume_ratio': float(best['volume_ratio']),
                'expected_sharpe': float(best['sharpe']),
                'expected_trades': int(best['num_trades']),
                'expected_max_drawdown': float(best['max_drawdown']),
                'expected_return': float(best['total_return']),
            }
            
            print(f"\n{'RECOMMENDED CONFIGURATION':^70}")
            print("-" * 70)
            print(f"RSI Oversold: {best['rsi_oversold']:.0f}")
            print(f"RSI Overbought: {best['rsi_overbought']:.0f}")
            print(f"Min Volume Ratio: {best['volume_ratio']:.1f}")
            print(f"\nExpected Performance:")
            print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
            print(f"  Annual Return: {best['total_return']*100:.1f}%")
            print(f"  Max Drawdown: {best['max_drawdown']*100:.1f}%")
            print(f"  Trades/Year: {best['num_trades']:.0f}")
            print(f"  Win Rate: {best['win_rate']*100:.1f}%")
        else:
            print("\n[WARNING] No configurations meet all criteria.")
            print("Consider relaxing constraints or adjusting parameter ranges.")
    
    return results


def save_results(results: dict, output_path: str = "data/search_results/hybrid_param_search.json"):
    """Save search results to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\n[OK] Results saved to {output_path}")
    return output_file


def load_and_display_results(json_path: str):
    """Load and display results from previous search."""
    with open(json_path) as f:
        results = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"HYBRID PARAMETER SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"Search timestamp: {results.get('timestamp', 'Unknown')}")
    
    if 'metrics_summary' in results:
        summary = results['metrics_summary']
        print(f"\nSummary:")
        print(f"  Best Sharpe: {summary.get('best_sharpe', 'N/A')}")
        print(f"  Mean Sharpe: {summary.get('mean_sharpe', 'N/A'):.2f}")
    
    if 'recommended_config' in results:
        config = results['recommended_config']
        print(f"\nRecommended Configuration:")
        print(f"  RSI Oversold: {config['rsi_oversold']}")
        print(f"  RSI Overbought: {config['rsi_overbought']}")
        print(f"  Min Volume Ratio: {config['volume_ratio']}")
        print(f"\n  Expected Performance:")
        print(f"    Sharpe: {config['expected_sharpe']:.2f}")
        print(f"    Return: {config['expected_return']*100:.1f}%")
        print(f"    Trades: {config['expected_trades']}")
        print(f"    Max DD: {config['expected_max_drawdown']*100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parameter search for Hybrid Phase 3 + Phase 6 strategy"
    )
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'AMZN'],
                       help='Tickers to test')
    parser.add_argument('--period', default='1y',
                       help='Data period (1y, 6mo, 3mo)')
    parser.add_argument('--rsi-oversold', nargs='+', type=int, default=[25, 30, 35, 40, 45],
                       help='RSI oversold thresholds to test')
    parser.add_argument('--rsi-overbought', nargs='+', type=int, default=[60, 65, 70, 75, 80],
                       help='RSI overbought thresholds to test')
    parser.add_argument('--volume-ratios', nargs='+', type=float, default=[0.6, 0.7, 0.8, 0.9],
                       help='Volume ratio thresholds to test')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital for backtest')
    parser.add_argument('--output', default='data/search_results/hybrid_param_search.json',
                       help='Output JSON file')
    parser.add_argument('--load', help='Load and display previous results from JSON')
    
    args = parser.parse_args()
    
    # If loading previous results
    if args.load:
        load_and_display_results(args.load)
    else:
        # Load data
        print(f"\nLoading data for {args.tickers}...")
        try:
            df = load_stock_bars(args.tickers, interval='1d', period=args.period)
            if df.empty:
                print("[ERROR] No data loaded.")
                sys.exit(1)
            print(f"[OK] Loaded {len(df):,} bars")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            sys.exit(1)
        
        # Run search
        results = run_parameter_search(
            df,
            rsi_oversold_values=args.rsi_oversold,
            rsi_overbought_values=args.rsi_overbought,
            volume_ratio_values=args.volume_ratios,
            initial_capital=args.capital,
        )
        
        # Save results
        save_results(results, args.output)
        
        print(f"\n{'='*70}")
        print(f"NEXT STEPS:")
        print(f"{'='*70}")
        print(f"1. Review recommended configuration above")
        print(f"2. Test on different time period:")
        print(f"   python scripts/hybrid_param_search.py --period 6mo --load {args.output}")
        print(f"3. Once validated, update hybrid strategy with best parameters")
        print(f"4. Run paper trading to confirm results")
