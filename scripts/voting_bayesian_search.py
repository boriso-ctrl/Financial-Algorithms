"""
Bayesian Optimization for Multi-Indicator Voting Strategy

Searches optimal:
1. buy_threshold (3-7)
2. Indicator weights (RSI, Trend, Momentum emphasis options)
3. max_position_size (2-5%)
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from skopt import gp_minimize, space
from skopt.utils import use_named_args

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.financial_algorithms.data.yfinance_loader import load_stock_bars
from src.financial_algorithms.strategies.multi_indicator_voting import HybridVotingStrategy


# Define weight profiles to test
WEIGHT_PROFILES = {
    'equal': {  # All indicators weighted equally
        'sma_crossover': 1.0, 'rsi': 1.0, 'macd': 1.0, 'bollinger_bands': 1.0,
        'volume': 1.0, 'adx': 1.0, 'stochastic': 1.0, 'atr_trend': 1.0,
    },
    'momentum_heavy': {  # Emphasize momentum indicators
        'sma_crossover': 0.8, 'rsi': 1.5, 'macd': 1.5, 'bollinger_bands': 1.0,
        'volume': 1.0, 'adx': 0.8, 'stochastic': 1.5, 'atr_trend': 0.8,
    },
    'trend_heavy': {  # Emphasize trend indicators
        'sma_crossover': 1.5, 'rsi': 0.8, 'macd': 0.8, 'bollinger_bands': 1.0,
        'volume': 1.0, 'adx': 1.5, 'stochastic': 0.8, 'atr_trend': 1.5,
    },
    'confirmation_heavy': {  # Emphasize confirmation: Volume, ADX, ATR
        'sma_crossover': 0.9, 'rsi': 0.9, 'macd': 0.9, 'bollinger_bands': 0.9,
        'volume': 1.5, 'adx': 1.5, 'stochastic': 0.9, 'atr_trend': 1.5,
    },
}


def run_backtest(
    df: pd.DataFrame,
    buy_threshold: int,
    max_position_size: float,
    weight_profile: str,
) -> dict:
    """Run single backtest with given parameters."""
    
    weights = WEIGHT_PROFILES.get(weight_profile, WEIGHT_PROFILES['equal'])
    
    strategy = HybridVotingStrategy(
        buy_threshold=buy_threshold,
        exit_threshold=0,
        max_position_size=max_position_size,
        target_avg_position=0.02,
        indicator_weights=weights,
    )
    
    try:
        metrics = strategy.backtest_daily(df, initial_capital=100000)
        return metrics
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        return {'sharpe': -999, 'num_trades': 0}


def bayesian_search(
    df: pd.DataFrame,
    n_calls: int = 50,
    n_initial: int = 10,
) -> dict:
    """
    Bayesian optimization search for best parameters.
    
    Args:
        df: OHLCV data
        n_calls: Total function evaluations
        n_initial: Random samples before optimization
    
    Returns:
        Dict with search results and best config
    """
    
    print(f"\n{'='*70}")
    print("BAYESIAN OPTIMIZATION: Multi-Indicator Voting Strategy")
    print(f"{'='*70}")
    print(f"Search space:")
    print(f"  buy_threshold: [3, 4, 5, 6, 7]")
    print(f"  weight_profile: ['equal', 'momentum_heavy', 'trend_heavy', 'confirmation_heavy']")
    print(f"  max_position_size: [2%, 3%, 4%, 5%]")
    print(f"  Total combinations: ~80 possible, testing {n_calls} with Bayesian optimization")
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_calls': n_calls,
        'n_initial': n_initial,
        'evaluations': [],
    }
    
    # Define search space as list (required by gp_minimize)
    dimensions = [
        space.Integer(3, 7, name='buy_threshold'),
        space.Integer(0, len(WEIGHT_PROFILES)-1, name='weight_profile_idx'),
        space.Real(0.02, 0.05, prior='uniform', name='max_position_size'),
    ]
    
    weight_profile_list = list(WEIGHT_PROFILES.keys())
    eval_count = [0]
    
    @use_named_args(dimensions)
    def objective(buy_threshold, weight_profile_idx, max_position_size):
        """Objective function to minimize (negative Sharpe)."""
        eval_count[0] += 1
        weight_profile = weight_profile_list[weight_profile_idx]
        
        metrics = run_backtest(df, buy_threshold, max_position_size, weight_profile)
        
        # Objective: maximize Sharpe (minimize negative Sharpe)
        # Also penalize if too few trades
        sharpe = metrics['sharpe']
        num_trades = metrics['num_trades']
        
        # Penalty for too few trades (< 10 per year)
        if num_trades < 10:
            sharpe -= 0.5 * (10 - num_trades) / 10
        
        result = {
            'buy_threshold': buy_threshold,
            'weight_profile': weight_profile,
            'max_position_size': max_position_size,
            'sharpe': sharpe,
            'num_trades': num_trades,
        }
        results['evaluations'].append(result)
        
        print(f"[{eval_count[0]:2d}] Threshold {buy_threshold}, {weight_profile:20s}, "
              f"Size {max_position_size:.1%} → Sharpe {sharpe:6.2f}, "
              f"Trades {num_trades:3.0f}")
        
        # Return negative Sharpe (we're minimizing)
        return -sharpe
    
    # Run Bayesian optimization
    print(f"\n{'Testing':<5} {'Threshold':<12} {'Weight Profile':<20} {'Position':<10} "
          f"{'Sharpe':<10} {'Trades':<10}")
    print("-" * 70)
    
    res = gp_minimize(
        objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        acq_func='EI',
        random_state=42,
        verbose=0,
    )
    
    # Get best result
    best_idx = np.argmin(res.func_vals)
    best_result = results['evaluations'][best_idx]
    
    results['best'] = best_result
    results['best_sharpe'] = best_result['sharpe']
    results['all_sharpes'] = [r['sharpe'] for r in results['evaluations']]
    
    return results


def main():
    """Run optimization and display results."""
    
    # Load data
    print("\nLoading data...")
    df = load_stock_bars(['AAPL', 'MSFT', 'AMZN'], interval='1d', period='1y')
    print(f"[OK] Loaded {len(df):,} bars")
    
    # Run Bayesian search
    results = bayesian_search(df, n_calls=50, n_initial=10)
    
    # Display results
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    
    best = results['best']
    print(f"\nBest Configuration Found:")
    print(f"  Buy Threshold: {best['buy_threshold']}")
    print(f"  Weight Profile: {best['weight_profile']}")
    print(f"  Max Position Size: {best['max_position_size']:.1%}")
    print(f"  Expected Sharpe: {best['sharpe']:.2f}")
    print(f"  Expected Trades: {best['num_trades']:.0f}")
    
    # Top 5 configurations
    sorted_evals = sorted(results['evaluations'], key=lambda x: x['sharpe'], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Threshold':<12} {'Profile':<20} {'Position':<12} "
          f"{'Sharpe':<10} {'Trades':<10}")
    print("-" * 70)
    
    for i, config in enumerate(sorted_evals[:5], 1):
        print(f"{i:<5} {config['buy_threshold']:<12} {config['weight_profile']:<20} "
              f"{config['max_position_size']:.1%}           {config['sharpe']:<10.2f} "
              f"{config['num_trades']:<10.0f}")
    
    # Statistics
    sharpes = results['all_sharpes']
    print(f"\n{'='*70}")
    print("SEARCH STATISTICS")
    print(f"{'='*70}")
    print(f"Best Sharpe: {max(sharpes):>8.2f}")
    print(f"Mean Sharpe: {np.mean(sharpes):>8.2f}")
    print(f"Std Sharpe:  {np.std(sharpes):>8.2f}")
    print(f"Improvement vs baseline (threshold=5, equal weights): "
          f"{max(sharpes) - 0.61:+.2f} Sharpe")
    
    # Save results
    output_file = Path('data/search_results/voting_bayesian_search.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, default=convert, indent=2)
    
    print(f"\n[OK] Results saved to {output_file}")
    
    # Next steps
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"1. Deploy best configuration:")
    print(f"   strategy = HybridVotingStrategy(")
    print(f"       buy_threshold={best['buy_threshold']},")
    print(f"       max_position_size={best['max_position_size']:.4f},")
    print(f"       indicator_weights=WEIGHT_PROFILES['{best['weight_profile']}'],")
    print(f"   )")
    print(f"2. Validate on different time period (2023-2024)")
    print(f"3. Test on other assets (SPY, QQQ)")
    print(f"4. Run paper trading for 2 weeks")


if __name__ == "__main__":
    main()
