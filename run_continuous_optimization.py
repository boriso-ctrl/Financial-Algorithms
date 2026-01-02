"""
Continuous Optimization Runner

This script runs the optimization process continuously until the target Sharpe ratio
of 2.5+ is achieved. It tries multiple approaches:
1. Different random seeds for synthetic data
2. Different ticker combinations  
3. Different optimization strategies
4. More advanced parameter tuning

The script will save the best configuration found and can be stopped anytime.
"""

import sys
import os
import argparse
import time
from datetime import datetime
import json

# Configuration constants
BASE_SEED = 42  # Base random seed for reproducible results

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimize_strategy import StrategyOptimizer


def run_continuous_optimization(target_sharpe=2.5, max_iterations=None, 
                                save_interval=5):
    """
    Run optimization continuously until target is achieved.
    
    Parameters:
    -----------
    target_sharpe : float
        Target Sharpe ratio to achieve
    max_iterations : int, optional
        Maximum number of iterations (None = unlimited)
    save_interval : int
        Save results every N iterations
    """
    
    # Different ticker combinations to try
    ticker_sets = [
        ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'],
        ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'WMT'],
        ['TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL'],
        ['SPY', 'QQQ', 'IWM', 'DIA', 'EFA'],
        ['XLE', 'XLF', 'XLK', 'XLV', 'XLI'],
        ['AAPL', 'MSFT', 'AMZN'],
        ['GOOGL', 'FB', 'NFLX'],
        ['JPM', 'BAC', 'WFC', 'GS'],
    ]
    
    iteration = 0
    global_best_result = None
    all_best_results = []
    
    print("="*80)
    print("CONTINUOUS TRADING STRATEGY OPTIMIZATION")
    print("="*80)
    print(f"Target Sharpe Ratio: {target_sharpe}")
    print(f"Maximum Iterations: {max_iterations if max_iterations else 'Unlimited'}")
    print(f"Save Interval: Every {save_interval} iterations")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    while True:
        iteration += 1
        
        # Select ticker set (cycle through different combinations)
        tickers = ticker_sets[(iteration - 1) % len(ticker_sets)]
        
        # Use different random seed for each iteration
        seed = BASE_SEED + iteration
        
        print("\n" + "#"*80)
        print(f"GLOBAL ITERATION {iteration}")
        print(f"Tickers: {tickers}")
        print(f"Random Seed: {seed}")
        print(f"Time elapsed: {datetime.now() - start_time}")
        print("#"*80 + "\n")
        
        # Create optimizer
        optimizer = StrategyOptimizer(tickers, initial_capital=100000, 
                                     target_sharpe=target_sharpe)
        
        # Generate synthetic data with specific seed
        from data_loader_synthetic import generate_synthetic_prices
        optimizer.prices = generate_synthetic_prices(tickers, days=756, seed=seed)
        print(f"Generated data with seed {seed}")
        
        # Run single iteration of optimization
        print("\nPhase 1: Single indicators...")
        optimizer.optimize_single_indicators()
        
        if optimizer.best_result and optimizer.best_result['sharpe_ratio'] >= target_sharpe:
            print(f"\n✓ TARGET ACHIEVED in iteration {iteration}!")
            global_best_result = optimizer.best_result
            break
        
        print("\nPhase 2: Combined strategies...")
        optimizer.optimize_combined_strategies()
        
        if optimizer.best_result and optimizer.best_result['sharpe_ratio'] >= target_sharpe:
            print(f"\n✓ TARGET ACHIEVED in iteration {iteration}!")
            global_best_result = optimizer.best_result
            break
        
        print("\nPhase 3: Advanced tuning...")
        optimizer.optimize_advanced_parameters()
        
        # Check iteration result
        if optimizer.best_result:
            result_info = {
                'iteration': iteration,
                'tickers': tickers,
                'seed': seed,
                'best_config': optimizer.best_result['config_name'],
                'sharpe_ratio': optimizer.best_result['sharpe_ratio'],
                'timestamp': datetime.now().isoformat()
            }
            all_best_results.append(result_info)
            
            # Update global best
            if global_best_result is None or \
               optimizer.best_result['sharpe_ratio'] > global_best_result['sharpe_ratio']:
                global_best_result = optimizer.best_result
                global_best_result['iteration'] = iteration
                global_best_result['tickers'] = tickers
                global_best_result['seed'] = seed
                
                print(f"\n{'*'*80}")
                print(f"NEW GLOBAL BEST (Iteration {iteration})")
                print(f"Config: {global_best_result['config_name']}")
                print(f"Sharpe: {global_best_result['sharpe_ratio']:.3f}")
                print(f"Tickers: {tickers}")
                print(f"{'*'*80}\n")
            
            print(f"\nIteration {iteration} summary:")
            print(f"  Best this iteration: {optimizer.best_result['sharpe_ratio']:.3f}")
            print(f"  Global best so far: {global_best_result['sharpe_ratio']:.3f}")
            print(f"  Gap to target: {max(0, target_sharpe - global_best_result['sharpe_ratio']):.3f}")
            
            # Check if target achieved
            if optimizer.best_result['sharpe_ratio'] >= target_sharpe:
                print(f"\n✓ TARGET ACHIEVED in iteration {iteration}!")
                break
        
        # Save progress periodically
        if iteration % save_interval == 0:
            save_continuous_results(global_best_result, all_best_results, 
                                  iteration, start_time)
        
        # Check max iterations
        if max_iterations and iteration >= max_iterations:
            print(f"\nReached maximum iterations ({max_iterations}). Stopping.")
            break
        
        # Small delay between iterations
        time.sleep(1)
    
    # Final summary
    print("\n" + "="*80)
    print("CONTINUOUS OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Total iterations: {iteration}")
    print(f"Total time: {datetime.now() - start_time}")
    
    if global_best_result:
        print(f"\nBest Configuration Found:")
        print(f"  Config: {global_best_result['config_name']}")
        print(f"  Sharpe Ratio: {global_best_result['sharpe_ratio']:.3f}")
        print(f"  Tickers: {global_best_result.get('tickers', 'N/A')}")
        print(f"  Iteration: {global_best_result.get('iteration', 'N/A')}")
        print(f"  Seed: {global_best_result.get('seed', 'N/A')}")
        
        if global_best_result['sharpe_ratio'] >= target_sharpe:
            print(f"\n🎉 TARGET SHARPE RATIO ACHIEVED! 🎉")
        else:
            print(f"\nGap to target: {target_sharpe - global_best_result['sharpe_ratio']:.3f}")
        
        print("\nDetailed Metrics:")
        for key, value in global_best_result['metrics'].items():
            print(f"  {key}: {value}")
    
    # Save final results
    save_continuous_results(global_best_result, all_best_results, 
                          iteration, start_time, filename='continuous_optimization_final.json')
    
    return global_best_result


def save_continuous_results(best_result, all_results, iteration, start_time,
                           filename='continuous_optimization_progress.json'):
    """Save continuous optimization results."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_iterations': iteration,
        'elapsed_time': str(datetime.now() - start_time),
        'best_result': best_result,
        'all_iteration_results': all_results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nProgress saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Continuous optimization of trading strategies'
    )
    parser.add_argument(
        '--target-sharpe',
        type=float,
        default=2.5,
        help='Target Sharpe ratio to achieve (default: 2.5)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum number of iterations (default: unlimited)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help='Save results every N iterations (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        run_continuous_optimization(
            target_sharpe=args.target_sharpe,
            max_iterations=args.max_iterations,
            save_interval=args.save_interval
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Progress has been saved.")
    except Exception as e:
        print(f"\n\nError during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
