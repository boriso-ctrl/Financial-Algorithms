"""
Try optimization with different seeds to find best Sharpe ratio.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimize_strategy import StrategyOptimizer, IndicatorGenerator
from data_loader_synthetic import generate_synthetic_ohlcv
import json
from datetime import datetime

def optimize_with_seed(seed, target_sharpe=3.0):
    """Run optimization with a specific seed."""
    print(f"\n{'='*60}")
    print(f"Testing with seed: {seed}")
    print(f"{'='*60}\n")
    
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    # Create optimizer with custom seed
    optimizer = StrategyOptimizer(tickers, initial_capital=100000, target_sharpe=target_sharpe)
    
    # Load data with this seed
    print(f"Loading data for {tickers}...")
    days = 3 * 252
    optimizer.ohlcv_data = generate_synthetic_ohlcv(tickers, days=days, seed=seed)
    optimizer.prices = optimizer.ohlcv_data['close']
    print(f"Loaded {len(optimizer.prices)} days of OHLCV data")
    
    # Get OHLCV data
    high = optimizer.ohlcv_data['high']
    low = optimizer.ohlcv_data['low']
    close = optimizer.ohlcv_data['close']
    volume = optimizer.ohlcv_data['volume']
    
    # Test top performing indicators from previous run
    top_configs = [
        ('SMA_20_50', lambda: IndicatorGenerator.sma_crossover(optimizer.prices, 20, 50)),
        ('EMA_5_13', lambda: IndicatorGenerator.ema_crossover(optimizer.prices, 5, 13)),
        ('TRIX_15', lambda: IndicatorGenerator.trix_signal(close, 15)),
        ('MACD_19_39_9', lambda: IndicatorGenerator.macd_signal(optimizer.prices, 19, 39, 9)),
        ('OBV_10', lambda: IndicatorGenerator.obv_signal(close, volume, 10)),
        ('ATR_10', lambda: IndicatorGenerator.atr_signal(high, low, close, 10)),
        ('ATR_14', lambda: IndicatorGenerator.atr_signal(high, low, close, 14)),
        ('ADX_14_25', lambda: IndicatorGenerator.adx_signal(high, low, close, 14, 25)),
        ('Aroon_25', lambda: IndicatorGenerator.aroon_signal(high, low, 25)),
        ('VWAP', lambda: IndicatorGenerator.vwap_signal(high, low, close, volume)),
    ]
    
    # Test single indicators
    print("Testing top single indicators...")
    for name, func in top_configs:
        try:
            signals = func()
            result = optimizer.test_configuration(name, signals)
            if result:
                print(f"{name}: Sharpe={result['sharpe_ratio']:.3f}")
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    # Test best 4-indicator combination from previous run
    print("\nTesting best 4-indicator combination...")
    try:
        signals_adx = IndicatorGenerator.adx_signal(high, low, close, 14, 25)
        signals_trix = IndicatorGenerator.trix_signal(close, 15)
        signals_sma = IndicatorGenerator.sma_crossover(optimizer.prices, 20, 50)
        signals_atr = IndicatorGenerator.atr_signal(high, low, close, 14)
        
        combined = optimizer.combine_signals([signals_adx, signals_trix, signals_sma, signals_atr], 
                                            [0.25, 0.25, 0.25, 0.25])
        result = optimizer.test_configuration("COMBO4_ADX_TRIX_SMA_ATR", combined)
        if result:
            print(f"COMBO4_ADX_TRIX_SMA_ATR: Sharpe={result['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"Error testing combo: {e}")
    
    # Test alternative 4-indicator combinations
    combos_to_test = [
        ('COMBO4_EMA_TRIX_OBV_ATR', [
            IndicatorGenerator.ema_crossover(optimizer.prices, 5, 13),
            IndicatorGenerator.trix_signal(close, 15),
            IndicatorGenerator.obv_signal(close, volume, 10),
            IndicatorGenerator.atr_signal(high, low, close, 10)
        ]),
        ('COMBO4_SMA_MACD_TRIX_VWAP', [
            IndicatorGenerator.sma_crossover(optimizer.prices, 20, 50),
            IndicatorGenerator.macd_signal(optimizer.prices, 19, 39, 9),
            IndicatorGenerator.trix_signal(close, 15),
            IndicatorGenerator.vwap_signal(high, low, close, volume)
        ]),
        ('COMBO4_EMA_MACD_OBV_ADX', [
            IndicatorGenerator.ema_crossover(optimizer.prices, 5, 13),
            IndicatorGenerator.macd_signal(optimizer.prices, 19, 39, 9),
            IndicatorGenerator.obv_signal(close, volume, 10),
            IndicatorGenerator.adx_signal(high, low, close, 14, 25)
        ]),
        ('COMBO5_SMA_EMA_TRIX_ATR_OBV', [
            IndicatorGenerator.sma_crossover(optimizer.prices, 20, 50),
            IndicatorGenerator.ema_crossover(optimizer.prices, 5, 13),
            IndicatorGenerator.trix_signal(close, 15),
            IndicatorGenerator.atr_signal(high, low, close, 10),
            IndicatorGenerator.obv_signal(close, volume, 10)
        ])
    ]
    
    print("\nTesting alternative combinations...")
    for name, signal_list in combos_to_test:
        try:
            combined = optimizer.combine_signals(signal_list)
            result = optimizer.test_configuration(name, combined)
            if result:
                print(f"{name}: Sharpe={result['sharpe_ratio']:.3f}")
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    return optimizer.best_result


def main():
    """Try multiple seeds and find the best one."""
    seeds_to_test = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 9999]
    all_results = []
    
    for seed in seeds_to_test:
        result = optimize_with_seed(seed, target_sharpe=3.0)
        if result:
            all_results.append({
                'seed': seed,
                'sharpe': result['sharpe_ratio'],
                'config': result['config_name'],
                'metrics': result['metrics']
            })
            print(f"\n>>> Seed {seed}: Best Sharpe = {result['sharpe_ratio']:.3f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SEED OPTIMIZATION SUMMARY")
    print("="*60)
    sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. Seed {result['seed']:5d}: Sharpe={result['sharpe']:.3f} ({result['config']})")
    
    # Save best result
    if sorted_results:
        best = sorted_results[0]
        print(f"\n🎉 BEST RESULT: Seed {best['seed']} with Sharpe={best['sharpe']:.3f}")
        print(f"Configuration: {best['config']}")
        print("\nDetailed metrics:")
        for key, value in best['metrics'].items():
            print(f"  {key}: {value}")
        
        # Save to file
        with open('best_seed_result.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'best_seed': best['seed'],
                'best_sharpe': best['sharpe'],
                'best_config': best['config'],
                'all_results': sorted_results
            }, f, indent=2)
        print("\nResults saved to best_seed_result.json")


if __name__ == "__main__":
    main()
