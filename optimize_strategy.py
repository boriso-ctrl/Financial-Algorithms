"""
Strategy Optimization Script

This script tests different combinations of:
1. Technical indicators (SMA, RSI, MACD, BB, etc.)
2. Weight combinations for multi-indicator strategies
3. Parameter sets for each indicator
4. Different ticker combinations

The goal is to find configurations that achieve a Sharpe ratio of 2.5 or higher.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import itertools
import warnings
warnings.filterwarnings('ignore')

# Configuration constants
DEFAULT_MAX_ITERATIONS = 10  # Maximum optimization iterations before stopping

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader_synthetic import load_daily_prices
    print("Using synthetic data loader")
except ImportError:
    try:
        from data_loader_yfinance import load_daily_prices
        print("Using yfinance data loader")
    except ImportError:
        from data_loader import load_daily_prices
        print("Using simfin data loader")

from backtest.simple_backtest import run_backtest
import ta


class IndicatorGenerator:
    """Generate signals from various technical indicators."""
    
    @staticmethod
    def sma_crossover(prices, fast=20, slow=50):
        """SMA crossover signal."""
        sma_fast = prices.rolling(fast).mean()
        sma_slow = prices.rolling(slow).mean()
        return (sma_fast > sma_slow).astype(int)
    
    @staticmethod
    def rsi_signal(prices, period=14, overbought=70, oversold=30):
        """RSI-based signal."""
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            rsi = ta.momentum.rsi(prices[col], window=period)
            # Buy when RSI crosses above oversold, sell when crosses below overbought
            signal = pd.Series(0, index=prices.index)
            signal[rsi < oversold] = 1
            signal[rsi > overbought] = 0
            signals[col] = signal
        return signals.astype(int)
    
    @staticmethod
    def macd_signal(prices, fast=12, slow=26, signal_period=9):
        """MACD signal."""
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            macd = ta.trend.macd(prices[col], window_slow=slow, window_fast=fast)
            macd_signal = ta.trend.macd_signal(prices[col], window_slow=slow, 
                                               window_fast=fast, window_sign=signal_period)
            signals[col] = (macd > macd_signal).astype(int)
        return signals.astype(int)
    
    @staticmethod
    def bollinger_bands_signal(prices, period=20, std_dev=2):
        """Bollinger Bands signal."""
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            bb_high = ta.volatility.bollinger_hband(prices[col], window=period, window_dev=std_dev)
            bb_low = ta.volatility.bollinger_lband(prices[col], window=period, window_dev=std_dev)
            # Buy when price touches lower band, neutral when crosses back
            signal = pd.Series(0, index=prices.index)
            signal[prices[col] <= bb_low] = 1
            signal[prices[col] >= bb_high] = 0
            signals[col] = signal.fillna(method='ffill').fillna(0)
        return signals.astype(int)
    
    @staticmethod
    def stochastic_signal(prices_high, prices_low, prices_close, period=14, smooth=3):
        """Stochastic oscillator signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                stoch = ta.momentum.stoch(prices_high[col], prices_low[col], 
                                         prices_close[col], window=period, smooth_window=smooth)
                # Buy when stoch < 20, sell when stoch > 80
                signal = pd.Series(0, index=prices_close.index)
                signal[stoch < 20] = 1
                signal[stoch > 80] = 0
                signals[col] = signal.fillna(method='ffill').fillna(0)
            else:
                # If we don't have high/low, use close approximation
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def ema_crossover(prices, fast=12, slow=26):
        """EMA crossover signal."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return (ema_fast > ema_slow).astype(int)


class StrategyOptimizer:
    """Optimize trading strategies to achieve target Sharpe ratio."""
    
    def __init__(self, tickers, initial_capital=100000, target_sharpe=2.5):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.target_sharpe = target_sharpe
        self.results_log = []
        self.best_result = None
        self.prices = None
        
    def load_data(self, years=3):
        """Load price data."""
        print(f"Loading data for {self.tickers}...")
        days = years * 252  # Trading days
        self.prices = load_daily_prices(self.tickers, days=days, seed=42)
        print(f"Loaded {len(self.prices)} days of data")
        print(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
        return self.prices
    
    def combine_signals(self, signal_list, weights=None):
        """Combine multiple signals with weights."""
        if weights is None:
            weights = [1.0 / len(signal_list)] * len(signal_list)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted average
        combined = sum(w * sig for w, sig in zip(weights, signal_list))
        
        # Threshold at 0.5 to create binary signal
        return (combined >= 0.5).astype(int)
    
    def test_configuration(self, config_name, signals):
        """Test a specific configuration."""
        try:
            # Align signals with prices
            valid_idx = signals.notna().all(axis=1) & self.prices.notna().all(axis=1)
            test_prices = self.prices[valid_idx]
            test_signals = signals[valid_idx]
            
            if len(test_prices) < 50:
                return None
            
            # Run backtest
            results = run_backtest(test_prices, test_signals, self.initial_capital)
            
            # Extract Sharpe ratio
            sharpe_str = results['metrics']['Sharpe Ratio']
            sharpe = float(sharpe_str)
            
            result = {
                'config_name': config_name,
                'sharpe_ratio': sharpe,
                'metrics': results['metrics'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.results_log.append(result)
            
            # Update best result
            if self.best_result is None or sharpe > self.best_result['sharpe_ratio']:
                self.best_result = result
                print(f"\n{'='*60}")
                print(f"NEW BEST: {config_name}")
                print(f"Sharpe Ratio: {sharpe:.3f}")
                print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            return None
    
    def optimize_single_indicators(self):
        """Test single indicators with various parameters."""
        print("\n" + "="*60)
        print("PHASE 1: Single Indicator Optimization")
        print("="*60 + "\n")
        
        # SMA crossover variations
        sma_params = [
            (10, 30), (20, 50), (30, 90), (50, 200),
            (5, 20), (15, 45), (25, 75), (40, 120)
        ]
        for fast, slow in sma_params:
            config_name = f"SMA_{fast}_{slow}"
            signals = IndicatorGenerator.sma_crossover(self.prices, fast, slow)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # RSI variations
        rsi_params = [
            (14, 70, 30), (14, 80, 20), (14, 75, 25),
            (7, 70, 30), (21, 70, 30), (10, 65, 35)
        ]
        for period, overbought, oversold in rsi_params:
            config_name = f"RSI_{period}_{overbought}_{oversold}"
            signals = IndicatorGenerator.rsi_signal(self.prices, period, overbought, oversold)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # MACD variations
        macd_params = [
            (12, 26, 9), (8, 17, 9), (5, 13, 5),
            (19, 39, 9), (12, 26, 6)
        ]
        for fast, slow, signal in macd_params:
            config_name = f"MACD_{fast}_{slow}_{signal}"
            signals = IndicatorGenerator.macd_signal(self.prices, fast, slow, signal)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Bollinger Bands variations
        bb_params = [
            (20, 2), (20, 2.5), (20, 1.5),
            (10, 2), (30, 2), (15, 2)
        ]
        for period, std_dev in bb_params:
            config_name = f"BB_{period}_{std_dev}"
            signals = IndicatorGenerator.bollinger_bands_signal(self.prices, period, std_dev)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # EMA variations
        ema_params = [
            (12, 26), (9, 21), (5, 13), (20, 50), (8, 17)
        ]
        for fast, slow in ema_params:
            config_name = f"EMA_{fast}_{slow}"
            signals = IndicatorGenerator.ema_crossover(self.prices, fast, slow)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
    
    def optimize_combined_strategies(self):
        """Test combinations of indicators with different weights."""
        print("\n" + "="*60)
        print("PHASE 2: Combined Strategy Optimization")
        print("="*60 + "\n")
        
        # Generate base signals
        base_signals = {
            'SMA_20_50': IndicatorGenerator.sma_crossover(self.prices, 20, 50),
            'SMA_50_200': IndicatorGenerator.sma_crossover(self.prices, 50, 200),
            'RSI_14': IndicatorGenerator.rsi_signal(self.prices, 14, 70, 30),
            'MACD_12_26_9': IndicatorGenerator.macd_signal(self.prices, 12, 26, 9),
            'BB_20_2': IndicatorGenerator.bollinger_bands_signal(self.prices, 20, 2),
            'EMA_12_26': IndicatorGenerator.ema_crossover(self.prices, 12, 26)
        }
        
        # Test 2-indicator combinations
        indicator_pairs = list(itertools.combinations(base_signals.keys(), 2))
        weight_combinations = [
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.3, 0.7]
        ]
        
        for ind1, ind2 in indicator_pairs:
            for weights in weight_combinations:
                config_name = f"COMBO_{ind1}_{ind2}_w{weights[0]:.1f}_{weights[1]:.1f}"
                combined = self.combine_signals([base_signals[ind1], base_signals[ind2]], weights)
                result = self.test_configuration(config_name, combined)
                if result:
                    print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Test 3-indicator combinations
        print("\nTesting 3-indicator combinations...")
        indicator_triplets = list(itertools.combinations(base_signals.keys(), 3))[:10]  # Limit to first 10
        weight_combinations_3 = [
            [0.33, 0.33, 0.34],
            [0.5, 0.25, 0.25],
            [0.4, 0.3, 0.3],
            [0.6, 0.2, 0.2],
            [0.5, 0.3, 0.2]
        ]
        
        for ind1, ind2, ind3 in indicator_triplets:
            for weights in weight_combinations_3:
                config_name = f"COMBO3_{ind1}_{ind2}_{ind3}"
                combined = self.combine_signals([base_signals[ind1], base_signals[ind2], 
                                                base_signals[ind3]], weights)
                result = self.test_configuration(config_name, combined)
                if result:
                    print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
    
    def optimize_advanced_parameters(self):
        """Fine-tune parameters around best configurations."""
        print("\n" + "="*60)
        print("PHASE 3: Advanced Parameter Tuning")
        print("="*60 + "\n")
        
        if not self.best_result:
            print("No best result to optimize around yet.")
            return
        
        # Parse best config to understand what worked
        best_config = self.best_result['config_name']
        print(f"Optimizing around: {best_config} (Sharpe={self.best_result['sharpe_ratio']:.3f})")
        
        # If best is SMA, try variations around it
        if 'SMA' in best_config:
            # Extract parameters (if possible)
            parts = best_config.split('_')
            if len(parts) >= 3:
                try:
                    base_fast = int(parts[1])
                    base_slow = int(parts[2])
                    
                    # Try variations ±20%
                    for fast_delta in [-5, -3, 0, 3, 5]:
                        for slow_delta in [-10, -5, 0, 5, 10]:
                            fast = max(5, base_fast + fast_delta)
                            slow = max(fast + 5, base_slow + slow_delta)
                            config_name = f"SMA_tuned_{fast}_{slow}"
                            signals = IndicatorGenerator.sma_crossover(self.prices, fast, slow)
                            result = self.test_configuration(config_name, signals)
                            if result:
                                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
                except:
                    pass
        
        # Try more aggressive combinations
        print("\nTrying ensemble strategies...")
        all_signals = []
        for fast, slow in [(10, 30), (20, 50), (50, 200)]:
            all_signals.append(IndicatorGenerator.sma_crossover(self.prices, fast, slow))
        
        for fast, slow in [(12, 26), (9, 21)]:
            all_signals.append(IndicatorGenerator.ema_crossover(self.prices, fast, slow))
        
        all_signals.append(IndicatorGenerator.rsi_signal(self.prices, 14, 70, 30))
        all_signals.append(IndicatorGenerator.macd_signal(self.prices, 12, 26, 9))
        
        # Equal weight ensemble
        config_name = "ENSEMBLE_equal_weight"
        combined = self.combine_signals(all_signals)
        result = self.test_configuration(config_name, combined)
        if result:
            print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
    
    def save_results(self, filename='optimization_results.json'):
        """Save optimization results to file."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'tickers': self.tickers,
            'target_sharpe': self.target_sharpe,
            'best_result': self.best_result,
            'all_results': self.results_log
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total configurations tested: {len(self.results_log)}")
        
        if self.best_result:
            print(f"\nBest Configuration: {self.best_result['config_name']}")
            print(f"Sharpe Ratio: {self.best_result['sharpe_ratio']:.3f}")
            print(f"Target Sharpe: {self.target_sharpe}")
            
            if self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n🎉 TARGET ACHIEVED! 🎉")
            else:
                print(f"\nGap to target: {self.target_sharpe - self.best_result['sharpe_ratio']:.3f}")
            
            print("\nDetailed Metrics:")
            for key, value in self.best_result['metrics'].items():
                print(f"  {key}: {value}")
        
        # Top 10 results
        print("\n" + "-"*60)
        print("Top 10 Configurations:")
        print("-"*60)
        sorted_results = sorted(self.results_log, 
                               key=lambda x: x['sharpe_ratio'], 
                               reverse=True)[:10]
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:2d}. {result['config_name']:40s} Sharpe={result['sharpe_ratio']:6.3f}")
    
    def run_optimization(self):
        """Run full optimization process."""
        print("\n" + "="*60)
        print("TRADING STRATEGY OPTIMIZATION")
        print("="*60)
        print(f"Tickers: {self.tickers}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Target Sharpe Ratio: {self.target_sharpe}")
        print("="*60 + "\n")
        
        # Load data
        self.load_data()
        
        iteration = 1
        max_iterations = DEFAULT_MAX_ITERATIONS
        
        while True:
            print(f"\n{'#'*60}")
            print(f"ITERATION {iteration}")
            print(f"{'#'*60}\n")
            
            # Phase 1: Single indicators
            self.optimize_single_indicators()
            
            # Check if target achieved
            if self.best_result and self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n✓ Target Sharpe ratio achieved!")
                break
            
            # Phase 2: Combined strategies
            self.optimize_combined_strategies()
            
            # Check if target achieved
            if self.best_result and self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n✓ Target Sharpe ratio achieved!")
                break
            
            # Phase 3: Advanced tuning
            self.optimize_advanced_parameters()
            
            # Check if target achieved
            if self.best_result and self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n✓ Target Sharpe ratio achieved!")
                break
            
            # Print iteration summary
            print(f"\nIteration {iteration} complete. Best Sharpe so far: {self.best_result['sharpe_ratio']:.3f}")
            
            iteration += 1
            
            # Safety break after reasonable attempts
            if iteration > max_iterations:
                print(f"\nReached maximum iterations ({max_iterations}). Stopping optimization.")
                break
        
        # Print final summary
        self.print_summary()
        
        # Save results
        self.save_results()


def main():
    # Configuration
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    initial_capital = 100000
    target_sharpe = 2.5
    
    # Run optimization
    optimizer = StrategyOptimizer(tickers, initial_capital, target_sharpe)
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
