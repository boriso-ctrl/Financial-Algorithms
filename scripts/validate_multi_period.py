"""
Multi-Period Validation for Voting Strategy
Tests best configuration across different time periods (2023, 2024, 2025)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_algorithms.strategies.multi_indicator_voting import HybridVotingStrategy


def fetch_period_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """Fetch OHLCV data for tickers in a specific period."""
    print(f"\n[*] Fetching data from {start_date} to {end_date} for {len(tickers)} assets...")
    
    prices = {}
    volumes = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < 50:
                print(f"  [!] Insufficient data for {ticker} ({len(df)} bars)")
                continue
            
            if isinstance(df['Close'], pd.Series):
                prices[ticker] = df['Close']
                volumes[ticker] = df['Volume']
            else:
                prices[ticker] = df['Close'][ticker]
                volumes[ticker] = df['Volume'][ticker]
            
            print(f"  [OK] {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")
    
    return prices, volumes


def run_period_backtest(prices: dict, volumes: dict, strategy: HybridVotingStrategy, period_name: str) -> dict:
    """Run backtest for a single period."""
    print(f"\n[>] Backtesting {period_name}...")
    
    if not prices:
        print(f"  [ERR] No data to backtest")
        return None
    
    all_data = []
    for ticker in prices.keys():
        try:
            price_series = prices[ticker]
            volume_series = volumes[ticker]
            
            if isinstance(price_series, pd.DataFrame):
                price_vals = price_series.iloc[:, 0].values.flatten()
            else:
                price_vals = price_series.values.flatten()
            
            if isinstance(volume_series, pd.DataFrame):
                volume_vals = volume_series.iloc[:, 0].values.flatten()
            else:
                volume_vals = volume_series.values.flatten()
            
            df = pd.DataFrame({
                'close': price_vals,
                'volume': volume_vals,
                'symbol': ticker,
            })
            
            df['high'] = df['close'] * 1.001
            df['low'] = df['close'] * 0.999
            df['open'] = df['close']
            all_data.append(df)
        except Exception as e:
            print(f"  [!] Could not prepare {ticker}: {e}")
    
    if not all_data:
        print(f"  [ERR] No valid data")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    try:
        results = strategy.backtest_daily(combined_df)
        
        if not results or not isinstance(results, dict):
            print(f"  [ERR] Invalid result")
            return None
        
        return {
            'period': period_name,
            'assets': len(prices),
            'sharpe': results.get('sharpe', 0),
            'num_trades': results.get('num_trades', 0),
            'total_return': results.get('total_return', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'win_rate': results.get('win_rate', 0),
            'avg_position': results.get('avg_position_size', 0),
        }
    
    except Exception as e:
        print(f"  [ERR] Backtest failed: {e}")
        return None


def print_multi_period_results(all_results: list):
    """Print multi-period comparison table."""
    print("\n" + "="*120)
    print("MULTI-PERIOD VALIDATION RESULTS")
    print("="*120)
    
    print(f"\n{'Period':<25} {'Assets':<10} {'Sharpe':<12} {'Trades':<12} {'Return':<12} {'Max DD':<12} {'Win Rate':<12}")
    print("-" * 120)
    
    valid_results = [r for r in all_results if r is not None]
    passed = 0
    
    for result in valid_results:
        sharpe_ok = "OK" if result['sharpe'] > 0.5 else "LOW"
        trades_ok = "OK" if result['num_trades'] > 50 else "LOW"
        status = "[PASS]" if result['sharpe'] > 0.5 and result['num_trades'] > 50 else "[WARN]"
        
        if status == "[PASS]":
            passed += 1
        
        print(f"{result['period']:<25} {result['assets']:<10} "
              f"{result['sharpe']:<12.2f} {result['num_trades']:<12} "
              f"{result['total_return']:<12.1%} {result['max_drawdown']:<12.1%} "
              f"{result['win_rate']:<12.1%} {status}")
    
    print("\n" + "="*120)
    print(f"VALIDATION SUMMARY")
    print("="*120)
    
    print(f"\nPeriods Passed: {passed}/{len(valid_results)} (threshold: Sharpe > 0.5 AND Trades > 50)")
    
    if valid_results:
        avg_sharpe = np.mean([r['sharpe'] for r in valid_results])
        avg_trades = np.mean([r['num_trades'] for r in valid_results])
        avg_return = np.mean([r['total_return'] for r in valid_results])
        
        print(f"\nAverage across periods:")
        print(f"  Sharpe:       {avg_sharpe:.2f}")
        print(f"  Trades:       {avg_trades:.0f}")
        print(f"  Return:       {avg_return:.1%}")
    
    print(f"\nTarget Criteria:")
    print(f"  Sharpe > 0.5:      {sum(1 for r in valid_results if r['sharpe'] > 0.5)}/{len(valid_results)}")
    print(f"  Trades > 50:       {sum(1 for r in valid_results if r['num_trades'] > 50)}/{len(valid_results)}")
    
    if passed >= len(valid_results) * 0.8:  # 80% pass rate
        print(f"\n[OK] VALIDATION PASSED - Strategy is robust across periods")
    else:
        print(f"\n[!] VALIDATION WARNING - Strategy needs improvement")


def main():
    print("\n" + "="*120)
    print("VOTING STRATEGY - MULTI-PERIOD VALIDATION")
    print("="*120)
    
    best_config = {
        'buy_threshold': 3,
        'max_position_size': 0.025,
        'indicator_weights': {
            'volume': 1.5,
            'adx': 1.5,
            'atr_trend': 1.5,
            'sma_crossover': 0.9,
            'rsi': 0.9,
            'macd': 0.9,
            'bollinger_bands': 0.9,
            'stochastic': 0.9,
        }
    }
    
    print(f"\nConfiguration: buy_threshold={best_config['buy_threshold']}, "
          f"max_position={best_config['max_position_size']:.1%}")
    print(f"Weights: confirmation_heavy")
    
    strategy = HybridVotingStrategy(
        buy_threshold=best_config['buy_threshold'],
        max_position_size=best_config['max_position_size'],
        indicator_weights=best_config['indicator_weights']
    )
    
    # Test tickers: Mix of assets
    baseline_tickers = ['AAPL', 'MSFT', 'AMZN']
    index_tickers = ['SPY', 'QQQ']
    growth_tickers = ['TSLA']
    
    all_test_results = []
    
    # Test 1: 2023 Data
    print("\n" + "-"*120)
    print("PERIOD 1: 2023 (Full Year)")
    print("-"*120)
    
    prices_2023, volumes_2023 = fetch_period_data(baseline_tickers, '2023-01-01', '2023-12-31')
    if prices_2023:
        result_2023 = run_period_backtest(prices_2023, volumes_2023, strategy, "2023 (Large-Cap Tech)")
        all_test_results.append(result_2023)
    
    prices_2023_idx, volumes_2023_idx = fetch_period_data(index_tickers, '2023-01-01', '2023-12-31')
    if prices_2023_idx:
        result_2023_idx = run_period_backtest(prices_2023_idx, volumes_2023_idx, strategy, "2023 (Index ETFs)")
        all_test_results.append(result_2023_idx)
    
    # Test 2: 2024 Data
    print("\n" + "-"*120)
    print("PERIOD 2: 2024 (Full Year)")
    print("-"*120)
    
    prices_2024, volumes_2024 = fetch_period_data(baseline_tickers, '2024-01-01', '2024-12-31')
    if prices_2024:
        result_2024 = run_period_backtest(prices_2024, volumes_2024, strategy, "2024 (Large-Cap Tech)")
        all_test_results.append(result_2024)
    
    prices_2024_idx, volumes_2024_idx = fetch_period_data(index_tickers, '2024-01-01', '2024-12-31')
    if prices_2024_idx:
        result_2024_idx = run_period_backtest(prices_2024_idx, volumes_2024_idx, strategy, "2024 (Index ETFs)")
        all_test_results.append(result_2024_idx)
    
    # Test 3: 2025 YTD
    print("\n" + "-"*120)
    print("PERIOD 3: 2025 (Year-to-Date)")
    print("-"*120)
    
    prices_2025, volumes_2025 = fetch_period_data(baseline_tickers, '2025-01-01', '2025-12-31')
    if prices_2025:
        result_2025 = run_period_backtest(prices_2025, volumes_2025, strategy, "2025 (Large-Cap Tech)")
        all_test_results.append(result_2025)
    
    prices_2025_idx, volumes_2025_idx = fetch_period_data(index_tickers, '2025-01-01', '2025-12-31')
    if prices_2025_idx:
        result_2025_idx = run_period_backtest(prices_2025_idx, volumes_2025_idx, strategy, "2025 (Index ETFs)")
        all_test_results.append(result_2025_idx)
    
    # Print results
    if all_test_results:
        print_multi_period_results(all_test_results)
        
        output_path = Path(__file__).parent.parent / "data" / "search_results" / "multi_period_validation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': best_config,
                'results': all_test_results
            }, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to {output_path}")


if __name__ == '__main__':
    main()
