"""
Simplified Direct Validation - Individual Symbol Backtests
Runs per-symbol backtests and aggregates for robust cross-asset validation
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


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_return = np.mean(returns) - risk_free_rate / 252
    return (excess_return / np.std(returns)) * np.sqrt(252)


def fetch_and_backtest(tickers: list, start_date: str, end_date: str, strategy: HybridVotingStrategy) -> dict:
    """Fetch data and run backtest for given period."""
    period_label = f"{start_date[:4]}"
    results_by_ticker = {}
    all_returns = []
    
    for ticker in tickers:
        try:
            # Fetch data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < 50:
                print(f"  [!] {ticker}: Insufficient data ({len(df)} bars)")
                continue
            
            # Prepare for single-symbol backtest - handle both Series and DataFrame
            close_vals = df['Close'].values if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0].values
            high_vals = df['High'].values if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0].values
            low_vals = df['Low'].values if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0].values
            open_vals = df['Open'].values if isinstance(df['Open'], pd.Series) else df['Open'].iloc[:, 0].values
            volume_vals = df['Volume'].values if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0].values
            
            # Ensure 1D arrays
            close_vals = close_vals.flatten() if len(close_vals.shape) > 1 else close_vals
            high_vals = high_vals.flatten() if len(high_vals.shape) > 1 else high_vals
            low_vals = low_vals.flatten() if len(low_vals.shape) > 1 else low_vals
            open_vals = open_vals.flatten() if len(open_vals.shape) > 1 else open_vals
            volume_vals = volume_vals.flatten() if len(volume_vals.shape) > 1 else volume_vals
            
            symbol_df = pd.DataFrame({
                'close': close_vals,
                'high': high_vals,
                'low': low_vals,
                'open': open_vals,
                'volume': volume_vals,
                'symbol': ticker,
            })
            
            # Run backtest
            result = strategy.backtest_daily(symbol_df)
            
            if result and isinstance(result, dict):
                results_by_ticker[ticker] = {
                    'sharpe': result.get('sharpe', 0),
                    'trades': result.get('num_trades', 0),
                    'return': result.get('total_return', 0),
                    'max_dd': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                }
                all_returns.append(result.get('total_return', 0))
                print(f"  [OK] {ticker}: Sharpe {result.get('sharpe', 0):.2f}, "
                      f"Trades {result.get('num_trades', 0)}, Return {result.get('total_return', 0):.1%}")
            else:
                print(f"  [!] {ticker}: Backtest failed")
        
        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")
    
    # Aggregate results
    if results_by_ticker:
        avg_sharpe = np.mean([r['sharpe'] for r in results_by_ticker.values()])
        total_trades = sum(r['trades'] for r in results_by_ticker.values())
        avg_return = np.mean([r['return'] for r in results_by_ticker.values()])
        
        return {
            'period': period_label,
            'num_tickers': len(results_by_ticker),
            'average_sharpe': avg_sharpe,
            'total_trades': total_trades,
            'average_return': avg_return,
            'by_ticker': results_by_ticker,
            'passed': avg_sharpe > 0.5 and total_trades > len(tickers) * 15,  # 15+ trades per ticker
        }
    
    return None


def print_results(period_results: list):
    """Print validation results."""
    print("\n" + "="*100)
    print("DIRECT VALIDATION - PER-SYMBOL BACKTESTS")
    print("="*100)
    
    print(f"\n{'Period':<12} {'Tickers':<10} {'Avg Sharpe':<14} {'Tot Trades':<14} {'Avg Return':<14} {'Status':<12}")
    print("-" * 100)
    
    passed = 0
    for result in period_results:
        status = "[PASS]" if result['passed'] else "[WARN]"
        if result['passed']:
            passed += 1
        
        print(f"{result['period']:<12} {result['num_tickers']:<10} {result['average_sharpe']:<14.2f} "
              f"{result['total_trades']:<14} {result['average_return']:<14.1%} {status:<12}")
    
    print("\n" + "="*100)
    print(f"RESULTS: {passed}/{len(period_results)} periods passed validation")
    print("="*100)
    
    if period_results:
        avg_sharpe = np.mean([r['average_sharpe'] for r in period_results])
        avg_trades = np.mean([r['total_trades'] for r in period_results])
        
        print(f"\nAverages:")
        print(f"  Sharpe:      {avg_sharpe:.2f}")
        print(f"  Trades/year: {avg_trades:.0f}")


def main():
    print("\n" + "="*100)
    print("VOTING STRATEGY - DIRECT CROSS-ASSET & MULTI-PERIOD VALIDATION")
    print("="*100)
    
    strategy = HybridVotingStrategy(
        buy_threshold=3,
        max_position_size=0.025,
        indicator_weights={
            'volume': 1.5,
            'adx': 1.5,
            'atr_trend': 1.5,
            'sma_crossover': 0.9,
            'rsi': 0.9,
            'macd': 0.9,
            'bollinger_bands': 0.9,
            'stochastic': 0.9,
        }
    )
    
    print(f"\nConfiguration: buy_threshold=3, max_position=2.5%, weights=confirmation_heavy")
    
    all_results = []
    
    # Test 1: 2023 Large-Cap Tech
    print("\n" + "-"*100)
    print("2023: Large-Cap Tech (AAPL, MSFT, AMZN)")
    print("-"*100)
    result = fetch_and_backtest(['AAPL', 'MSFT', 'AMZN'], '2023-01-01', '2023-12-31', strategy)
    if result:
        all_results.append(result)
    
    # Test 2: 2023 Index ETFs
    print("\n" + "-"*100)
    print("2023: Index ETFs (SPY, QQQ)")
    print("-"*100)
    result = fetch_and_backtest(['SPY', 'QQQ'], '2023-01-01', '2023-12-31', strategy)
    if result:
        all_results.append(result)
    
    # Test 3: 2024 Large-Cap Tech
    print("\n" + "-"*100)
    print("2024: Large-Cap Tech (AAPL, MSFT, AMZN)")
    print("-"*100)
    result = fetch_and_backtest(['AAPL', 'MSFT', 'AMZN'], '2024-01-01', '2024-12-31', strategy)
    if result:
        all_results.append(result)
    
    # Test 4: 2024 Index ETFs
    print("\n" + "-"*100)
    print("2024: Index ETFs (SPY, QQQ)")
    print("-"*100)
    result = fetch_and_backtest(['SPY', 'QQQ'], '2024-01-01', '2024-12-31', strategy)
    if result:
        all_results.append(result)
    
    # Test 5: 2025 YTD Large-Cap Tech
    print("\n" + "-"*100)
    print("2025 YTD: Large-Cap Tech (AAPL, MSFT, AMZN)")
    print("-"*100)
    result = fetch_and_backtest(['AAPL', 'MSFT', 'AMZN'], '2025-01-01', datetime.now().strftime('%Y-%m-%d'), strategy)
    if result:
        all_results.append(result)
    
    # Test 6: 2025 YTD Index ETFs
    print("\n" + "-"*100)
    print("2025 YTD: Index ETFs (SPY, QQQ)")
    print("-"*100)
    result = fetch_and_backtest(['SPY', 'QQQ'], '2025-01-01', datetime.now().strftime('%Y-%m-%d'), strategy)
    if result:
        all_results.append(result)
    
    # Print and save results
    if all_results:
        print_results(all_results)
        
        output_path = Path(__file__).parent.parent / "data" / "search_results" / "direct_validation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to {output_path}")


if __name__ == '__main__':
    main()
