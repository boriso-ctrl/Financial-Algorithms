"""
Quick Parameter Tuning - Buy Threshold Optimization
Tests buy_threshold values [2, 2.5, 3, 3.5, 4] to find optimal entry sensitivity
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


def quick_test(ticker: str, threshold: int) -> dict:
    """Quick single-stock test for given threshold."""
    try:
        # Fetch 1 year data
        df = yf.download(ticker, period='1y', progress=False)
        if df.empty:
            return None
        
        # Flatten values
        symbol_df = pd.DataFrame({
            'close': df['Close'].values.flatten() if len(df['Close'].values.shape) > 1 else df['Close'].values,
            'high': df['High'].values.flatten() if len(df['High'].values.shape) > 1 else df['High'].values,
            'low': df['Low'].values.flatten() if len(df['Low'].values.shape) > 1 else df['Low'].values,
            'open': df['Open'].values.flatten() if len(df['Open'].values.shape) > 1 else df['Open'].values,
            'volume': df['Volume'].values.flatten() if len(df['Volume'].values.shape) > 1 else df['Volume'].values,
            'symbol': ticker,
        })
        
        # Test with given threshold
        strategy = HybridVotingStrategy(
            buy_threshold=threshold,
            max_position_size=0.03,
            indicator_weights={
                'volume': 1.0,
                'adx': 1.0,
                'atr_trend': 1.0,
                'sma_crossover': 1.0,
                'rsi': 1.0,
                'macd': 1.0,
                'bollinger_bands': 1.0,
                'stochastic': 1.0,
            }
        )
        
        result = strategy.backtest_daily(symbol_df)
        
        if result and isinstance(result, dict):
            return {
                'ticker': ticker,
                'threshold': threshold,
                'sharpe': result.get('sharpe', 0),
                'trades': result.get('num_trades', 0),
                'return': result.get('total_return', 0),
                'win_rate': result.get('win_rate', 0),
            }
    except:
        pass
    
    return None


def main():
    print("\n" + "="*100)
    print("PARAMETER TUNING: BUY THRESHOLD OPTIMIZATION")
    print("="*100)
    
    tickers = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TSLA']  # Diverse assets
    thresholds = [1, 2, 2.5, 3, 3.5, 4, 5]  # Range of entry sensitivity
    
    all_results = []
    
    print(f"\nTesting {len(tickers)} assets across {len(thresholds)} thresholds...\n")
    
    for threshold in thresholds:
        print(f"\n{'─'*100}")
        print(f"Testing buy_threshold = {threshold}")
        print(f"{'─'*100}")
        
        threshold_results = []
        
        for ticker in tickers:
            result = quick_test(ticker, threshold)
            if result:
                threshold_results.append(result)
                print(f"  {result['ticker']:<8} Sharpe: {result['sharpe']:>7.2f}, "
                      f"Trades: {result['trades']:>3}, Return: {result['return']:>7.1%}, "
                      f"WinRate: {result['win_rate']:>5.1%}")
            else:
                print(f"  {ticker:<8} [FAILED]")
        
        if threshold_results:
            avg_sharpe = np.mean([r['sharpe'] for r in threshold_results])
            avg_trades = np.mean([r['trades'] for r in threshold_results])
            passed = sum(1 for r in threshold_results if r['sharpe'] > 0.5)
            
            print(f"\n  Summary: AvgSharpe={avg_sharpe:.2f}, AvgTrades={avg_trades:.0f}, "
                  f"Passed={passed}/{len(threshold_results)}")
            
            all_results.extend(threshold_results)
    
    # Print final analysis
    print("\n" + "="*100)
    print("THRESHOLD ANALYSIS")
    print("="*100)
    
    # Group by threshold
    by_threshold = {}
    for r in all_results:
        t = r['threshold']
        if t not in by_threshold:
            by_threshold[t] = []
        by_threshold[t].append(r)
    
    print(f"\n{'Threshold':<12} {'Avg Sharpe':<14} {'Avg Trades':<14} {'Best Ticker':<14} {'Worst Ticker':<14}")
    print("-" * 100)
    
    for threshold in sorted(by_threshold.keys()):
        results = by_threshold[threshold]
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_trades = np.mean([r['trades'] for r in results])
        best = max(results, key=lambda x: x['sharpe'])
        worst = min(results, key=lambda x: x['sharpe'])
        
        print(f"{threshold:<12} {avg_sharpe:<14.2f} {avg_trades:<14.0f} "
              f"{best['ticker']} ({best['sharpe']:.2f})  {worst['ticker']} ({worst['sharpe']:.2f})")
    
    # Recommend best threshold
    best_by_sharpe = {}
    best_by_trades = {}
    
    for threshold, results in by_threshold.items():
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_trades = np.mean([r['trades'] for r in results])
        best_by_sharpe[threshold] = avg_sharpe
        best_by_trades[threshold] = avg_trades
    
    best_threshold_sharpe = max(best_by_sharpe, key=best_by_sharpe.get)
    best_threshold_trades = max(best_by_trades, key=best_by_trades.get)
    
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    sharpe_rec = by_threshold[best_threshold_sharpe]
    avg_sharpe_rec = np.mean([r['sharpe'] for r in sharpe_rec])
    
    print(f"\nBest for Sharpe: threshold={best_threshold_sharpe}")
    print(f"  Average Sharpe: {avg_sharpe_rec:.2f}")
    print(f"  Average Trades: {np.mean([r['trades'] for r in sharpe_rec]):.0f}")
    print(f"  Assets passing (Sharpe > 0.5): {sum(1 for r in sharpe_rec if r['sharpe'] > 0.5)}/{len(sharpe_rec)}")
    
    if avg_sharpe_rec < 0.5:
        print(f"\n[!] WARNING: Even best threshold only achieves Sharpe {avg_sharpe_rec:.2f}")
        print(f"    Recommendation: Consider alternative strategies or indicator adjustments")
    else:
        print(f"\n[OK] Recommendation: Use buy_threshold={best_threshold_sharpe}")
        print(f"    Expected performance: Sharpe {avg_sharpe_rec:.2f}, {np.mean([r['trades'] for r in sharpe_rec]):.0f} trades/year")
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "search_results" / "threshold_tuning.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'all_results': all_results,
            'summary_by_threshold': {
                str(t): {
                    'avg_sharpe': float(np.mean([r['sharpe'] for r in results])),
                    'avg_trades': float(np.mean([r['trades'] for r in results])),
                    'count': len(results)
                }
                for t, results in by_threshold.items()
            }
        }, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to {output_path}")


if __name__ == '__main__':
    main()
