"""
Real vs Synthetic Data Comparison

This script runs the VWAP + ATR strategy on both real and synthetic data
to compare performance and validate the strategy's effectiveness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals
from backtest.intraday_backtest import run_intraday_backtest


def load_forex_data(csv_path: str, num_years: int = 3) -> pd.DataFrame:
    """Load forex data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    df['timestamp'] = df['Local time'].str.replace(r' GMT[+-]\d{4}', '', regex=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
    
    # Rename columns
    df.columns = [col.lower() if col != 'timestamp' else col for col in df.columns]
    
    # Keep only needed columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols].copy()
    
    # Remove NaN
    df = df.dropna(subset=['timestamp'])
    df = df[df['timestamp'].notna()]
    
    # Set index
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    # Get recent years
    if len(df) > 0:
        cutoff_date = df.index[-1] - pd.DateOffset(years=num_years)
        df = df[df.index >= cutoff_date]
    
    # Add session
    df['session'] = pd.to_datetime(df.index.date).astype(str)
    df.index.name = 'timestamp'
    
    return df


def generate_synthetic_data(num_years: int = 3, initial_price: float = 1.18) -> pd.DataFrame:
    """Generate synthetic hourly data matching the real data period."""
    start_date = '2017-09-05'
    end_date = '2020-09-05'
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    data = []
    current_price = initial_price
    current_dt = start_dt
    
    while current_dt < end_dt:
        for hour_offset in range(24):
            bar_time = current_dt + timedelta(hours=hour_offset)
            session_id = bar_time.date().strftime('%Y-%m-%d')
            
            # Generate returns
            base_return = np.random.normal(0, 0.0015)
            
            if len(data) > 0:
                prev_return = (data[-1]['close'] - data[-1]['open']) / data[-1]['open']
                momentum = prev_return * 0.3
            else:
                momentum = 0
            
            week_trend = np.sin(len(data) / 168) * 0.0005
            total_return = base_return + momentum + week_trend
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + total_return)
            
            bar_range = abs(close_price - open_price) * np.random.uniform(1.5, 3.0)
            if close_price > open_price:
                high_price = close_price + bar_range * 0.3
                low_price = open_price - bar_range * 0.1
            else:
                high_price = open_price + bar_range * 0.1
                low_price = close_price - bar_range * 0.3
            
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            volume = int(np.random.uniform(200000, 400000))
            
            data.append({
                'timestamp': bar_time,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume,
                'session': session_id
            })
            
            current_price = close_price
        
        current_dt += timedelta(days=1)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 24) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe


def run_strategy_test(df: pd.DataFrame, name: str) -> dict:
    """Run strategy on given data and return metrics."""
    # Calculate indicators
    df = calculate_indicators(df, session_col='session', atr_period=14, rsi_period=14, ema_period=20)
    
    # Detect regime
    df = detect_full_regime(df, atr_lookback=20)
    
    # Generate signals
    df = generate_signals(df, session_col='session')
    
    # Run backtest
    valid_mask = df['atr'].notna() & df['rsi'].notna() & df['ema'].notna()
    df_valid = df[valid_mask].copy()
    
    results = run_intraday_backtest(df_valid, initial_capital=100000, position_size_pct=1.0)
    
    # Calculate Sharpe
    equity_curve = results['equity_curve']
    returns = equity_curve['equity'].pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year=252 * 24)
    
    # Compile metrics
    metrics = {
        'name': name,
        'bars': len(df_valid),
        'years': (df_valid.index[-1] - df_valid.index[0]).days / 365.25,
        'total_trades': results['metrics']['Total Trades'],
        'win_rate': float(results['metrics']['Win Rate'].rstrip('%')),
        'total_return': results['metrics']['Total Return'],
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': results['metrics']['Max Drawdown'],
        'profit_factor': results['metrics']['Profit Factor']
    }
    
    return metrics


def main():
    """Main execution function."""
    print("=" * 80)
    print("VWAP + ATR STRATEGY - REAL vs SYNTHETIC DATA COMPARISON")
    print("=" * 80)
    print()
    
    # Test forex pairs
    forex_pairs = [
        ('EURUSD', 'Financial-Algorithm Contents/Forex/Data/EURUSD1H.csv'),
        ('GBPUSD', 'Financial-Algorithm Contents/Forex/Data/GBPUSD1H.csv'),
        ('USDJPY', 'Financial-Algorithm Contents/Forex/Data/USDJPY1H.csv'),
        ('AUDUSD', 'Financial-Algorithm Contents/Forex/Data/AUDUSD1H.csv')
    ]
    
    results = []
    
    # Test real data
    print("Testing REAL data...")
    print("-" * 80)
    for pair_name, csv_path in forex_pairs:
        try:
            print(f"\n[{pair_name}] Loading data...")
            df = load_forex_data(csv_path, num_years=3)
            
            if len(df) < 1000:
                print(f"  Skipped (insufficient data: {len(df)} bars)")
                continue
            
            print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            print(f"  Running backtest...")
            
            metrics = run_strategy_test(df, f"{pair_name} (Real)")
            results.append(metrics)
            
            print(f"  ✓ Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test synthetic data
    print("\n" + "=" * 80)
    print("Testing SYNTHETIC data...")
    print("-" * 80)
    
    try:
        print("\nGenerating synthetic hourly data...")
        df_synthetic = generate_synthetic_data(num_years=3, initial_price=1.18)
        print(f"  Generated {len(df_synthetic)} bars from {df_synthetic.index[0]} to {df_synthetic.index[-1]}")
        print(f"  Running backtest...")
        
        metrics = run_strategy_test(df_synthetic, "Synthetic")
        results.append(metrics)
        
        print(f"  ✓ Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Display comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        
        print("PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Instrument':<20} {'Sharpe':>8} {'Return':>10} {'Win Rate':>10} {'Trades':>8} {'Max DD':>8}")
        print("-" * 80)
        
        for _, row in df_results.iterrows():
            print(f"{row['name']:<20} {row['sharpe_ratio']:>8.4f} {row['total_return']:>10} {row['win_rate']:>9.2f}% {row['total_trades']:>8} {row['max_drawdown']:>8}")
        
        print("-" * 80)
        
        # Calculate averages
        real_data = df_results[df_results['name'].str.contains('Real')]
        synthetic_data = df_results[df_results['name'].str.contains('Synthetic')]
        
        if len(real_data) > 0:
            print(f"\nREAL DATA AVERAGE:")
            print(f"  Sharpe Ratio: {real_data['sharpe_ratio'].mean():.4f}")
            print(f"  Win Rate: {real_data['win_rate'].mean():.2f}%")
            print(f"  Avg Trades/Year: {(real_data['total_trades'] / real_data['years']).mean():.2f}")
        
        if len(synthetic_data) > 0:
            print(f"\nSYNTHETIC DATA:")
            print(f"  Sharpe Ratio: {synthetic_data['sharpe_ratio'].mean():.4f}")
            print(f"  Win Rate: {synthetic_data['win_rate'].mean():.2f}%")
            print(f"  Avg Trades/Year: {(synthetic_data['total_trades'] / synthetic_data['years']).mean():.2f}")
        
        print()
        print("=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)
        print()
        
        if len(real_data) > 0 and len(synthetic_data) > 0:
            real_avg_sharpe = real_data['sharpe_ratio'].mean()
            synth_avg_sharpe = synthetic_data['sharpe_ratio'].mean()
            
            print(f"✓ Real data Sharpe ratio: {real_avg_sharpe:.4f}")
            print(f"✓ Synthetic data Sharpe ratio: {synth_avg_sharpe:.4f}")
            print(f"✓ Difference: {abs(synth_avg_sharpe - real_avg_sharpe):.4f}")
            print()
            
            if real_avg_sharpe < 1.0:
                print("⚠️  Real data shows lower-than-expected Sharpe ratio.")
                print("    This is normal - real markets are more challenging than synthetic data.")
                print("    Consider: parameter optimization, transaction costs, different instruments.")
            else:
                print("✓ Real data shows promising results!")
            
        print()
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
