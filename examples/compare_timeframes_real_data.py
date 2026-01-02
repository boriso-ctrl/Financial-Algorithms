"""
Timeframe Comparison Analysis with Real Data

This script compares the VWAP + ATR strategy performance across different timeframes
using REAL market data (forex) to determine which has the highest Sharpe ratio.

Timeframes tested: 1-hour, 4-hour, daily (based on available data)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals
from backtest.intraday_backtest import run_intraday_backtest

# Configuration
FOREX_DATA_DIR = 'Financial-Algorithm Contents/Forex/Data'
FOREX_PAIRS = {
    'EURUSD': f'{FOREX_DATA_DIR}/EURUSD1H.csv',
    'GBPUSD': f'{FOREX_DATA_DIR}/GBPUSD1H.csv',
    'USDJPY': f'{FOREX_DATA_DIR}/USDJPY1H.csv',
    'AUDUSD': f'{FOREX_DATA_DIR}/AUDUSD1H.csv'
}


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


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample hourly data to different timeframes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hourly OHLCV data
    timeframe : str
        Target timeframe ('1H', '4H', '1D')
        
    Returns
    -------
    pd.DataFrame
        Resampled OHLCV data
    """
    # Resample OHLCV data
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(timeframe).first()
    resampled['high'] = df['high'].resample(timeframe).max()
    resampled['low'] = df['low'].resample(timeframe).min()
    resampled['close'] = df['close'].resample(timeframe).last()
    resampled['volume'] = df['volume'].resample(timeframe).sum()
    
    # Drop NaN rows
    resampled = resampled.dropna()
    
    # Add session column
    resampled['session'] = pd.to_datetime(resampled.index.date).astype(str)
    
    return resampled


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 6.5) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Annualize
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe


def test_pair_timeframe(pair_name: str, csv_path: str, timeframe: str, 
                        periods_per_year: int, verbose: bool = False) -> dict:
    """
    Test strategy on a specific pair and timeframe.
    
    Parameters
    ----------
    pair_name : str
        Currency pair name
    csv_path : str
        Path to CSV file
    timeframe : str
        Timeframe ('1H', '4H', '1D')
    periods_per_year : int
        Annualization factor
    verbose : bool
        Whether to print details
        
    Returns
    -------
    dict
        Performance metrics
    """
    try:
        # Load data
        df = load_forex_data(csv_path, num_years=3)
        
        if len(df) < 100:
            if verbose:
                print(f"  Insufficient data: {len(df)} bars")
            return None
        
        # Resample if needed
        if timeframe != '1H':
            df = resample_to_timeframe(df, timeframe)
            
            if len(df) < 50:
                if verbose:
                    print(f"  Insufficient data after resampling: {len(df)} bars")
                return None
        
        # Calculate indicators
        df = calculate_indicators(df, session_col='session', atr_period=14, 
                                 rsi_period=14, ema_period=20)
        
        # Detect regime
        df = detect_full_regime(df, atr_lookback=20)
        
        # Generate signals
        df = generate_signals(df, session_col='session')
        
        # Run backtest
        valid_mask = df['atr'].notna() & df['rsi'].notna() & df['ema'].notna()
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) < 50:
            if verbose:
                print(f"  Insufficient valid data: {len(df_valid)} bars")
            return None
        
        results = run_intraday_backtest(df_valid, initial_capital=100000, 
                                       position_size_pct=1.0)
        
        # Calculate Sharpe ratio
        equity_curve = results['equity_curve']
        returns = equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year)
        
        # Extract metrics
        total_trades = len(results['trades'])
        
        if total_trades == 0:
            return None
        
        win_rate = float(results['metrics']['Win Rate'].rstrip('%'))
        total_return = results['metrics']['Total Return']
        max_dd = results['metrics']['Max Drawdown']
        profit_factor = results['metrics']['Profit Factor']
        
        if verbose:
            print(f"  ✓ Sharpe: {sharpe_ratio:.4f}, Return: {total_return}, Trades: {total_trades}")
        
        return {
            'pair': pair_name,
            'timeframe': timeframe,
            'bars': len(df_valid),
            'trades': total_trades,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor
        }
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def main():
    """
    Compare strategy performance across timeframes with real data.
    """
    print("=" * 80)
    print("TIMEFRAME COMPARISON ANALYSIS - REAL MARKET DATA")
    print("VWAP + ATR Strategy - Forex Data (2017-2020)")
    print("=" * 80)
    print()
    
    # Define timeframes and their annualization factors
    timeframes = [
        ('1H', 252 * 6.5, 'Hourly'),
        ('4H', 252 * 6.5 / 4, '4-Hour'),
        ('1D', 252, 'Daily')
    ]
    
    results = []
    
    print("Testing strategy on multiple timeframes...")
    print()
    
    for tf_code, periods_per_year, tf_name in timeframes:
        print(f"[{tf_name} Timeframe]")
        print("-" * 80)
        
        for pair_name, csv_path in FOREX_PAIRS.items():
            print(f"  {pair_name}...", end=' ')
            
            result = test_pair_timeframe(
                pair_name, csv_path, tf_code, 
                int(periods_per_year), verbose=True
            )
            
            if result:
                results.append(result)
            else:
                print("  ✗ Skipped (insufficient data or no trades)")
        
        print()
    
    # Display results
    if len(results) == 0:
        print("No results to display.")
        return
    
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    df_results = pd.DataFrame(results)
    
    # Overall best by Sharpe
    print("Top 10 Configurations by Sharpe Ratio:")
    print("-" * 80)
    print(f"{'Pair':<10} {'Timeframe':<12} {'Sharpe':<10} {'Return':<12} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 80)
    
    top_results = df_results.nlargest(10, 'sharpe_ratio')
    for _, row in top_results.iterrows():
        print(f"{row['pair']:<10} {row['timeframe']:<12} {row['sharpe_ratio']:<10.4f} "
              f"{row['total_return']:<12} {row['win_rate']:<9.2f}% {row['trades']:<8}")
    
    print()
    
    # Average by timeframe
    print("Average Performance by Timeframe:")
    print("-" * 80)
    print(f"{'Timeframe':<12} {'Avg Sharpe':<12} {'Pairs Tested':<15} {'Avg Trades':<12}")
    print("-" * 80)
    
    for tf_code, _, tf_name in timeframes:
        tf_data = df_results[df_results['timeframe'] == tf_code]
        if len(tf_data) > 0:
            avg_sharpe = tf_data['sharpe_ratio'].mean()
            num_pairs = len(tf_data)
            avg_trades = tf_data['trades'].mean()
            print(f"{tf_name:<12} {avg_sharpe:<12.4f} {num_pairs:<15} {avg_trades:<12.1f}")
    
    print()
    
    # Average by pair
    print("Average Performance by Currency Pair:")
    print("-" * 80)
    print(f"{'Pair':<12} {'Avg Sharpe':<12} {'Best TF':<15} {'Best Sharpe':<12}")
    print("-" * 80)
    
    for pair in FOREX_PAIRS.keys():
        pair_data = df_results[df_results['pair'] == pair]
        if len(pair_data) > 0:
            avg_sharpe = pair_data['sharpe_ratio'].mean()
            best_row = pair_data.loc[pair_data['sharpe_ratio'].idxmax()]
            print(f"{pair:<12} {avg_sharpe:<12.4f} {best_row['timeframe']:<15} "
                  f"{best_row['sharpe_ratio']:<12.4f}")
    
    print()
    print("=" * 80)
    
    # Highlight overall best
    best = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    print("BEST OVERALL CONFIGURATION:")
    print(f"  Pair: {best['pair']}")
    print(f"  Timeframe: {best['timeframe']}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.4f}")
    print(f"  Total Return: {best['total_return']}")
    print(f"  Win Rate: {best['win_rate']:.2f}%")
    print(f"  Max Drawdown: {best['max_drawdown']}")
    print(f"  Profit Factor: {best['profit_factor']}")
    print(f"  Total Trades: {best['trades']}")
    
    print()
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    
    # Compare timeframes
    hourly_avg = df_results[df_results['timeframe'] == '1H']['sharpe_ratio'].mean()
    four_hour_avg = df_results[df_results['timeframe'] == '4H']['sharpe_ratio'].mean()
    daily_avg = df_results[df_results['timeframe'] == '1D']['sharpe_ratio'].mean()
    
    print(f"• Hourly (1H): Average Sharpe = {hourly_avg:.4f}")
    if not pd.isna(four_hour_avg):
        print(f"• 4-Hour (4H): Average Sharpe = {four_hour_avg:.4f}")
    if not pd.isna(daily_avg):
        print(f"• Daily (1D): Average Sharpe = {daily_avg:.4f}")
    
    print()
    
    # Find best timeframe
    tf_avgs = df_results.groupby('timeframe')['sharpe_ratio'].mean().sort_values(ascending=False)
    if len(tf_avgs) > 0:
        best_tf = tf_avgs.index[0]
        best_tf_sharpe = tf_avgs.iloc[0]
        print(f"✓ Best performing timeframe: {best_tf} (Avg Sharpe: {best_tf_sharpe:.4f})")
    
    # Find best pair
    pair_avgs = df_results.groupby('pair')['sharpe_ratio'].mean().sort_values(ascending=False)
    if len(pair_avgs) > 0:
        best_pair = pair_avgs.index[0]
        best_pair_sharpe = pair_avgs.iloc[0]
        print(f"✓ Best performing pair: {best_pair} (Avg Sharpe: {best_pair_sharpe:.4f})")
    
    print()
    print("NOTE: These results are based on REAL forex market data (2017-2020).")
    print("Performance may vary in different market conditions and time periods.")
    print("=" * 80)


if __name__ == "__main__":
    main()
