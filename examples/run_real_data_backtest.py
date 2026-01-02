"""
Real Data Historical Backtest

This script runs the VWAP + ATR strategy on real historical data
from Yahoo Finance and reports the Sharpe ratio for comparison
with synthetic data results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals
from backtest.intraday_backtest import run_intraday_backtest


def load_real_data_from_csv(csv_path: str, 
                            num_years: int = 3) -> pd.DataFrame:
    """
    Load real historical intraday data from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing OHLCV data
    num_years : int
        Number of recent years to use (default: 3)
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with session column
    """
    print(f"Loading data from {csv_path}...")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Parse timestamp column (handle different formats)
    if 'Local time' in df.columns:
        # Remove timezone info and parse as datetime
        df['timestamp'] = df['Local time'].str.replace(r' GMT[+-]\d{4}', '', regex=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
    elif 'Date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise ValueError("Could not find timestamp column in CSV")
    
    # Rename columns to lowercase for consistency
    df.columns = [col.lower() if col != 'timestamp' else col for col in df.columns]
    
    # Keep only needed columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols].copy()
    
    # Remove any NaN or duplicate timestamps
    df = df.dropna(subset=['timestamp'])
    df = df[df['timestamp'].notna()]
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Remove any remaining NaN values in price/volume
    df = df.dropna()
    
    # Remove duplicate index values (keep first)
    df = df[~df.index.duplicated(keep='first')]
    
    # Get only the most recent N years
    if len(df) > 0:
        cutoff_date = df.index[-1] - pd.DateOffset(years=num_years)
        df = df[df.index >= cutoff_date]
    
    # Add session column (date string for daily sessions)
    df['session'] = pd.to_datetime(df.index.date).astype(str)
    
    # Ensure timestamp is the index name
    df.index.name = 'timestamp'
    
    return df


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 6.5) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of trading periods per year (for hourly: 252 days * 6.5 hours)
        
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


def main():
    """
    Main execution function for real data backtest.
    """
    print("=" * 80)
    print("VWAP + ATR STRATEGY - REAL DATA HISTORICAL BACKTEST")
    print("=" * 80)
    print()
    
    # Configuration
    csv_path = 'Financial-Algorithm Contents/Forex/Data/EURUSD1H.csv'
    num_years = 3  # Use 3 years of data for consistency with synthetic test
    
    # Load real data from CSV
    print("[1/5] Loading real historical data from CSV...")
    try:
        df = load_real_data_from_csv(csv_path=csv_path, num_years=num_years)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print(f"✓ Loaded {len(df)} hourly bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Sessions: {df['session'].nunique()}")
    print(f"  Years: {(df.index[-1] - df.index[0]).days / 365.25:.2f}")
    print()
    
    # Calculate indicators
    print("[2/5] Calculating indicators...")
    df = calculate_indicators(
        df,
        session_col='session',
        atr_period=14,
        rsi_period=14,
        ema_period=20
    )
    print("✓ Indicators calculated")
    print()
    
    # Detect regime
    print("[3/5] Detecting market regimes...")
    df = detect_full_regime(df, atr_lookback=20)
    
    regime_counts = df['regime'].value_counts()
    print(f"  Trend bars: {regime_counts.get('trend', 0)} ({regime_counts.get('trend', 0)/len(df)*100:.1f}%)")
    print(f"  Rotational bars: {regime_counts.get('rotational', 0)} ({regime_counts.get('rotational', 0)/len(df)*100:.1f}%)")
    print()
    
    # Generate signals
    print("[4/5] Generating trading signals...")
    df = generate_signals(df, session_col='session')
    
    signals = df[df['signal'] != 'none']
    print(f"  Total signals: {len(signals)}")
    print(f"  Long signals: {len(signals[signals['signal'] == 'long'])}")
    print(f"  Short signals: {len(signals[signals['signal'] == 'short'])}")
    print()
    
    # Run backtest
    print("[5/5] Running backtest...")
    initial_capital = 100000
    
    # Drop NaN rows
    valid_mask = df['atr'].notna() & df['rsi'].notna() & df['ema'].notna()
    df_valid = df[valid_mask].copy()
    
    results = run_intraday_backtest(
        df_valid,
        initial_capital=initial_capital,
        position_size_pct=1.0
    )
    
    print("✓ Backtest complete")
    print()
    
    # Calculate Sharpe ratio
    equity_curve = results['equity_curve']
    returns = equity_curve['equity'].pct_change().dropna()
    
    # For hourly data: 252 trading days * 6.5 hours per day
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year=252 * 6.5)
    
    # Display results
    print("=" * 80)
    print("RESULTS (REAL DATA)")
    print("=" * 80)
    print()
    
    print("PERFORMANCE METRICS")
    print("-" * 80)
    for key, value in results['metrics'].items():
        print(f"{key:.<30} {value}")
    
    print()
    print(f"{'Sharpe Ratio (Annualized)':.<30} {sharpe_ratio:.4f}")
    print()
    
    # Additional statistics
    if len(results['trades']) > 0:
        trades = results['trades']
        years = (df_valid.index[-1] - df_valid.index[0]).days / 365.25
        
        print("TRADE STATISTICS")
        print("-" * 80)
        print(f"{'Total Trades':.<30} {len(trades)}")
        print(f"{'Trades Per Year':.<30} {len(trades) / years:.2f}")
        print(f"{'Average Trade Duration':.<30} {(trades['exit_time'] - trades['entry_time']).mean()}")
        print()
        
        print("RETURN DISTRIBUTION")
        print("-" * 80)
        returns_pct = trades['pnl_pct'] * 100
        print(f"{'Mean':.<30} {returns_pct.mean():.4f}%")
        print(f"{'Median':.<30} {returns_pct.median():.4f}%")
        print(f"{'Std Dev':.<30} {returns_pct.std():.4f}%")
        print(f"{'Skewness':.<30} {returns_pct.skew():.4f}")
        print(f"{'Kurtosis':.<30} {returns_pct.kurtosis():.4f}")
        print()
    
    print("=" * 80)
    print(f"SHARPE RATIO (REAL DATA): {sharpe_ratio:.4f}")
    print("=" * 80)
    print()
    print(f"Data source: EUR/USD Hourly (Historical Forex Data)")
    print(f"Period: {num_years} years")
    print()
    
    return sharpe_ratio


if __name__ == "__main__":
    sharpe = main()
