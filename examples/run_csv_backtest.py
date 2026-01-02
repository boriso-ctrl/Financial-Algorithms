"""
CSV-Based Real Historical Data Backtest

This script runs the VWAP + ATR strategy on real historical market data
loaded from CSV files. Users can provide their own data from any source.

CSV Format Required:
- Columns: timestamp (or datetime/date), open, high, low, close, volume
- Timestamp should be parseable datetime format
- OHLCV values should be numeric

Example usage:
    python run_csv_backtest.py --csv data/SPY_5min.csv
    python run_csv_backtest.py --csv data/SPY_5min.csv --interval 5m
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals
from backtest.intraday_backtest import run_intraday_backtest


def load_csv_data(csv_path: str, timestamp_col: str = None) -> pd.DataFrame:
    """
    Load historical data from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    timestamp_col : str
        Name of timestamp column (auto-detect if None)
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with session column
    """
    print(f"Loading data from: {csv_path}")
    
    try:
        # Try to read CSV
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Columns: {', '.join(df.columns)}")
        
        # Find timestamp column
        timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Datetime', 'Date', 'Time', 'Timestamp']
        
        if timestamp_col:
            time_col = timestamp_col
        else:
            time_col = None
            for col in timestamp_cols:
                if col in df.columns:
                    time_col = col
                    break
        
        if not time_col:
            print(f"Error: Could not find timestamp column. Available columns: {df.columns.tolist()}")
            print(f"Expected one of: {timestamp_cols}")
            return None
        
        # Parse timestamp
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        
        # Standardize column names (case-insensitive)
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                rename_map[col] = col_lower
        
        df = df.rename(columns=rename_map)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Keep only OHLCV columns
        df = df[required_cols].copy()
        
        # Add session column
        df['session'] = df.index.date.astype(str)
        
        # Sort by timestamp
        df = df.sort_index()
        
        print(f"✓ Processed data:")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Sessions: {df['session'].nunique()}")
        print(f"  Trading days: {(df.index[-1] - df.index[0]).days}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe


def run_csv_backtest(csv_path: str,
                     interval: str = '5m',
                     initial_capital: float = 100000):
    """
    Run backtest on CSV data.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with OHLCV data
    interval : str
        Data interval ('5m', '15m', '30m', '60m', '1d')
    initial_capital : float
        Starting capital
        
    Returns
    -------
    dict
        Backtest results
    """
    print("=" * 80)
    print(f"CSV-BASED REAL HISTORICAL DATA BACKTEST")
    print("=" * 80)
    print()
    
    # Load data
    print("[1/5] Loading CSV data...")
    df = load_csv_data(csv_path)
    
    if df is None or len(df) == 0:
        print("Error: Could not load data")
        return None
    
    print()
    
    # Calculate indicators
    print("[2/5] Calculating indicators...")
    df = calculate_indicators(df, session_col='session')
    print("✓ Indicators calculated")
    print()
    
    # Detect regime
    print("[3/5] Detecting market regimes...")
    df = detect_full_regime(df)
    
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
    
    # Drop NaN rows
    valid_mask = df['atr'].notna() & df['rsi'].notna() & df['ema'].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("Error: No valid data after indicator calculation")
        return None
    
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
    
    # Determine periods per year based on interval
    if interval == '5m':
        bars_per_day = 78
    elif interval == '15m':
        bars_per_day = 26
    elif interval == '30m':
        bars_per_day = 13
    elif interval == '60m' or interval == '1h':
        bars_per_day = 6.5
    elif interval == '1d':
        bars_per_day = 1
    else:
        # Try to infer from data
        days_unique = df_valid['session'].nunique()
        total_bars = len(df_valid)
        bars_per_day = total_bars / days_unique if days_unique > 0 else 78
    
    periods_per_year = 252 * bars_per_day
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year)
    
    # Display results
    print("=" * 80)
    print("RESULTS")
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
        print("TRADE STATISTICS")
        print("-" * 80)
        print(f"{'Total Trades':.<30} {len(trades)}")
        
        # Calculate trading period
        days_traded = (df_valid.index[-1] - df_valid.index[0]).days
        if days_traded > 0:
            print(f"{'Trades Per Day':.<30} {len(trades) / days_traded:.2f}")
        
        print(f"{'Average Trade Duration':.<30} {(trades['exit_time'] - trades['entry_time']).mean()}")
        print()
        
        print("RETURN DISTRIBUTION")
        print("-" * 80)
        returns_pct = trades['pnl_pct'] * 100
        print(f"{'Mean':.<30} {returns_pct.mean():.4f}%")
        print(f"{'Median':.<30} {returns_pct.median():.4f}%")
        print(f"{'Std Dev':.<30} {returns_pct.std():.4f}%")
        print(f"{'Best Trade':.<30} {returns_pct.max():.4f}%")
        print(f"{'Worst Trade':.<30} {returns_pct.min():.4f}%")
        print()
        
        # Regime breakdown
        print("PERFORMANCE BY REGIME")
        print("-" * 80)
        from backtest.intraday_backtest import analyze_regime_performance
        regime_perf = analyze_regime_performance(df_valid, trades)
        if len(regime_perf) > 0:
            print(regime_perf.to_string(index=False))
        print()
        
        # Sample trades
        print("SAMPLE TRADES (First 5)")
        print("-" * 80)
        sample = trades.head(5)[['entry_time', 'exit_time', 'direction', 
                                 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason']]
        print(sample.to_string(index=False))
        print()
    
    print("=" * 80)
    print(f"SHARPE RATIO: {sharpe_ratio:.4f}")
    print("=" * 80)
    print()
    
    print("IMPORTANT NOTES:")
    print("-" * 80)
    print("• This backtest uses REAL historical market data from CSV")
    print("• Performance reflects actual market conditions")
    print(f"• Data file: {csv_path}")
    print(f"• Interval: {interval}")
    print("• No slippage or commissions included")
    print("• Add transaction costs for realistic expectations")
    print("=" * 80)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'results': results,
        'df': df_valid
    }


def main():
    """
    Main execution function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Backtest VWAP + ATR strategy on CSV historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Format Requirements:
  - Must have timestamp column (or datetime/date)
  - Must have OHLCV columns: open, high, low, close, volume
  - Column names are case-insensitive

Example CSV format:
  timestamp,open,high,low,close,volume
  2024-01-02 09:30:00,100.5,101.2,100.3,101.0,1000000
  2024-01-02 09:35:00,101.0,101.5,100.8,101.2,950000

Where to get data:
  - Yahoo Finance: download historical data
  - Interactive Brokers: export historical data
  - Polygon.io: API for historical data
  - AlphaVantage: API for historical data
        """
    )
    
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--interval', type=str, default='5m', help='Data interval (5m, 15m, 30m, 60m, 1d)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--timestamp-col', type=str, default=None, help='Name of timestamp column (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    run_csv_backtest(
        csv_path=args.csv,
        interval=args.interval,
        initial_capital=args.capital
    )


if __name__ == "__main__":
    main()
