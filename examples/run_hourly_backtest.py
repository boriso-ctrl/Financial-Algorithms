"""
Historical Backtest with Hourly Data

This script runs the VWAP + ATR strategy on 3 years of hourly data
and reports the Sharpe ratio.
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


def generate_hourly_data(start_date: str = '2021-01-01', 
                        end_date: str = '2024-01-01',
                        initial_price: float = 100.0,
                        volatility: float = 0.015) -> pd.DataFrame:
    """
    Generate synthetic hourly OHLCV data for 3 years.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    initial_price : float
        Starting price
    volatility : float
        Price volatility (standard deviation of returns)
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: timestamp, open, high, low, close, volume, session
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    data = []
    current_price = initial_price
    current_dt = start_dt
    
    # Generate hourly bars for 3 years (trading hours: 9:30 AM - 4:00 PM)
    while current_dt < end_dt:
        # Skip weekends
        if current_dt.weekday() >= 5:
            current_dt += timedelta(days=1)
            current_dt = current_dt.replace(hour=9, minute=30)
            continue
        
        # Trading hours: 9:30 AM to 4:00 PM (6.5 hours = ~6-7 bars per day)
        for hour_offset in [0, 1, 2, 3, 4, 5, 6]:
            bar_time = current_dt.replace(hour=9, minute=30) + timedelta(hours=hour_offset)
            
            # Stop at market close (4:00 PM)
            if bar_time.hour >= 16:
                break
            
            session_id = bar_time.date().strftime('%Y-%m-%d')
            
            # Generate returns with trending and mean reversion patterns
            base_return = np.random.normal(0, volatility / np.sqrt(252 / 6.5))
            
            # Add some autocorrelation (momentum)
            if len(data) > 0:
                prev_return = (data[-1]['close'] - data[-1]['open']) / data[-1]['open']
                momentum = prev_return * 0.3
            else:
                momentum = 0
            
            # Add weekly trends
            week_trend = np.sin(len(data) / (5 * 6.5)) * 0.001
            
            total_return = base_return + momentum + week_trend
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + total_return)
            
            # High and low with realistic spread
            bar_range = abs(close_price - open_price) * np.random.uniform(1.5, 3.0)
            if close_price > open_price:
                high_price = close_price + bar_range * 0.3
                low_price = open_price - bar_range * 0.1
            else:
                high_price = open_price + bar_range * 0.1
                low_price = close_price - bar_range * 0.3
            
            # Ensure OHLC relationships are valid
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume (higher during market open/close)
            if hour_offset in [0, 6]:  # First and last hour
                base_volume = np.random.uniform(500000, 1000000)
            else:
                base_volume = np.random.uniform(300000, 600000)
            
            volume = int(base_volume * (1 + abs(total_return) * 10))
            
            data.append({
                'timestamp': bar_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'session': session_id
            })
            
            current_price = close_price
        
        # Move to next day
        current_dt += timedelta(days=1)
        current_dt = current_dt.replace(hour=9, minute=30)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
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
    Main execution function for 3-year hourly backtest.
    """
    print("=" * 80)
    print("VWAP + ATR STRATEGY - 3-YEAR HOURLY BACKTEST")
    print("=" * 80)
    print()
    
    # Generate 3 years of hourly data
    print("[1/5] Generating 3 years of hourly data...")
    df = generate_hourly_data(
        start_date='2021-01-01',
        end_date='2024-01-01',
        initial_price=150.0,
        volatility=0.015
    )
    
    print(f"✓ Generated {len(df)} hourly bars")
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
        print(f"{'Trades Per Year':.<30} {len(trades) / 3:.2f}")
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
    print(f"SHARPE RATIO: {sharpe_ratio:.4f}")
    print("=" * 80)
    
    return sharpe_ratio


if __name__ == "__main__":
    sharpe = main()
