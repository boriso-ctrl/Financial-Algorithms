"""
Generate Sample Intraday Data

This utility generates synthetic 5-minute OHLCV data for testing
the VWAP + ATR strategy without requiring external data sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_intraday_data(start_date: str = '2024-01-01', 
                           num_days: int = 20,
                           initial_price: float = 100.0,
                           volatility: float = 0.02) -> pd.DataFrame:
    """
    Generate synthetic 5-minute OHLCV data.
    
    Creates realistic-looking intraday data with:
    - 5-minute bars
    - Daily sessions (9:30 AM - 4:00 PM EST)
    - Realistic OHLCV relationships
    - Trends and mean reversion patterns
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    num_days : int
        Number of trading days to generate
    initial_price : float
        Starting price
    volatility : float
        Price volatility (standard deviation of returns)
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: timestamp, open, high, low, close, volume, session
    """
    bars_per_day = 78  # 6.5 hours * 60 minutes / 5 minutes = 78 bars
    
    data = []
    current_price = initial_price
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    
    for day in range(num_days):
        # Skip weekends (simplified)
        session_date = start_dt + timedelta(days=day)
        if session_date.weekday() >= 5:  # Saturday or Sunday
            continue
        
        session_id = session_date.strftime('%Y-%m-%d')
        
        # Generate a daily trend (or mean reversion)
        daily_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        
        for bar in range(bars_per_day):
            # Timestamp for this bar (starting at 9:30 AM)
            bar_time = session_date.replace(hour=9, minute=30) + timedelta(minutes=bar * 5)
            
            # Generate returns with some autocorrelation
            base_return = np.random.normal(0, volatility / np.sqrt(bars_per_day))
            
            # Add daily trend component
            trend_component = daily_trend * 0.0005
            
            # Add mean reversion component (price tends to revert to session average)
            # This creates more realistic intraday patterns
            if bar > 10:
                session_avg = np.mean([d['close'] for d in data[-10:]])
                mean_reversion = (session_avg - current_price) / current_price * 0.1
            else:
                mean_reversion = 0
            
            total_return = base_return + trend_component + mean_reversion
            
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
            
            # Generate volume (higher at open and close)
            if bar < 12 or bar > 65:  # First hour and last hour
                base_volume = np.random.uniform(80000, 150000)
            else:
                base_volume = np.random.uniform(40000, 80000)
            
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
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def add_realistic_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realistic intraday patterns to the data.
    
    - Morning volatility
    - Lunch hour consolidation
    - End-of-day movements
    
    Parameters
    ----------
    df : pd.DataFrame
        Generated intraday data
        
    Returns
    -------
    pd.DataFrame
        Data with enhanced patterns
    """
    result = df.copy()
    
    # Add some trending sessions and some choppy sessions
    sessions = result['session'].unique()
    
    for i, session in enumerate(sessions):
        session_mask = result['session'] == session
        session_data = result[session_mask]
        
        # Every 3rd session: create a strong trend
        if i % 3 == 0:
            trend_direction = 1 if i % 2 == 0 else -1
            trend_strength = np.linspace(0, trend_direction * 0.03, len(session_data))
            result.loc[session_mask, 'close'] *= (1 + trend_strength)
        
        # Every 4th session: create choppy/rotational behavior
        elif i % 4 == 0:
            # Add oscillation around mean
            session_mean = session_data['close'].mean()
            oscillation = np.sin(np.linspace(0, 4 * np.pi, len(session_data))) * session_mean * 0.01
            result.loc[session_mask, 'close'] += oscillation
    
    # Recalculate high and low to maintain consistency
    for idx in result.index:
        result.loc[idx, 'high'] = max(result.loc[idx, 'high'], result.loc[idx, 'close'])
        result.loc[idx, 'low'] = min(result.loc[idx, 'low'], result.loc[idx, 'close'])
    
    return result


if __name__ == '__main__':
    # Generate sample data
    print("Generating sample intraday data...")
    df = generate_intraday_data(num_days=30)
    df = add_realistic_patterns(df)
    
    print(f"\nGenerated {len(df)} bars across {df['session'].nunique()} sessions")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nLast few rows:")
    print(df.tail(10))
    
    # Save to CSV
    output_file = 'sample_intraday_data.csv'
    df.to_csv(output_file)
    print(f"\nSaved to {output_file}")
