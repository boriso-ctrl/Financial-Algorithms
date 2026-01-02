"""
Timeframe Comparison Analysis

This script compares the VWAP + ATR strategy performance across different timeframes
to determine which has the highest Sharpe ratio.

Timeframes tested: 5-minute, 15-minute, 30-minute, 1-hour, 2-hour, 4-hour
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


def generate_data_for_timeframe(timeframe_minutes: int, 
                                start_date: str = '2021-01-01', 
                                end_date: str = '2024-01-01',
                                initial_price: float = 150.0,
                                volatility: float = 0.015) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for specified timeframe.
    
    Parameters
    ----------
    timeframe_minutes : int
        Timeframe in minutes (e.g., 5, 15, 60)
    start_date : str
        Start date
    end_date : str
        End date
    initial_price : float
        Starting price
    volatility : float
        Base volatility (adjusted per timeframe)
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with session column
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    data = []
    current_price = initial_price
    current_dt = start_dt
    
    # Adjust volatility for timeframe (longer timeframes have higher volatility per bar)
    timeframe_vol = volatility * np.sqrt(timeframe_minutes / 5)
    
    # Trading hours per day
    trading_minutes = 6.5 * 60  # 9:30 AM to 4:00 PM
    bars_per_day = int(trading_minutes / timeframe_minutes)
    
    while current_dt < end_dt:
        # Skip weekends
        if current_dt.weekday() >= 5:
            current_dt += timedelta(days=1)
            current_dt = current_dt.replace(hour=9, minute=30)
            continue
        
        session_id = current_dt.date().strftime('%Y-%m-%d')
        
        # Generate bars for this day
        for bar_idx in range(bars_per_day):
            bar_time = current_dt.replace(hour=9, minute=30) + timedelta(minutes=bar_idx * timeframe_minutes)
            
            # Stop at market close
            if bar_time.hour >= 16:
                break
            
            # Generate returns with trends and mean reversion
            base_return = np.random.normal(0, timeframe_vol / np.sqrt(252))
            
            # Add momentum from previous bar
            if len(data) > 0:
                prev_return = (data[-1]['close'] - data[-1]['open']) / data[-1]['open']
                momentum = prev_return * 0.3
            else:
                momentum = 0
            
            # Add longer-term trend
            trend = np.sin(len(data) / (bars_per_day * 5)) * 0.001
            
            total_return = base_return + momentum + trend
            
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
            
            # Volume (higher for longer timeframes)
            base_volume = np.random.uniform(100000, 500000) * (timeframe_minutes / 5)
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


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe


def test_timeframe(timeframe_minutes: int, verbose: bool = False):
    """
    Test strategy on a specific timeframe.
    
    Parameters
    ----------
    timeframe_minutes : int
        Timeframe in minutes
    verbose : bool
        Whether to print detailed output
        
    Returns
    -------
    dict
        Performance metrics including Sharpe ratio
    """
    # Generate data
    df = generate_data_for_timeframe(timeframe_minutes)
    
    # Calculate indicators
    df = calculate_indicators(df, session_col='session')
    
    # Detect regime
    df = detect_full_regime(df)
    
    # Generate signals
    df = generate_signals(df, session_col='session')
    
    # Run backtest
    valid_mask = df['atr'].notna() & df['rsi'].notna() & df['ema'].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        return None
    
    results = run_intraday_backtest(df_valid, initial_capital=100000, position_size_pct=1.0)
    
    # Calculate Sharpe ratio
    equity_curve = results['equity_curve']
    returns = equity_curve['equity'].pct_change().dropna()
    
    # Periods per year varies by timeframe
    trading_days = 252
    bars_per_day = (6.5 * 60) / timeframe_minutes
    periods_per_year = trading_days * bars_per_day
    
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year)
    
    # Extract key metrics
    total_trades = len(results['trades'])
    total_return = float(results['metrics']['Total Return'].strip('%')) / 100 if total_trades > 0 else 0
    win_rate = float(results['metrics']['Win Rate'].strip('%')) / 100 if total_trades > 0 else 0
    max_dd = float(results['metrics']['Max Drawdown'].strip('%')) / 100 if total_trades > 0 else 0
    
    if verbose:
        print(f"\n{timeframe_minutes}-minute timeframe:")
        print(f"  Bars: {len(df_valid)}")
        print(f"  Signals: {len(df_valid[df_valid['signal'] != 'none'])}")
        print(f"  Trades: {total_trades}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Max Drawdown: {max_dd:.2%}")
    
    return {
        'timeframe': f"{timeframe_minutes}min",
        'bars': len(df_valid),
        'signals': len(df_valid[df_valid['signal'] != 'none']),
        'trades': total_trades,
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'profit_factor': float(results['metrics']['Profit Factor']) if total_trades > 0 else 0
    }


def main():
    """
    Compare strategy performance across multiple timeframes.
    """
    print("=" * 80)
    print("TIMEFRAME COMPARISON ANALYSIS")
    print("VWAP + ATR Strategy - 3 Years of Data")
    print("=" * 80)
    print()
    
    print("Testing strategy on multiple timeframes...")
    print("This may take a few minutes...")
    print()
    
    # Test different timeframes
    timeframes = [5, 15, 30, 60, 120, 240]  # 5min, 15min, 30min, 1hr, 2hr, 4hr
    results = []
    
    for tf in timeframes:
        print(f"Testing {tf}-minute timeframe...", end=' ')
        try:
            result = test_timeframe(tf, verbose=False)
            if result:
                results.append(result)
                print(f"✓ Sharpe: {result['sharpe_ratio']:.4f}")
            else:
                print("✗ Insufficient data")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Create comparison table
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('sharpe_ratio', ascending=False)
        
        print("Performance by Timeframe (Sorted by Sharpe Ratio):")
        print("-" * 80)
        print(f"{'Timeframe':<12} {'Sharpe':<10} {'Return':<12} {'Win Rate':<12} {'Max DD':<12} {'Trades':<8}")
        print("-" * 80)
        
        for _, row in df_results.iterrows():
            print(f"{row['timeframe']:<12} "
                  f"{row['sharpe_ratio']:<10.4f} "
                  f"{row['total_return']:<12.2%} "
                  f"{row['win_rate']:<12.2%} "
                  f"{row['max_drawdown']:<12.2%} "
                  f"{row['trades']:<8.0f}")
        
        print()
        print("=" * 80)
        
        # Highlight best timeframe
        best = df_results.iloc[0]
        print(f"BEST TIMEFRAME: {best['timeframe']}")
        print(f"  Sharpe Ratio: {best['sharpe_ratio']:.4f}")
        print(f"  Total Return: {best['total_return']:.2%}")
        print(f"  Win Rate: {best['win_rate']:.2%}")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Max Drawdown: {best['max_drawdown']:.2%}")
        print(f"  Total Trades: {best['trades']:.0f}")
        print("=" * 80)
        
        # Additional insights
        print()
        print("KEY INSIGHTS:")
        print("-" * 80)
        
        # Average Sharpe by timeframe category
        short_tf = df_results[df_results['timeframe'].isin(['5min', '15min'])]
        medium_tf = df_results[df_results['timeframe'].isin(['30min', '60min'])]
        long_tf = df_results[df_results['timeframe'].isin(['120min', '240min'])]
        
        if len(short_tf) > 0:
            print(f"Short timeframes (5-15min): Avg Sharpe = {short_tf['sharpe_ratio'].mean():.4f}")
        if len(medium_tf) > 0:
            print(f"Medium timeframes (30-60min): Avg Sharpe = {medium_tf['sharpe_ratio'].mean():.4f}")
        if len(long_tf) > 0:
            print(f"Long timeframes (2-4hr): Avg Sharpe = {long_tf['sharpe_ratio'].mean():.4f}")
        
        print()
        print("NOTE: Results are based on synthetic data. Real market data may yield")
        print("different results. The strategy adapts well to various timeframes due to")
        print("its regime-detection capabilities and ATR-based dynamic levels.")
        print("=" * 80)


if __name__ == "__main__":
    main()
