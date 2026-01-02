"""
Real Historical Data Backtest

This script downloads real historical market data and runs the VWAP + ATR strategy
to provide realistic performance metrics with actual market conditions.

Uses yfinance for free historical data access.
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


def download_real_data(ticker: str = 'SPY', 
                      period: str = '2y',
                      interval: str = '5m') -> pd.DataFrame:
    """
    Download real historical market data using yfinance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY', 'QQQ', 'AAPL')
    period : str
        Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        Note: yfinance limits intraday data to last 60 days
    interval : str
        Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d')
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with session column, or None if download fails
    """
    try:
        import yfinance as yf
        print(f"Downloading {ticker} data (period={period}, interval={interval})...")
        
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)
        
        if df.empty:
            print(f"Error: No data returned for {ticker}")
            return None
        
        # Rename columns to match our format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add session column (date only)
        df['session'] = df.index.date.astype(str)
        
        # Reset index to have timestamp as column
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp', 'Datetime': 'timestamp'})
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns and 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        
        df.set_index('timestamp', inplace=True)
        
        print(f"✓ Downloaded {len(df)} bars")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Sessions: {df['session'].nunique()}")
        
        return df
        
    except ImportError:
        print("Error: yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe


def run_real_data_backtest(ticker: str = 'SPY',
                           period: str = '60d',
                           interval: str = '5m',
                           initial_capital: float = 100000):
    """
    Run backtest on real historical data.
    
    Parameters
    ----------
    ticker : str
        Stock ticker to test
    period : str
        Time period (yfinance limitation: max 60d for intraday)
    interval : str
        Data interval ('5m', '15m', '30m', '60m')
    initial_capital : float
        Starting capital
        
    Returns
    -------
    dict
        Backtest results
    """
    print("=" * 80)
    print(f"REAL HISTORICAL DATA BACKTEST: {ticker}")
    print("=" * 80)
    print()
    
    # Download data
    print("[1/5] Downloading real market data...")
    df = download_real_data(ticker, period, interval)
    
    if df is None or len(df) == 0:
        print("Error: Could not download data")
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
        bars_per_day = 78  # 6.5 hours * 60 / 5
    elif interval == '15m':
        bars_per_day = 26
    elif interval == '30m':
        bars_per_day = 13
    elif interval == '60m' or interval == '1h':
        bars_per_day = 6.5
    else:
        bars_per_day = 78  # default
    
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
        
        # Calculate days traded
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
    
    print("=" * 80)
    print(f"SHARPE RATIO: {sharpe_ratio:.4f}")
    print("=" * 80)
    print()
    
    print("IMPORTANT NOTES:")
    print("-" * 80)
    print("• This backtest uses REAL historical market data")
    print("• Performance reflects actual market conditions")
    print(f"• Ticker: {ticker}")
    print(f"• Period: {period} (yfinance limits intraday data to 60 days)")
    print(f"• Interval: {interval}")
    print("• No slippage or commissions included")
    print("• Results may vary with different tickers and time periods")
    print("=" * 80)
    
    return {
        'ticker': ticker,
        'sharpe_ratio': sharpe_ratio,
        'results': results,
        'df': df_valid
    }


def test_multiple_tickers(tickers: list = None,
                         period: str = '60d',
                         interval: str = '5m'):
    """
    Test strategy on multiple tickers with real data.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols to test
    period : str
        Time period
    interval : str
        Data interval
    """
    if tickers is None:
        tickers = ['SPY', 'QQQ', 'IWM']  # Common ETFs
    
    print("=" * 80)
    print(f"MULTI-TICKER REAL DATA BACKTEST")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {period}, Interval: {interval}")
    print("=" * 80)
    print()
    
    results_summary = []
    
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        print("=" * 80)
        
        result = run_real_data_backtest(ticker, period, interval)
        
        if result:
            results_summary.append({
                'ticker': ticker,
                'sharpe_ratio': result['sharpe_ratio'],
                'total_return': float(result['results']['metrics']['Total Return'].strip('%')) / 100,
                'win_rate': float(result['results']['metrics']['Win Rate'].strip('%')) / 100,
                'max_dd': float(result['results']['metrics']['Max Drawdown'].strip('%')) / 100,
                'total_trades': result['results']['metrics']['Total Trades']
            })
        
        print()
    
    # Summary table
    if results_summary:
        print()
        print("=" * 80)
        print("SUMMARY: ALL TICKERS")
        print("=" * 80)
        print()
        
        df_summary = pd.DataFrame(results_summary)
        df_summary = df_summary.sort_values('sharpe_ratio', ascending=False)
        
        print(f"{'Ticker':<10} {'Sharpe':<10} {'Return':<12} {'Win Rate':<12} {'Max DD':<12} {'Trades':<8}")
        print("-" * 80)
        
        for _, row in df_summary.iterrows():
            print(f"{row['ticker']:<10} "
                  f"{row['sharpe_ratio']:<10.4f} "
                  f"{row['total_return']:<12.2%} "
                  f"{row['win_rate']:<12.2%} "
                  f"{row['max_dd']:<12.2%} "
                  f"{row['total_trades']:<8.0f}")
        
        print()
        print(f"Average Sharpe Ratio: {df_summary['sharpe_ratio'].mean():.4f}")
        print(f"Average Win Rate: {df_summary['win_rate'].mean():.2%}")
        print("=" * 80)


def main():
    """
    Main execution function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest VWAP + ATR strategy on real historical data')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--period', type=str, default='60d', help='Time period (default: 60d, max for intraday)')
    parser.add_argument('--interval', type=str, default='5m', help='Data interval (default: 5m)')
    parser.add_argument('--multi', action='store_true', help='Test multiple tickers (SPY, QQQ, IWM)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    
    args = parser.parse_args()
    
    if args.multi:
        test_multiple_tickers(period=args.period, interval=args.interval)
    else:
        run_real_data_backtest(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            initial_capital=args.capital
        )


if __name__ == "__main__":
    main()
