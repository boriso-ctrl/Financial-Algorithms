"""
Alternative data loader using yfinance instead of simfin.
Works better in sandboxed environments.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def load_daily_prices(tickers, start_date=None, end_date=None):
    """
    Load daily close prices for a list of tickers using yfinance.

    Parameters:
    -----------
    tickers : list[str]
        List of ticker symbols
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 5 years ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.

    Returns:
    --------
    pd.DataFrame
        DataFrame with:
        - index: Date (datetime)
        - columns: Tickers
        - values: Close prices
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data from {start_date} to {end_date}...")
    
    # Download data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, 
                      group_by='ticker', threads=True)
    
    if len(tickers) == 1:
        # Single ticker case
        prices = data['Close'].to_frame()
        prices.columns = tickers
    else:
        # Multiple tickers
        prices = pd.DataFrame()
        for ticker in tickers:
            try:
                if ticker in data:
                    prices[ticker] = data[ticker]['Close']
                else:
                    print(f"Warning: {ticker} not found in downloaded data")
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
    
    # Clean the data
    prices = prices.dropna(how='all')  # Remove rows with all NaN
    prices = prices.ffill().bfill()  # Forward/backward fill
    
    return prices


def load_daily_ohlcv(tickers, start_date=None, end_date=None):
    """
    Load daily OHLCV data for a list of tickers.

    Returns:
    --------
    dict
        Dictionary with keys 'open', 'high', 'low', 'close', 'volume'
        Each value is a DataFrame with tickers as columns
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading OHLCV data from {start_date} to {end_date}...")
    
    # Download data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, progress=False,
                      group_by='ticker', threads=True)
    
    result = {
        'open': pd.DataFrame(),
        'high': pd.DataFrame(),
        'low': pd.DataFrame(),
        'close': pd.DataFrame(),
        'volume': pd.DataFrame()
    }
    
    if len(tickers) == 1:
        # Single ticker case
        for key in result.keys():
            col_name = key.capitalize()
            result[key][tickers[0]] = data[col_name]
    else:
        # Multiple tickers
        for ticker in tickers:
            try:
                if ticker in data:
                    result['open'][ticker] = data[ticker]['Open']
                    result['high'][ticker] = data[ticker]['High']
                    result['low'][ticker] = data[ticker]['Low']
                    result['close'][ticker] = data[ticker]['Close']
                    result['volume'][ticker] = data[ticker]['Volume']
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
    
    # Clean the data
    for key in result.keys():
        result[key] = result[key].dropna(how='all')
        result[key] = result[key].ffill().bfill()
    
    return result


def load_daily_returns(tickers, start_date=None, end_date=None):
    """
    Load daily percentage returns for a list of tickers.
    """
    prices = load_daily_prices(tickers, start_date, end_date)
    returns = prices.pct_change().dropna()
    return returns
