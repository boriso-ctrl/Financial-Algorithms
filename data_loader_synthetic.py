"""
Generate synthetic price data for backtesting when real data is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Volume generation constants (for lognormal distribution)
VOLUME_LOG_MEAN = 13.8  # Generates approximately 1 million shares per day
VOLUME_LOG_STD = 0.5    # Standard deviation for realistic volume variation


def generate_synthetic_prices(tickers, days=756, start_date=None, 
                              initial_price=100, volatility=0.02, 
                              trend=0.0003, seed=None):
    """
    Generate synthetic price data with realistic characteristics.
    
    Parameters:
    -----------
    tickers : list[str]
        List of ticker symbols
    days : int
        Number of trading days to generate (default 756 = ~3 years)
    start_date : datetime, optional
        Starting date for the data
    initial_price : float
        Starting price for all tickers
    volatility : float
        Daily volatility (std dev of returns)
    trend : float
        Daily drift/trend
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Price data with DatetimeIndex
    """
    if seed is not None:
        np.random.seed(seed)
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Generate date range (business days only)
    date_range = pd.bdate_range(start=start_date, periods=days)
    
    prices = pd.DataFrame(index=date_range, columns=tickers)
    
    for ticker in tickers:
        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, days)
        
        # Add some autocorrelation for realism
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Convert to prices
        price_series = initial_price * np.exp(np.cumsum(returns))
        prices[ticker] = price_series
    
    return prices.astype(float)


def generate_synthetic_ohlcv(tickers, days=756, start_date=None, 
                             initial_price=100, volatility=0.02, 
                             trend=0.0003, seed=None):
    """
    Generate synthetic OHLCV data.
    
    Returns:
    --------
    dict
        Dictionary with keys 'open', 'high', 'low', 'close', 'volume'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate close prices first
    close_prices = generate_synthetic_prices(tickers, days, start_date, 
                                            initial_price, volatility, 
                                            trend, seed)
    
    result = {
        'close': close_prices,
        'open': pd.DataFrame(index=close_prices.index, columns=tickers),
        'high': pd.DataFrame(index=close_prices.index, columns=tickers),
        'low': pd.DataFrame(index=close_prices.index, columns=tickers),
        'volume': pd.DataFrame(index=close_prices.index, columns=tickers)
    }
    
    for ticker in tickers:
        close = close_prices[ticker].values
        
        # Generate realistic OHLC
        # Open is close from previous day with small gap
        open_prices = np.zeros_like(close)
        open_prices[0] = close[0]
        open_prices[1:] = close[:-1] * (1 + np.random.normal(0, volatility/4, len(close)-1))
        
        # High is max of open/close plus some random amount
        high_prices = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, volatility/2, len(close))))
        
        # Low is min of open/close minus some random amount
        low_prices = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, volatility/2, len(close))))
        
        # Volume (random around 1M shares)
        # Using lognormal to generate realistic volume around 1 million
        # mean = exp(mu + sigma^2/2) ≈ 1,000,000 shares
        volume = np.random.lognormal(VOLUME_LOG_MEAN, VOLUME_LOG_STD, len(close))
        
        result['open'][ticker] = open_prices
        result['high'][ticker] = high_prices
        result['low'][ticker] = low_prices
        result['volume'][ticker] = volume
    
    return result


def load_daily_prices(tickers, use_synthetic=True, **kwargs):
    """
    Load or generate daily price data.
    
    Parameters:
    -----------
    tickers : list[str]
        List of ticker symbols
    use_synthetic : bool
        If True, generate synthetic data
    **kwargs : dict
        Additional arguments for synthetic data generation
        
    Returns:
    --------
    pd.DataFrame
        Price data
    """
    if use_synthetic:
        print(f"Generating synthetic data for {len(tickers)} tickers...")
        return generate_synthetic_prices(tickers, **kwargs)
    else:
        # Try to load real data
        try:
            from data_loader_yfinance import load_daily_prices as load_real
            return load_real(tickers)
        except:
            print("Failed to load real data, falling back to synthetic")
            return generate_synthetic_prices(tickers, **kwargs)
