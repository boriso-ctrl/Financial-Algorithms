"""
VWAP + ATR Indicator Calculations

This module implements all indicators required for the VWAP + ATR trading strategy:
- Session-anchored VWAP
- ATR(14) with ±1×ATR and ±2×ATR bands
- Volume Profile (POC, VAH, VAL)
- RSI(14) for regime detection
- EMA for trend filtering

All indicators are designed to avoid lookahead bias and repainting.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_vwap(df: pd.DataFrame, session_col: str = 'session') -> pd.Series:
    """
    Calculate session-anchored VWAP.
    
    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume
        Must include a 'session' column to identify session boundaries
    session_col : str
        Column name identifying trading sessions
        
    Returns
    -------
    pd.Series
        VWAP values, reset at each new session
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    
    # Create temporary dataframe for groupby operations
    temp_df = df.copy()
    temp_df['_pv'] = pv
    
    # Calculate cumulative sum within each session
    cumulative_pv = temp_df.groupby(session_col)['_pv'].cumsum()
    cumulative_volume = df.groupby(session_col)['volume'].cumsum()
    
    vwap = cumulative_pv / cumulative_volume
    vwap.name = 'vwap'
    
    return vwap


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA of True Range over 'period' bars
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: high, low, close
    period : int
        Lookback period for ATR calculation
        
    Returns
    -------
    pd.Series
        ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    # Calculate True Range components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the exponential moving average of True Range
    atr = tr.ewm(span=period, adjust=False).mean()
    atr.name = 'atr'
    
    return atr


def calculate_atr_bands(df: pd.DataFrame, vwap: pd.Series, atr: pd.Series) -> pd.DataFrame:
    """
    Calculate VWAP ± ATR bands.
    
    Creates bands at:
    - VWAP + 1×ATR
    - VWAP + 2×ATR
    - VWAP - 1×ATR
    - VWAP - 2×ATR
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used for index alignment)
    vwap : pd.Series
        VWAP values
    atr : pd.Series
        ATR values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: vwap_upper1, vwap_upper2, vwap_lower1, vwap_lower2
    """
    bands = pd.DataFrame(index=df.index)
    bands['vwap_upper1'] = vwap + atr
    bands['vwap_upper2'] = vwap + 2 * atr
    bands['vwap_lower1'] = vwap - atr
    bands['vwap_lower2'] = vwap - 2 * atr
    
    return bands


def calculate_volume_profile(df: pd.DataFrame, session_col: str = 'session', 
                            num_bins: int = 50, value_area_pct: float = 0.70) -> pd.DataFrame:
    """
    Calculate Volume Profile metrics: POC, VAH, VAL.
    
    Volume Profile shows where volume was traded at different price levels.
    - POC (Point of Control): Price level with highest volume
    - VAH (Value Area High): Upper bound of value area (70% of volume)
    - VAL (Value Area Low): Lower bound of value area (70% of volume)
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: high, low, close, volume
    session_col : str
        Column name identifying trading sessions
    num_bins : int
        Number of price bins for histogram
    value_area_pct : float
        Percentage of volume to include in value area (default 0.70 for 70%)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: poc, vah, val
    """
    result = pd.DataFrame(index=df.index)
    result['poc'] = np.nan
    result['vah'] = np.nan
    result['val'] = np.nan
    
    # Calculate for each session
    for session in df[session_col].unique():
        session_mask = df[session_col] == session
        session_data = df[session_mask]
        
        if len(session_data) == 0:
            continue
            
        # Create price bins for the session
        price_min = session_data['low'].min()
        price_max = session_data['high'].max()
        
        if price_min >= price_max or price_max == 0:
            continue
            
        price_bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # Allocate volume to price bins
        # Use typical price as representative price for each bar
        typical_prices = (session_data['high'] + session_data['low'] + session_data['close']) / 3
        volumes = session_data['volume']
        
        # Create histogram
        volume_profile = np.zeros(num_bins)
        for price, vol in zip(typical_prices, volumes):
            if pd.notna(price) and pd.notna(vol) and vol > 0:
                bin_idx = np.searchsorted(price_bins[1:], price)
                bin_idx = min(bin_idx, num_bins - 1)
                volume_profile[bin_idx] += vol
        
        if volume_profile.sum() == 0:
            continue
            
        # POC: Price level with maximum volume
        poc_bin = np.argmax(volume_profile)
        poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
        
        # Value Area: Find price range containing value_area_pct of volume
        # Start from POC and expand outward
        total_volume = volume_profile.sum()
        target_volume = total_volume * value_area_pct
        
        va_bins = [poc_bin]
        va_volume = volume_profile[poc_bin]
        
        lower_idx = poc_bin - 1
        upper_idx = poc_bin + 1
        
        # Expand value area
        while va_volume < target_volume and (lower_idx >= 0 or upper_idx < num_bins):
            lower_vol = volume_profile[lower_idx] if lower_idx >= 0 else 0
            upper_vol = volume_profile[upper_idx] if upper_idx < num_bins else 0
            
            if lower_vol >= upper_vol and lower_idx >= 0:
                va_bins.append(lower_idx)
                va_volume += lower_vol
                lower_idx -= 1
            elif upper_idx < num_bins:
                va_bins.append(upper_idx)
                va_volume += upper_vol
                upper_idx += 1
            else:
                break
        
        # VAH and VAL
        va_bins_sorted = sorted(va_bins)
        val_price = price_bins[va_bins_sorted[0]]
        vah_price = price_bins[va_bins_sorted[-1] + 1]
        
        # Assign to all bars in this session
        result.loc[session_mask, 'poc'] = poc_price
        result.loc[session_mask, 'vah'] = vah_price
        result.loc[session_mask, 'val'] = val_price
    
    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period
    
    Used for regime detection, not traditional overbought/oversold signals.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with column: close
    period : int
        Lookback period for RSI calculation
        
    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    close = df['close']
    delta = close.diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate exponential moving averages
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = 'rsi'
    
    return rsi


def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Used as trend filter.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with column: close
    period : int
        Lookback period for EMA calculation
        
    Returns
    -------
    pd.Series
        EMA values
    """
    ema = df['close'].ewm(span=period, adjust=False).mean()
    ema.name = f'ema_{period}'
    
    return ema


def calculate_indicators(df: pd.DataFrame, session_col: str = 'session',
                        atr_period: int = 14, rsi_period: int = 14,
                        ema_period: int = 20) -> pd.DataFrame:
    """
    Calculate all indicators for the VWAP + ATR strategy.
    
    This is the main function to compute all required indicators at once.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: timestamp, open, high, low, close, volume
        Must include a 'session' column to identify session boundaries
    session_col : str
        Column name identifying trading sessions
    atr_period : int
        Period for ATR calculation
    rsi_period : int
        Period for RSI calculation
    ema_period : int
        Period for EMA calculation
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with all indicators added as new columns:
        - vwap
        - atr
        - vwap_upper1, vwap_upper2, vwap_lower1, vwap_lower2
        - poc, vah, val
        - rsi
        - ema
    """
    result = df.copy()
    
    # VWAP
    result['vwap'] = calculate_vwap(df, session_col)
    
    # ATR
    result['atr'] = calculate_atr(df, period=atr_period)
    
    # ATR Bands
    bands = calculate_atr_bands(df, result['vwap'], result['atr'])
    result = pd.concat([result, bands], axis=1)
    
    # Volume Profile
    vp = calculate_volume_profile(df, session_col)
    result = pd.concat([result, vp], axis=1)
    
    # RSI
    result['rsi'] = calculate_rsi(df, period=rsi_period)
    
    # EMA
    result['ema'] = calculate_ema(df, period=ema_period)
    
    return result
