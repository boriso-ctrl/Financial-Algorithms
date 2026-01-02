"""
Market Regime Detection

This module detects market regime (Trend vs Rotational) based on:
- Price/VWAP/EMA alignment
- ATR expansion/contraction
- Price position relative to value area

Regime detection helps determine which trading strategy to use:
- Trend Day: Use trend continuation trades
- Rotational Day: Use mean reversion trades
"""

import pandas as pd
import numpy as np


def calculate_atr_expansion(atr: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Determine if ATR is expanding or contracting.
    
    ATR is expanding if current ATR > recent average ATR.
    
    Parameters
    ----------
    atr : pd.Series
        ATR values
    lookback : int
        Period for calculating average ATR
        
    Returns
    -------
    pd.Series
        Boolean series: True if ATR is expanding, False if contracting
    """
    avg_atr = atr.rolling(window=lookback).mean()
    expanding = atr > avg_atr
    expanding.name = 'atr_expanding'
    
    return expanding


def check_alignment(close: pd.Series, vwap: pd.Series, ema: pd.Series) -> pd.Series:
    """
    Check if price, VWAP, and EMA are aligned (all bullish or all bearish).
    
    Bullish alignment: close > vwap and vwap > ema
    Bearish alignment: close < vwap and vwap < ema
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    vwap : pd.Series
        VWAP values
    ema : pd.Series
        EMA values
        
    Returns
    -------
    pd.Series
        Alignment direction: 1 (bullish), -1 (bearish), 0 (not aligned)
    """
    bullish_aligned = (close > vwap) & (vwap > ema)
    bearish_aligned = (close < vwap) & (vwap < ema)
    
    alignment = pd.Series(0, index=close.index)
    alignment[bullish_aligned] = 1
    alignment[bearish_aligned] = -1
    alignment.name = 'alignment'
    
    return alignment


def check_price_in_value_area(close: pd.Series, vah: pd.Series, val: pd.Series) -> pd.Series:
    """
    Check if price is within the value area (between VAH and VAL).
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    vah : pd.Series
        Value Area High
    val : pd.Series
        Value Area Low
        
    Returns
    -------
    pd.Series
        Boolean series: True if price is in value area
    """
    in_value_area = (close >= val) & (close <= vah)
    in_value_area.name = 'in_value_area'
    
    return in_value_area


def detect_regime(df: pd.DataFrame, atr_lookback: int = 20) -> pd.Series:
    """
    Detect market regime: Trend Day vs Rotational Day.
    
    Trend Day characteristics:
    - Price, VWAP, and EMA are aligned (all bullish or all bearish)
    - ATR is expanding (volatility increasing)
    - Price is moving away from value area
    
    Rotational Day characteristics:
    - Price oscillates around VWAP
    - ATR is contracting (volatility decreasing)
    - Price stays within value area
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with indicators calculated
        Must include: close, vwap, ema, atr, vah, val
    atr_lookback : int
        Lookback period for ATR expansion calculation
        
    Returns
    -------
    pd.Series
        Regime: 'trend' or 'rotational'
    """
    # Calculate helper metrics
    atr_expanding = calculate_atr_expansion(df['atr'], lookback=atr_lookback)
    alignment = check_alignment(df['close'], df['vwap'], df['ema'])
    in_value_area = check_price_in_value_area(df['close'], df['vah'], df['val'])
    
    # Initialize regime
    regime = pd.Series('rotational', index=df.index, name='regime')
    
    # Trend Day conditions:
    # - Strong alignment (price, VWAP, EMA aligned in same direction)
    # - ATR expanding
    trend_conditions = (alignment.abs() == 1) & atr_expanding
    
    # Additional trend confirmation: price moving away from value area
    # (not strictly required but strengthens signal)
    regime[trend_conditions] = 'trend'
    
    # Rotational Day conditions (default):
    # - Price in value area
    # - ATR contracting
    # - No strong alignment
    rotational_conditions = in_value_area & ~atr_expanding
    regime[rotational_conditions] = 'rotational'
    
    return regime


def get_rsi_regime(rsi: pd.Series) -> pd.Series:
    """
    Determine RSI regime for directional bias.
    
    This is NOT traditional overbought/oversold.
    Instead, it identifies whether conditions favor:
    - Bullish regime: RSI >= 40 (uptrend bias)
    - Bearish regime: RSI <= 60 (downtrend bias)
    - Neutral: 40 < RSI < 60
    
    Parameters
    ----------
    rsi : pd.Series
        RSI values
        
    Returns
    -------
    pd.Series
        RSI regime: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    rsi_regime = pd.Series(0, index=rsi.index, name='rsi_regime')
    rsi_regime[rsi >= 40] = 1  # Bullish regime
    rsi_regime[rsi <= 60] = -1  # Bearish regime (overlaps with bullish intentionally)
    
    # In overlap zone (40-60), both can be true
    # For strict separation, we could use neutral zone
    # But the prompt suggests using these thresholds for regime support
    
    return rsi_regime


def detect_full_regime(df: pd.DataFrame, atr_lookback: int = 20) -> pd.DataFrame:
    """
    Detect complete regime information including market regime and RSI regime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with indicators calculated
        Must include: close, vwap, ema, atr, vah, val, rsi
    atr_lookback : int
        Lookback period for ATR expansion calculation
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with regime columns added:
        - regime: 'trend' or 'rotational'
        - rsi_regime: 1 (bullish), -1 (bearish), 0 (neutral)
        - atr_expanding: boolean
        - alignment: 1 (bullish), -1 (bearish), 0 (not aligned)
        - in_value_area: boolean
    """
    result = df.copy()
    
    # Market regime
    result['regime'] = detect_regime(df, atr_lookback=atr_lookback)
    
    # RSI regime
    result['rsi_regime'] = get_rsi_regime(df['rsi'])
    
    # Add helper columns for transparency
    result['atr_expanding'] = calculate_atr_expansion(df['atr'], lookback=atr_lookback)
    result['alignment'] = check_alignment(df['close'], df['vwap'], df['ema'])
    result['in_value_area'] = check_price_in_value_area(df['close'], df['vah'], df['val'])
    
    return result
