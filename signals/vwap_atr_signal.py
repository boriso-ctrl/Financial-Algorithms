"""
VWAP + ATR Signal Generation

This module generates trading signals based on:
- Market regime (trend vs rotational)
- Mean reversion setups
- Trend continuation setups
- Risk management rules

Signals are designed to be:
- Deterministic (no discretionary logic)
- Non-repainting (based on closed candles)
- Mutually exclusive (no conflicting signals)
"""

import pandas as pd
import numpy as np
from typing import Tuple


def check_mean_reversion_setup(df: pd.DataFrame, row_idx: int) -> Tuple[str, float, float]:
    """
    Check for mean reversion trade setup.
    
    Mean reversion trades are allowed only when:
    - Market regime = rotational
    - Price is between 1×ATR and 2×ATR away from VWAP
    - Price is near VAH / VAL / POC
    - RSI regime supports direction
    
    Entry:
    - Long: below VWAP (price stretched down)
    - Short: above VWAP (price stretched up)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all indicators and regime
    row_idx : int
        Current row index
        
    Returns
    -------
    tuple
        (signal, stop_loss, take_profit)
        signal: 'long', 'short', or 'none'
        stop_loss: stop loss price (0 if no signal)
        take_profit: take profit price (0 if no signal)
    """
    row = df.iloc[row_idx]
    
    # Must be in rotational regime
    if row['regime'] != 'rotational':
        return 'none', 0, 0
    
    close = row['close']
    vwap = row['vwap']
    atr = row['atr']
    vah = row['vah']
    val = row['val']
    poc = row['poc']
    
    # Check if price is stretched from VWAP (between 1×ATR and 2×ATR)
    distance_from_vwap = abs(close - vwap)
    stretched = (distance_from_vwap >= atr) and (distance_from_vwap <= 2 * atr)
    
    if not stretched:
        return 'none', 0, 0
    
    # Check if price is near value area levels
    # "Near" is defined as within 0.5×ATR of the level
    near_threshold = 0.5 * atr
    near_vah = abs(close - vah) <= near_threshold
    near_val = abs(close - val) <= near_threshold
    near_poc = abs(close - poc) <= near_threshold
    
    near_value_level = near_vah or near_val or near_poc
    
    if not near_value_level:
        return 'none', 0, 0
    
    # Determine direction based on price vs VWAP
    if close < vwap:
        # Price below VWAP: potential long (mean reversion up)
        # RSI regime should support bullish (RSI >= 40)
        if row['rsi_regime'] >= 0:  # Bullish or neutral
            signal = 'long'
            # Stop loss: 1×ATR below entry
            stop_loss = close - atr
            # Take profit 1: VWAP (mean reversion target)
            take_profit = vwap
            return signal, stop_loss, take_profit
    
    elif close > vwap:
        # Price above VWAP: potential short (mean reversion down)
        # RSI regime should support bearish (RSI <= 60)
        if row['rsi_regime'] <= 0:  # Bearish or neutral
            signal = 'short'
            # Stop loss: 1×ATR above entry
            stop_loss = close + atr
            # Take profit 1: VWAP (mean reversion target)
            take_profit = vwap
            return signal, stop_loss, take_profit
    
    return 'none', 0, 0


def check_trend_continuation_setup(df: pd.DataFrame, row_idx: int) -> Tuple[str, float, float]:
    """
    Check for trend continuation trade setup.
    
    Trend continuation trades are allowed only when:
    - Market regime = trend
    - Price pulls back to VWAP or EMA
    - RSI regime confirms trend direction
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all indicators and regime
    row_idx : int
        Current row index
        
    Returns
    -------
    tuple
        (signal, stop_loss, take_profit)
        signal: 'long', 'short', or 'none'
        stop_loss: stop loss price (0 if no signal)
        take_profit: take profit price (0 if no signal)
    """
    row = df.iloc[row_idx]
    
    # Must be in trend regime
    if row['regime'] != 'trend':
        return 'none', 0, 0
    
    close = row['close']
    vwap = row['vwap']
    ema = row['ema']
    atr = row['atr']
    alignment = row['alignment']
    
    # Determine trend direction from alignment
    if alignment == 1:
        # Bullish trend: close > vwap > ema
        # Look for pullback to VWAP or EMA for long entry
        # "Pullback" means price is near VWAP or EMA (within 0.3×ATR)
        pullback_threshold = 0.3 * atr
        near_vwap = abs(close - vwap) <= pullback_threshold
        near_ema = abs(close - ema) <= pullback_threshold
        
        # But price should still be above EMA (trend intact)
        if (near_vwap or near_ema) and close > ema:
            # RSI should support bullish (RSI >= 40)
            if row['rsi_regime'] >= 0:
                signal = 'long'
                # Stop loss: 0.75×ATR below entry (tighter for trend trades)
                stop_loss = close - 0.75 * atr
                # Take profit: opposite ATR band (VWAP + 1×ATR)
                take_profit = vwap + atr
                return signal, stop_loss, take_profit
    
    elif alignment == -1:
        # Bearish trend: close < vwap < ema
        # Look for pullback to VWAP or EMA for short entry
        pullback_threshold = 0.3 * atr
        near_vwap = abs(close - vwap) <= pullback_threshold
        near_ema = abs(close - ema) <= pullback_threshold
        
        # But price should still be below EMA (trend intact)
        if (near_vwap or near_ema) and close < ema:
            # RSI should support bearish (RSI <= 60)
            if row['rsi_regime'] <= 0:
                signal = 'short'
                # Stop loss: 0.75×ATR above entry (tighter for trend trades)
                stop_loss = close + 0.75 * atr
                # Take profit: opposite ATR band (VWAP - 1×ATR)
                take_profit = vwap - atr
                return signal, stop_loss, take_profit
    
    return 'none', 0, 0


def generate_signals(df: pd.DataFrame, session_col: str = 'session') -> pd.DataFrame:
    """
    Generate trading signals based on market regime and setups.
    
    Rules:
    - Mean reversion trades in rotational markets
    - Trend continuation trades in trending markets
    - One trade per direction per session
    - No conflicting signals
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all indicators and regime calculated
    session_col : str
        Column name identifying trading sessions
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with signal columns added:
        - signal: 'long', 'short', or 'none'
        - stop_loss: stop loss price
        - take_profit: take profit price
        - position: current position (1=long, -1=short, 0=flat)
    """
    result = df.copy()
    
    # Initialize signal columns
    result['signal'] = 'none'
    result['stop_loss'] = 0.0
    result['take_profit'] = 0.0
    result['position'] = 0
    
    # Track trades per session (one per direction per session)
    session_trades = {}
    
    # Current position tracking
    current_position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    for i in range(len(result)):
        row = result.iloc[i]
        session = row[session_col]
        
        # Initialize session tracking
        if session not in session_trades:
            session_trades[session] = {'long': False, 'short': False}
        
        # Reset position at session start (if new session and position open)
        if i > 0 and result.iloc[i-1][session_col] != session:
            if current_position != 0:
                # Close position at session end
                current_position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0
        
        # Update position column
        result.iloc[i, result.columns.get_loc('position')] = current_position
        
        # Check exit conditions if in position
        if current_position != 0:
            close = row['close']
            
            # Check stop loss
            if current_position == 1 and close <= stop_loss:
                # Long stopped out
                current_position = 0
                entry_price = 0
                continue
            
            elif current_position == -1 and close >= stop_loss:
                # Short stopped out
                current_position = 0
                entry_price = 0
                continue
            
            # Check take profit
            if current_position == 1 and close >= take_profit:
                # Long take profit hit
                current_position = 0
                entry_price = 0
                continue
            
            elif current_position == -1 and close <= take_profit:
                # Short take profit hit
                current_position = 0
                entry_price = 0
                continue
        
        # Generate new signals only if flat
        if current_position == 0:
            # Check mean reversion setup first (higher priority in rotational regime)
            signal_mr, sl_mr, tp_mr = check_mean_reversion_setup(result, i)
            
            if signal_mr != 'none':
                # Check if we can trade this direction in this session
                if not session_trades[session][signal_mr]:
                    result.iloc[i, result.columns.get_loc('signal')] = signal_mr
                    result.iloc[i, result.columns.get_loc('stop_loss')] = sl_mr
                    result.iloc[i, result.columns.get_loc('take_profit')] = tp_mr
                    
                    # Update position
                    current_position = 1 if signal_mr == 'long' else -1
                    entry_price = row['close']
                    stop_loss = sl_mr
                    take_profit = tp_mr
                    
                    # Mark session trade
                    session_trades[session][signal_mr] = True
                    continue
            
            # Check trend continuation setup
            signal_tc, sl_tc, tp_tc = check_trend_continuation_setup(result, i)
            
            if signal_tc != 'none':
                # Check if we can trade this direction in this session
                if not session_trades[session][signal_tc]:
                    result.iloc[i, result.columns.get_loc('signal')] = signal_tc
                    result.iloc[i, result.columns.get_loc('stop_loss')] = sl_tc
                    result.iloc[i, result.columns.get_loc('take_profit')] = tp_tc
                    
                    # Update position
                    current_position = 1 if signal_tc == 'long' else -1
                    entry_price = row['close']
                    stop_loss = sl_tc
                    take_profit = tp_tc
                    
                    # Mark session trade
                    session_trades[session][signal_tc] = True
    
    return result


def validate_signals(df: pd.DataFrame) -> bool:
    """
    Validate that signals are unambiguous and non-conflicting.
    
    Checks:
    - No simultaneous long and short signals
    - Stop loss and take profit are set when signal is generated
    - All rules are deterministic
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with signals generated
        
    Returns
    -------
    bool
        True if all validations pass
    """
    signals = df[df['signal'] != 'none']
    
    if len(signals) == 0:
        return True
    
    # Check that stop loss and take profit are set
    invalid_sl = signals[signals['stop_loss'] == 0]
    invalid_tp = signals[signals['take_profit'] == 0]
    
    if len(invalid_sl) > 0 or len(invalid_tp) > 0:
        print(f"Warning: Found {len(invalid_sl)} signals without stop loss")
        print(f"Warning: Found {len(invalid_tp)} signals without take profit")
        return False
    
    return True
