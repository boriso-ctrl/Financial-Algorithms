"""
Intraday Backtest Engine

This module provides backtesting functionality for intraday strategies
with proper handling of:
- Position tracking
- Stop loss and take profit execution
- Intraday session boundaries
- Performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def run_intraday_backtest(df: pd.DataFrame, initial_capital: float = 100000.0,
                          position_size_pct: float = 1.0) -> Dict:
    """
    Run backtest for intraday strategy with stop loss and take profit.
    
    This backtest assumes signals are already generated and includes:
    - position column indicating current position
    - stop_loss and take_profit levels
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OHLCV data, indicators, regime, and signals
        Must include columns: timestamp, close, signal, position, stop_loss, take_profit
    initial_capital : float
        Starting capital in dollars
    position_size_pct : float
        Percentage of capital to use per trade (default 1.0 = 100%)
        
    Returns
    -------
    dict
        Backtest results including:
        - equity_curve: Series of portfolio value over time
        - trades: DataFrame of all trades executed
        - metrics: Dictionary of performance metrics
    """
    # Initialize tracking
    capital = initial_capital
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0
    entry_time = None
    shares = 0
    
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['timestamp'] if 'timestamp' in df.columns else df.index[i]
        close = row['close']
        
        # Record equity
        if position == 0:
            equity = capital
        elif position == 1:
            equity = capital + shares * (close - entry_price)
        else:  # position == -1
            equity = capital - shares * (close - entry_price)
        
        equity_curve.append({'timestamp': current_time, 'equity': equity})
        
        # Check for signal
        if row['signal'] != 'none' and position == 0:
            # Enter new position
            signal = row['signal']
            position = 1 if signal == 'long' else -1
            entry_price = close
            entry_time = current_time
            
            # Calculate position size
            position_value = capital * position_size_pct
            shares = position_value / entry_price
            
        # Check exit conditions
        elif position != 0:
            exit_reason = None
            exit_price = None
            
            # Check stop loss
            if position == 1 and close <= row['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = row['stop_loss']
            elif position == -1 and close >= row['stop_loss']:
                exit_reason = 'stop_loss'
                exit_price = row['stop_loss']
            
            # Check take profit
            elif position == 1 and close >= row['take_profit']:
                exit_reason = 'take_profit'
                exit_price = row['take_profit']
            elif position == -1 and close <= row['take_profit']:
                exit_reason = 'take_profit'
                exit_price = row['take_profit']
            
            # Execute exit
            if exit_reason is not None:
                # Calculate P&L
                if position == 1:
                    pnl = shares * (exit_price - entry_price)
                else:  # position == -1
                    pnl = shares * (entry_price - exit_price)
                
                pnl_pct = pnl / (shares * entry_price)
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'long' if position == 1 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason
                })
                
                # Update capital
                capital += pnl
                
                # Reset position
                position = 0
                entry_price = 0
                entry_time = None
                shares = 0
    
    # Close any open position at end
    if position != 0:
        row = df.iloc[-1]
        current_time = row['timestamp'] if 'timestamp' in df.columns else df.index[-1]
        close = row['close']
        
        if position == 1:
            pnl = shares * (close - entry_price)
        else:
            pnl = shares * (entry_price - close)
        
        pnl_pct = pnl / (shares * entry_price)
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': current_time,
            'direction': 'long' if position == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': close,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': 'end_of_data'
        })
        
        capital += pnl
    
    # Convert to DataFrames
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('timestamp', inplace=True)
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Calculate metrics
    metrics = calculate_intraday_metrics(trades_df, equity_df, initial_capital)
    
    return {
        'equity_curve': equity_df,
        'trades': trades_df,
        'metrics': metrics
    }


def calculate_intraday_metrics(trades: pd.DataFrame, equity: pd.DataFrame, 
                               initial_capital: float) -> Dict:
    """
    Calculate performance metrics for intraday strategy.
    
    Parameters
    ----------
    trades : pd.DataFrame
        DataFrame of all trades
    equity : pd.DataFrame
        Equity curve
    initial_capital : float
        Starting capital
        
    Returns
    -------
    dict
        Performance metrics
    """
    if len(trades) == 0:
        return {
            'Total Trades': 0,
            'Total Return': '0.00%',
            'Win Rate': '0.00%',
            'Avg Win': '0.00%',
            'Avg Loss': '0.00%',
            'Profit Factor': '0.00',
            'Max Drawdown': '0.00%',
            'Final Equity': f'${initial_capital:,.2f}'
        }
    
    # Basic stats
    num_trades = len(trades)
    final_equity = equity['equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Win rate
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    
    # Average win/loss
    avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
    
    # Profit factor
    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Maximum drawdown
    running_max = equity['equity'].cummax()
    drawdown = (equity['equity'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Stop loss vs take profit exits
    sl_exits = len(trades[trades['exit_reason'] == 'stop_loss'])
    tp_exits = len(trades[trades['exit_reason'] == 'take_profit'])
    
    return {
        'Total Trades': num_trades,
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Total Return': f'{total_return:.2%}',
        'Win Rate': f'{win_rate:.2%}',
        'Avg Win': f'{avg_win:.4%}',
        'Avg Loss': f'{avg_loss:.4%}',
        'Profit Factor': f'{profit_factor:.2f}',
        'Max Drawdown': f'{max_drawdown:.2%}',
        'Stop Loss Exits': sl_exits,
        'Take Profit Exits': tp_exits,
        'Final Equity': f'${final_equity:,.2f}'
    }


def analyze_regime_performance(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by market regime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with regime information
    trades : pd.DataFrame
        Trades dataframe
        
    Returns
    -------
    pd.DataFrame
        Performance breakdown by regime
    """
    if len(trades) == 0:
        return pd.DataFrame()
    
    # Merge trades with regime information
    trades_with_regime = trades.copy()
    
    # For each trade, get the regime at entry time
    regime_at_entry = []
    for _, trade in trades.iterrows():
        entry_time = trade['entry_time']
        regime_row = df[df.index == entry_time]
        if len(regime_row) > 0:
            regime_at_entry.append(regime_row.iloc[0]['regime'])
        else:
            regime_at_entry.append('unknown')
    
    trades_with_regime['regime'] = regime_at_entry
    
    # Group by regime
    regime_stats = []
    for regime in ['trend', 'rotational']:
        regime_trades = trades_with_regime[trades_with_regime['regime'] == regime]
        
        if len(regime_trades) > 0:
            winning = regime_trades[regime_trades['pnl'] > 0]
            win_rate = len(winning) / len(regime_trades)
            avg_pnl = regime_trades['pnl_pct'].mean()
            
            regime_stats.append({
                'regime': regime,
                'num_trades': len(regime_trades),
                'win_rate': f'{win_rate:.2%}',
                'avg_pnl_pct': f'{avg_pnl:.4%}',
                'total_pnl': regime_trades['pnl'].sum()
            })
    
    return pd.DataFrame(regime_stats)
