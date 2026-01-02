"""
Simple vectorized backtest engine.

Core principles:
- No lookahead bias (signals shifted by 1 day)
- Equal-weighted positions across all tickers
- Computes basic performance metrics
- Validates signal mechanics before adding complexity
"""

import pandas as pd
import numpy as np


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 100000.0):
    """
    Run a simple vectorized backtest.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (index=Date, columns=tickers, values=Close)
    signals : pd.DataFrame
        Signal matrix (index=Date, columns=tickers, values=0 or 1)
        1 = long position, 0 = no position
    initial_capital : float
        Starting capital in dollars
        
    Returns
    -------
    dict
        Backtest results including equity curve and metrics
    """
    # Align prices and signals
    aligned_prices = prices.reindex(signals.index).ffill()
    
    # Compute daily returns
    returns = aligned_prices.pct_change()
    
    # Shift signals by 1 day (trade on tomorrow's open using today's signal)
    # This prevents lookahead bias
    shifted_signals = signals.shift(1)
    
    # Strategy returns: only earn returns when signal = 1
    strategy_returns = returns * shifted_signals
    
    # Equal-weight portfolio: average across all tickers each day
    # Only count tickers with active signals
    num_positions = shifted_signals.sum(axis=1)
    portfolio_returns = strategy_returns.sum(axis=1) / num_positions.replace(0, np.nan)
    
    # Handle days with no positions (returns = 0)
    portfolio_returns = portfolio_returns.fillna(0)
    
    # Compute equity curve
    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()
    equity_curve.name = "Equity"
    
    # Compute metrics
    metrics = compute_metrics(portfolio_returns, equity_curve, initial_capital)
    
    return {
        "equity_curve": equity_curve,
        "portfolio_returns": portfolio_returns,
        "strategy_returns": strategy_returns,
        "num_positions": num_positions,
        "metrics": metrics
    }


def compute_metrics(returns: pd.Series, equity: pd.Series, initial_capital: float):
    """Compute performance metrics."""
    total_return = (equity.iloc[-1] / initial_capital) - 1
    num_years = len(returns) / 252  # Approximate trading days per year
    cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    
    # Sharpe ratio (annualized, assuming risk-free rate = 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate (days with positive returns)
    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Avg Win": f"{avg_win:.4%}",
        "Avg Loss": f"{avg_loss:.4%}",
        "Total Days": int(total_days),
        "Winning Days": int(winning_days),
        "Final Equity": f"${equity.iloc[-1]:,.2f}"
    }
