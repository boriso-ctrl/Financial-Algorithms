"""
Simple vectorized backtest engine.

Core principles:
- No lookahead bias (signals shifted by 1 day)
- Equal-weighted positions across all tickers
- Computes basic performance metrics
- Validates signal mechanics before adding complexity
"""

from typing import Optional

import pandas as pd
import numpy as np


def _validate_price_data(prices: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Basic integrity checks for price data.

    strict=True will raise on duplicates or remaining NaNs; strict=False will
    sort index, forward/backward fill, and drop duplicate index entries.
    """
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    if prices.index.duplicated().any():
        if strict:
            raise ValueError("Price index has duplicates; strict mode enabled.")
        prices = prices[~prices.index.duplicated(keep="first")]

    if prices.isna().values.any():
        prices = prices.ffill().bfill()
        if strict and prices.isna().values.any():
            raise ValueError("Price data contains NaNs after fill; strict mode enabled.")

    return prices


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100000.0,
    cost_bps: float = 1.0,
    slippage_bps: float = 5.0,
    allow_short: bool = False,
    risk_free_rate: float = 0.0,
    max_gross_leverage: float = 1.0,
    max_position_weight: float = 1.0,
    max_signal_abs: float = 5.0,
    strict_data: bool = False,
):
    """
    Run a vectorized backtest with basic trading frictions.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (index=Date, columns=tickers, values=Close)
    signals : pd.DataFrame
        Signal matrix (index=Date, columns=tickers, values in {-1, 0, 1})
        1 = long, 0 = flat, -1 = short (shorts ignored when allow_short=False)
    initial_capital : float
        Starting capital in dollars
    cost_bps : float
        Commission/fees per notional turnover in basis points
    slippage_bps : float
        Slippage per notional turnover in basis points
    allow_short : bool
        If True, allow -1 signals; otherwise shorts are clipped to 0
    risk_free_rate : float
        Annualized risk-free rate (e.g., 0.05 for 5%) for excess-return Sharpe
    max_gross_leverage : float
        Maximum sum of absolute weights; scales down if exceeded
    max_position_weight : float
        Maximum absolute weight per position; clipped if exceeded
    max_signal_abs : float
        Clip raw signal magnitudes to +/- this value before weighting (supports conviction up to 5)
    strict_data : bool
        If True, raise on duplicate dates or remaining NaNs after fill
        
    Returns
    -------
    dict
        Backtest results including equity curve, returns, and metrics
    """
    prices = _validate_price_data(prices, strict=strict_data)

    # Align prices and signals
    signals = signals.reindex(prices.index).fillna(0)
    common = prices.columns.intersection(signals.columns)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between prices and signals.")
    prices = prices[common]
    signals = signals[common].clip(lower=-max_signal_abs, upper=max_signal_abs)
    
    # Compute daily returns
    returns = prices.pct_change()
    
    # Shift signals by 1 day (trade on tomorrow's open using today's signal)
    # This prevents lookahead bias
    shifted_signals = signals.shift(1).fillna(0)
    if not allow_short:
        shifted_signals = shifted_signals.clip(lower=0)

    # Equal-weight exposures across active signals (including shorts when enabled)
    abs_sum = shifted_signals.abs().sum(axis=1)
    weights = shifted_signals.div(abs_sum, axis=0).fillna(0)
    weights = weights.clip(lower=-max_position_weight, upper=max_position_weight)

    gross = weights.abs().sum(axis=1)
    scale = (max_gross_leverage / gross).where(gross > max_gross_leverage, other=1.0)
    weights = weights.mul(scale, axis=0)

    # Strategy gross returns from weights
    strategy_returns = returns * weights
    portfolio_gross = strategy_returns.sum(axis=1)

    # Turnover-based trading costs (commission + slippage)
    prev_weights = weights.shift().fillna(0)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    cost_rate = (cost_bps + slippage_bps) / 10000.0
    trading_cost = turnover * cost_rate
    portfolio_returns = (portfolio_gross - trading_cost).fillna(0)

    # Compute equity curve
    equity_curve = initial_capital * (1 + portfolio_returns).cumprod()
    equity_curve.name = "Equity"
    
    # Compute metrics
    metrics = compute_metrics(
        portfolio_returns,
        equity_curve,
        initial_capital,
        risk_free_rate=risk_free_rate,
        turnover=turnover,
        gross_exposure=gross,
    )
    
    return {
        "equity_curve": equity_curve,
        "portfolio_returns": portfolio_returns,
        "strategy_returns": strategy_returns,
        "num_positions": abs_sum,
        "weights": weights,
        "turnover": turnover,
        "gross_exposure": gross,
        "metrics": metrics
    }


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    turnover: Optional[pd.Series] = None,
    gross_exposure: Optional[pd.Series] = None,
):
    """Compute performance metrics."""
    total_return = (equity.iloc[-1] / initial_capital) - 1
    num_years = len(returns) / 252  # Approximate trading days per year
    cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    
    # Sharpe ratio (annualized) using excess returns
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sortino ratio (downside risk only)
    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = excess.mean() / downside_std * np.sqrt(252) if downside_std and downside_std > 0 else 0

    # Calmar ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Win rate (days with positive returns)
    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    avg_turnover = turnover.mean() if turnover is not None else 0
    avg_gross = gross_exposure.mean() if gross_exposure is not None else 0
    
    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Avg Win": f"{avg_win:.4%}",
        "Avg Loss": f"{avg_loss:.4%}",
        "Avg Turnover": f"{avg_turnover:.2f}",
        "Avg Gross Leverage": f"{avg_gross:.2f}",
        "Total Days": int(total_days),
        "Winning Days": int(winning_days),
        "Final Equity": f"${equity.iloc[-1]:,.2f}"
    }
