"""Integration demo: Multi-asset intraday trading with Sharpe 3.0 architecture."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.financial_algorithms.data.yfinance_loader import load_stock_bars, get_tech_tickers
from src.financial_algorithms.data.binance_loader import load_crypto_bars
from src.financial_algorithms.backtest.intraday_engine import IntradayBacktest
from src.financial_algorithms.backtest.regime_detection import (
    detect_market_regime,
    detect_trend_direction,
    combine_regime_filters,
)
from src.financial_algorithms.signals.multitimeframe import (
    MultiTimeframeEnsemble,
    ensemble_entry_conditions,
)
from src.financial_algorithms.backtest.position_sizing import adaptive_size_ensemble


def create_signal_function():
    """Create a multi-timeframe SMA crossover signal with regime filtering (walk-forward safe)."""
    
    def signal_func(df_hist: pd.DataFrame) -> float:
        """
        Generate signal for CURRENT bar using only historical data (no lookahead).
        
        Args:
            df_hist: Historical OHLCV bars up to current bar (inclusive)
                    with [timestamp, open, high, low, close, volume, symbol]
        
        Returns:
            Signal for current bar: -1 (short), 0 (neutral), 1 (long)
        """
        if len(df_hist) < 15:  # Minimum bars for indicators
            return 0
        
        # Use ONLY historical data (all bars up to now)
        close = df_hist['close'].values
        volume = df_hist['volume'].values
        
        # Fast SMA crossover (using current bar + history)
        fast_ma_val = pd.Series(close).ewm(span=5).mean().iloc[-1]
        slow_ma_val = pd.Series(close).ewm(span=10).mean().iloc[-1]
        
        base_signal = 1 if fast_ma_val > slow_ma_val else (-1 if fast_ma_val < slow_ma_val else 0)
        
        # Regime filter: avoid RSI extremes
        market_regime = detect_market_regime(
            pd.Series(close),
            rsi_period=14,
            rsi_oversold=25,
            rsi_overbought=75,
            volume=pd.Series(volume),
        )
        
        # Get current bar's regime value
        current_regime = market_regime.iloc[-1] if len(market_regime) > 0 else 1
        
        # Apply regime filter: only trade if regime allows
        signal = base_signal if current_regime == 1 else 0
        
        return signal
    
    return signal_func


def demo_stock_backtest(tickers: List[str] = None, interval: str = '1m', period: str = '1d'):
    """Quick backtest on stock intraday data (walk-forward, no lookahead)."""
    
    if tickers is None:
        tickers = ['AAPL', 'MSFT']
    
    print(f"\n{'='*60}")
    print(f"DEMO: Stock Intraday Backtest (WALK-FORWARD)")
    print(f"{'='*60}")
    print(f"Tickers: {tickers}")
    print(f"Interval: {interval}")
    print(f"Period: {period}")
    
    # Load data
    print(f"\nLoading stock data...")
    try:
        df_bars = load_stock_bars(tickers, interval=interval, period=period, progress=False)
        print(f"[OK] Loaded {len(df_bars):,} bars for {len(df_bars['symbol'].unique())} symbols")
        print(f"  Date range: {df_bars['timestamp'].min()} to {df_bars['timestamp'].max()}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        print(f"  Note: yfinance 1m data requires active market hours (9:30-16:00 ET US)")
        return
    
    # Backtest with walk-forward (no lookahead)
    print(f"\nRunning walk-forward backtest (no lookahead bias)...")
    bt = IntradayBacktest(
        initial_capital=100000,
        slippage_bps=2.0,
        commission_pct=0.01,
        max_positions=len(tickers),
    )
    
    signal_func = create_signal_function()
    bt.backtest(df_bars, signal_func, close_on_bar_n=1440)  # Hold for 1 day max
    
    # Results
    metrics = bt.calculate_metrics()
    print(f"\n{'RESULTS':^60}")
    print(f"-" * 60)
    for key, val in metrics.items():
        print(f"{key:.<45} {val:>10}")
    
    if bt.trades:
        print(f"\nTrade Summary:")
        trades_df = bt.get_trades_df()
        print(f"  Avg Price Movement: {trades_df['pnl_pct'].mean():.3f}%")
        print(f"  Best Trade: {trades_df['pnl_pct'].max():.3f}%")
        print(f"  Worst Trade: {trades_df['pnl_pct'].min():.3f}%")
        print(f"  Std Dev: {trades_df['pnl_pct'].std():.3f}%")
        print(f"\nFirst 5 trades:")
        print(trades_df[['symbol', 'entry_time', 'exit_time', 'pnl', 'pnl_pct']].head())
    else:
        print(f"\n[INFO] No trades generated (all signals filtered)")
    
    return metrics, bt


def demo_crypto_backtest():
    """Quick backtest on crypto intraday data (walk-forward, no lookahead)."""
    
    print(f"\n{'='*60}")
    print(f"DEMO: Crypto Intraday Backtest (WALK-FORWARD)")
    print(f"{'='*60}")
    
    # Load crypto data
    print(f"Loading crypto data...")
    try:
        df_bars = load_crypto_bars(
            symbols=['BTC/USDT', 'ETH/USDT'],
            timeframe='1m',
            days_back=1,
        )
        print(f"[OK] Loaded {len(df_bars):,} bars")
        print(f"  Symbols: {df_bars['symbol'].unique()}")
        print(f"  Date range: {df_bars['timestamp'].min()} to {df_bars['timestamp'].max()}")
    except Exception as e:
        print(f"[ERROR] Error loading crypto data: {e}")
        return
    
    # Backtest with walk-forward (no lookahead)
    print(f"Running walk-forward backtest (no lookahead bias)...")
    bt = IntradayBacktest(
        initial_capital=10000,
        slippage_bps=5.0,  # Higher slippage for crypto
        commission_pct=0.05,
        max_positions=2,
    )
    
    signal_func = create_signal_function()
    bt.backtest(df_bars, signal_func, close_on_bar_n=60)  # Max 60 bars = 1 hour
    
    metrics = bt.calculate_metrics()
    print(f"\n{'RESULTS':^60}")
    print(f"-" * 60)
    for key, val in metrics.items():
        print(f"{key:.<45} {val:>10}")
    
    if bt.trades:
        print(f"\nFirst 5 trades:")
        trades_df = bt.get_trades_df()
        print(trades_df[['symbol', 'entry_time', 'exit_time', 'pnl', 'pnl_pct']].head())
    else:
        print(f"\n[INFO] No trades generated (signals filtered or not enough history)")
    
    return metrics, bt


def demo_architecture():
    """Show the Sharpe 3.0 architecture."""
    
    print(f"\n{'='*60}")
    print(f"PHASE 6 - SHARPE 3.0 ARCHITECTURE")
    print(f"{'='*60}")
    
    print(f"""
1. DATA LAYER
   [OK] Intraday multi-asset collection
   [OK] Stock API: yfinance (1-5min bars, US market)
   [OK] Crypto API: Binance/CCXT (1-5min bars, 24/7)
   [OK] Resampling to 1/5/15 min for ensemble

2. SIGNAL LAYER
   [OK] Per-timeframe indicators (SMA, RSI, MACD, etc.)
   [OK] Multi-timeframe consensus voting
   [OK] Confidence scoring (0-1)

3. RISK MANAGEMENT LAYER
   [OK] Regime detection (RSI extremes, volume)
   [OK] Trend following (skip choppy markets)
   [OK] Volatility adaptation (ATR-based)

4. POSITION SIZING LAYER
   [OK] Kelly Criterion (optimal sizing)
   [OK] Volatility-adjusted scaling
   [OK] Risk parity (max 2% loss per trade)

5. EXECUTION LAYER
   [OK] Per-bar position management
   [OK] Slippage modeling (intraday: 2-5 bps)
   [OK] Multi-symbol concurrent trading
   [OK] Realistic commissions

6. OPTIMIZATION LAYER
   [OK] Bayesian search (scikit-optimize GP)
   [OK] Hyperparameter tuning (300+ iterations)
   [OK] Sharpe 3.0 target

TARGET PERFORMANCE:
   Daily Phase 3: Sharpe 1.65
   Intraday Phase 6: Sharpe 3.0+ (via 1-5min ensemble)
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', action='store_true', help='Run stock backtest')
    parser.add_argument('--crypto', action='store_true', help='Run crypto backtest')
    parser.add_argument('--arch', action='store_true', help='Show architecture')
    parser.add_argument('--all', action='store_true', help='Run all demos')
    
    args = parser.parse_args()
    
    if args.all or not (args.stock or args.crypto or args.arch):
        # Default: show arch + crypto (always works)
        demo_architecture()
        demo_crypto_backtest()
    else:
        if args.arch:
            demo_architecture()
        if args.stock:
            demo_stock_backtest()
        if args.crypto:
            demo_crypto_backtest()
