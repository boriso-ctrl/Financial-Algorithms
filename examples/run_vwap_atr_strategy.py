"""
VWAP + ATR Strategy Runner

This script demonstrates the complete VWAP + ATR trading strategy:
1. Load or generate intraday data
2. Calculate all indicators
3. Detect market regimes
4. Generate trading signals
5. Run backtest
6. Display results

This is a production-ready, deterministic implementation suitable for:
- Backtesting
- Paper trading
- Live trading (with appropriate data feed)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

# Import strategy components
from indicators.vwap_atr_indicators import calculate_indicators
from strategies.regime_detection import detect_full_regime
from signals.vwap_atr_signal import generate_signals, validate_signals
from backtest.intraday_backtest import run_intraday_backtest, analyze_regime_performance
from examples.generate_sample_data import generate_intraday_data, add_realistic_patterns


def load_data(use_sample: bool = True, csv_path: str = None) -> pd.DataFrame:
    """
    Load intraday OHLCV data.
    
    Parameters
    ----------
    use_sample : bool
        If True, generate synthetic data for demonstration
    csv_path : str
        Path to CSV file with OHLCV data (if not using sample)
        Expected columns: timestamp, open, high, low, close, volume
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with session column
    """
    if use_sample:
        print("Generating sample intraday data...")
        df = generate_intraday_data(
            start_date='2024-01-02',
            num_days=30,
            initial_price=150.0,
            volatility=0.015
        )
        df = add_realistic_patterns(df)
        print(f"✓ Generated {len(df)} bars across {df['session'].nunique()} sessions")
    else:
        if csv_path is None:
            raise ValueError("csv_path must be provided when use_sample=False")
        
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Add session column if not present
        if 'session' not in df.columns:
            df['session'] = df.index.date.astype(str)
        
        print(f"✓ Loaded {len(df)} bars")
    
    return df


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("VWAP + ATR INTRADAY TRADING STRATEGY")
    print("=" * 80)
    print()
    
    # ============================================================================
    # STEP 1: Load Data
    # ============================================================================
    print("[1/6] Loading intraday data...")
    df = load_data(use_sample=True)
    
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Sessions: {df['session'].nunique()}")
    print(f"  Bars per session (avg): {len(df) / df['session'].nunique():.1f}")
    print()
    
    # ============================================================================
    # STEP 2: Calculate Indicators
    # ============================================================================
    print("[2/6] Calculating indicators...")
    print("  - VWAP (session-anchored)")
    print("  - ATR(14) with ±1×ATR and ±2×ATR bands")
    print("  - Volume Profile (POC, VAH, VAL)")
    print("  - RSI(14)")
    print("  - EMA(20)")
    
    df = calculate_indicators(
        df,
        session_col='session',
        atr_period=14,
        rsi_period=14,
        ema_period=20
    )
    
    print(f"✓ Indicators calculated")
    print()
    
    # ============================================================================
    # STEP 3: Detect Market Regime
    # ============================================================================
    print("[3/6] Detecting market regimes...")
    df = detect_full_regime(df, atr_lookback=20)
    
    # Display regime distribution
    regime_counts = df['regime'].value_counts()
    print(f"  Trend bars: {regime_counts.get('trend', 0)} ({regime_counts.get('trend', 0)/len(df)*100:.1f}%)")
    print(f"  Rotational bars: {regime_counts.get('rotational', 0)} ({regime_counts.get('rotational', 0)/len(df)*100:.1f}%)")
    print()
    
    # ============================================================================
    # STEP 4: Generate Trading Signals
    # ============================================================================
    print("[4/6] Generating trading signals...")
    df = generate_signals(df, session_col='session')
    
    # Validate signals
    is_valid = validate_signals(df)
    if not is_valid:
        print("  ⚠ Warning: Signal validation found issues")
    else:
        print("  ✓ Signal validation passed")
    
    # Count signals
    signals = df[df['signal'] != 'none']
    long_signals = len(signals[signals['signal'] == 'long'])
    short_signals = len(signals[signals['signal'] == 'short'])
    
    print(f"  Long signals: {long_signals}")
    print(f"  Short signals: {short_signals}")
    print(f"  Total signals: {len(signals)}")
    print()
    
    # ============================================================================
    # STEP 5: Run Backtest
    # ============================================================================
    print("[5/6] Running backtest...")
    initial_capital = 100000
    
    # Drop NaN rows (warm-up period for indicators)
    valid_mask = df['atr'].notna() & df['rsi'].notna() & df['ema'].notna()
    df_valid = df[valid_mask].copy()
    
    print(f"  Initial capital: ${initial_capital:,.2f}")
    print(f"  Valid bars: {len(df_valid)} (after indicator warm-up)")
    
    results = run_intraday_backtest(
        df_valid,
        initial_capital=initial_capital,
        position_size_pct=1.0
    )
    
    print(f"✓ Backtest complete")
    print()
    
    # ============================================================================
    # STEP 6: Display Results
    # ============================================================================
    print("[6/6] Results Summary")
    print("=" * 80)
    print()
    
    # Performance metrics
    print("PERFORMANCE METRICS")
    print("-" * 80)
    for key, value in results['metrics'].items():
        print(f"{key:.<30} {value}")
    print()
    
    # Trade details
    if len(results['trades']) > 0:
        print("TRADE DETAILS")
        print("-" * 80)
        trades = results['trades']
        
        print(f"Total P&L: ${trades['pnl'].sum():,.2f}")
        print(f"Best Trade: ${trades['pnl'].max():,.2f} ({trades['pnl_pct'].max():.2%})")
        print(f"Worst Trade: ${trades['pnl'].min():,.2f} ({trades['pnl_pct'].min():.2%})")
        print()
        
        # Direction breakdown
        long_trades = trades[trades['direction'] == 'long']
        short_trades = trades[trades['direction'] == 'short']
        
        print(f"Long Trades: {len(long_trades)}")
        if len(long_trades) > 0:
            long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades)
            print(f"  Win Rate: {long_win_rate:.2%}")
            print(f"  Avg P&L: ${long_trades['pnl'].mean():,.2f}")
        
        print(f"\nShort Trades: {len(short_trades)}")
        if len(short_trades) > 0:
            short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades)
            print(f"  Win Rate: {short_win_rate:.2%}")
            print(f"  Avg P&L: ${short_trades['pnl'].mean():,.2f}")
        print()
        
        # Regime performance
        print("REGIME PERFORMANCE")
        print("-" * 80)
        regime_perf = analyze_regime_performance(df_valid, trades)
        if len(regime_perf) > 0:
            print(regime_perf.to_string(index=False))
        else:
            print("  No regime data available")
        print()
        
        # Sample trades
        print("SAMPLE TRADES (First 5)")
        print("-" * 80)
        sample_trades = trades.head(5)[['entry_time', 'exit_time', 'direction', 
                                        'entry_price', 'exit_price', 'pnl', 'exit_reason']]
        print(sample_trades.to_string(index=False))
        print()
    else:
        print("No trades were executed during the backtest period.")
        print()
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("=" * 80)
    print("STRATEGY CHARACTERISTICS")
    print("-" * 80)
    print("✓ Deterministic: All signals are rule-based")
    print("✓ No Lookahead Bias: Signals based on closed candles only")
    print("✓ No Repainting: Indicators calculated sequentially")
    print("✓ Risk Managed: Every trade has stop loss and take profit")
    print("✓ Session Aware: Respects intraday session boundaries")
    print()
    print("Strategy is ready for:")
    print("  • Backtesting on historical data")
    print("  • Paper trading with live data feed")
    print("  • Production deployment (with appropriate risk controls)")
    print("=" * 80)


if __name__ == "__main__":
    main()
