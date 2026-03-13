#!/usr/bin/env python
"""
Multi-Timeframe Signal Diagnostic Tool
Analyzes signal generation to debug confluence detection.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_multi_timeframe import MultiTimeframeVotingStrategy
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def diagnose_signals(asset='SPY', start_date='2026-02-01', end_date='2026-03-12'):
    """Analyze signal generation across both timeframes."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Diagnosing signals for {asset} | {start_date} to {end_date}")
    logger.info(f"{'='*60}\n")
    
    # Load data
    print(f"Loading {asset} 1H data...")
    df_1h = yf.download(
        asset,
        start=start_date,
        end=end_date,
        interval='60m',
        progress=False,
    )
    
    print(f"Loading {asset} 15M data...")
    df_15m = yf.download(
        asset,
        start=start_date,
        end=end_date,
        interval='15m',
        progress=False,
    )
    
    if df_15m.empty or df_1h.empty:
        print(f"No data available!")
        return
    
    # Rename columns
    for df in [df_15m, df_1h]:
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        }, inplace=True)
    
    print(f"Data loaded: {len(df_15m)} 15M bars, {len(df_1h)} 1H bars\n")
    
    # Initialize strategy
    strategy = MultiTimeframeVotingStrategy(min_confluence=3)  # Relaxed threshold
    
    # Sample signal checks across the 15M timeline
    sample_indices = [
        len(df_15m) // 4,
        len(df_15m) // 2,
        3 * len(df_15m) // 4,
        len(df_15m) - 1,
    ]
    
    for idx_15m in sample_indices:
        if idx_15m < 50 or idx_15m < 35:
            continue
        
        timestamp_15m = df_15m.index[idx_15m]
        
        # Find corresponding 1H
        idx_1h = None
        for i in range(len(df_1h)-1, -1, -1):
            if df_1h.index[i] <= timestamp_15m:
                idx_1h = i
                break
        
        if idx_1h is None or idx_1h < 35:
            continue
        
        # Prepare arrays
        close_15m = df_15m['close'].iloc[:idx_15m+1].values
        high_15m = df_15m['high'].iloc[:idx_15m+1].values
        low_15m = df_15m['low'].iloc[:idx_15m+1].values
        volume_15m = df_15m['volume'].iloc[:idx_15m+1].values
        
        close_1h = df_1h['close'].iloc[:idx_1h+1].values
        high_1h = df_1h['high'].iloc[:idx_1h+1].values
        low_1h = df_1h['low'].iloc[:idx_1h+1].values
        
        # Check signals individually
        m15_reversal = strategy.detect_m15_reversal(close_15m, low_15m, volume_15m)
        m15_momentum = strategy.detect_m15_momentum(close_15m)
        h1_divergence = strategy.detect_h1_divergence(close_1h, low_1h)
        h1_trend = strategy.detect_h1_trend(close_1h, high_1h)
        h1_momentum = strategy.detect_h1_momentum(close_1h)
        
        confluence = sum([m15_reversal, m15_momentum, h1_divergence, h1_trend, h1_momentum])
        
        print(f"\n{timestamp_15m.strftime('%Y-%m-%d %H:%M')}")
        price_val = close_15m[-1] if isinstance(close_15m[-1], (int, float)) else close_15m[-1].item()
        print(f"  Price: ${price_val:.2f}")
        print(f"  Confluence: {confluence}/5")
        print(f"    15M Reversal: {'✓' if m15_reversal else '✗'}")
        print(f"    15M Momentum: {'✓' if m15_momentum else '✗'}")
        print(f"    1H Divergence: {'✓' if h1_divergence else '✗'}")
        print(f"    1H Trend: {'✓' if h1_trend else '✗'}")
        print(f"    1H Momentum: {'✓' if h1_momentum else '✗'}")


if __name__ == '__main__':
    diagnose_signals()
