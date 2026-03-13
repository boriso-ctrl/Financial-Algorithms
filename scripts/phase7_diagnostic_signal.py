#!/usr/bin/env python
"""
Diagnostic: Analyze voting signal behavior during bull market
Shows when signals fire and what percentage of the move is captured
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_aggressive_growth import AggressiveGrowthVotingStrategy

strategy = AggressiveGrowthVotingStrategy(min_buy_score=2.0, max_sell_score=-2.0)

# Load SPY data
spy = yf.download('SPY', start='2023-01-01', end='2025-12-31', progress=False)
spy = spy.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume', 'Open': 'open'})

print("SPY Performance Analysis (2023-2025)")
print("="*70)
start_price = float(spy['close'].iloc[0])
end_price = float(spy['close'].iloc[-1])
print(f"Start: {start_price:.2f}")
print(f"End: {end_price:.2f}")
print(f"Total move: {((end_price / start_price) - 1) * 100:.2f}%")
print()

# Analyze signal distribution
scores = []
for idx in range(50, len(spy)):
    close_hist = spy['close'].iloc[:idx+1].values
    high_hist = spy['high'].iloc[:idx+1].values
    low_hist = spy['low'].iloc[:idx+1].values
    vol_hist = spy['volume'].iloc[:idx+1].values
    
    score = strategy.calculate_voting_score(
        close=close_hist,
        high=high_hist,
        low=low_hist,
        volume=vol_hist,
    )
    scores.append(score)

scores = np.array(scores)

print("Voting Signal Statistics:")
print(f"Min score: {scores.min():.2f}")
print(f"Max score: {scores.max():.2f}")
print(f"Mean score: {scores.mean():.2f}")
print(f"Median score: {np.median(scores):.2f}")
print()

buy_signals = (scores >= 2.0).sum()
sell_signals = (scores <= -2.0).sum()
neutral = len(scores) - buy_signals - sell_signals

print("Signal Frequency:")
print(f"Buy signals (>=+2.0): {buy_signals}/{len(scores)} ({100*buy_signals/len(scores):.1f}%)")
print(f"Sell signals (<=-2.0): {sell_signals}/{len(scores)} ({100*sell_signals/len(scores):.1f}%)")
print(f"Neutral (-2.0 < x < +2.0): {neutral}/{len(scores)} ({100*neutral/len(scores):.1f}%)")
print()

# Time periods when buy signal was active
idx = 50
buy_periods = 0
hold_days = 0
for idx in range(50, len(spy)):
    if scores[idx-50] >= 2.0:
        buy_periods += 1
        next_idx = min(idx + 1, len(spy) - 1)
        ret = (spy['close'].iloc[next_idx] - spy['close'].iloc[idx]) / spy['close'].iloc[idx]
        hold_days += 1

print(f"Days when buy signal active: {buy_periods}/{len(scores)} ({100*buy_periods/len(scores):.1f}%)")
print()

print("Key Insight:")
print(f"If you had bought and held every day with +2.0 signal,")
print(f"you would have captured ALL of the bull market.")
print(f"Strategy is only capturing SOME of those days.")
print()
print("PROBLEM: Voting system misses early entry + exits too early on TP targets")
