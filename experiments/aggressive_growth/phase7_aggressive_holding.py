#!/usr/bin/env python
"""
Aggressive Trend Holding: Early Entry, Late Exit
Enter on +1.5 signal (earlier), exit on -3.0 signal (later pullbacks allowed).
Target: Capture more of 87.67% SPY bull move.

Usage:
    python scripts/phase7_aggressive_holding.py --asset SPY --output aggressive_holding_spy.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from financial_algorithms.strategies.voting_aggressive_growth import AggressiveGrowthVotingStrategy


class AggressiveHoldingBacktest:
    """Early entry (+1.5), late exit (-3.0) for max trend capture."""
    
    def __init__(
        self,
        asset: str,
        start_date: str = '2023-01-01',
        end_date: str = '2025-12-31',
        initial_equity: float = 100000,
        entry_score: float = 1.5,      # Early entry
        exit_score: float = -3.0,       # Late exit (allow mild pullbacks)
        position_size_pct: float = 15.0,
    ):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.entry_score = entry_score
        self.exit_score = exit_score
        self.position_size_pct = position_size_pct
        
        self.strategy = AggressiveGrowthVotingStrategy(
            min_buy_score=entry_score,
            max_sell_score=exit_score,
        )
        
        logger.info(f"Aggressive Trend Holding: Entry={entry_score}, Exit={exit_score}, Size={position_size_pct}%")
    
    def load_data(self) -> pd.DataFrame:
        df = yf.download(self.asset, start=self.start_date, end=self.end_date, progress=False, interval='1d')
        df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume', 'Open': 'open'})
        df = df.ffill()
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        equity = self.initial_equity
        position = None
        trades = []
        
        for idx in range(50, len(df)):
            date = df.index[idx]
            row = df.iloc[idx]
            
            close_hist = df['close'].iloc[:idx+1].values
            high_hist = df['high'].iloc[:idx+1].values
            low_hist = df['low'].iloc[:idx+1].values
            vol_hist = df['volume'].iloc[:idx+1].values
            
            score = self.strategy.calculate_voting_score(
                close=close_hist,
                high=high_hist,
                low=low_hist,
                volume=vol_hist,
            )
            
            current_price = float(row['close'])
            
            # ENTER
            if position is None and score >= self.entry_score:
                pos_size = self.initial_equity * self.position_size_pct / 100
                qty = pos_size / current_price
                position = {
                    'entry_price': current_price,
                    'entry_date': date,
                    'entry_score': score,
                    'quantity': qty,
                    'position_size': pos_size,
                }
                logger.info(f"{date.date()} ENTER @ {current_price:7.2f}, ${pos_size/1000:5.1f}k, score={score:.2f}")
            
            # EXIT
            if position is not None and (score <= self.exit_score or idx == len(df) - 1):
                exit_price = current_price
                exit_ret = (exit_price - position['entry_price']) / position['entry_price']
                exit_pl = position['quantity'] * (exit_price - position['entry_price'])
                equity += exit_pl
                
                hold_days = (date - position['entry_date']).days
                reason = 'REVERSAL' if score <= self.exit_score else 'END'
                
                logger.info(
                    f"{date.date()} EXIT({reason:8s}) @ {exit_price:7.2f}, "
                    f"PL ${exit_pl:7,.0f}, Return {exit_ret*100:+6.2f}%, "
                    f"Hold {hold_days:3d}d, Score={score:.2f}"
                )
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': date,
                    'exit_price': exit_price,
                    'return_pct': exit_ret * 100,
                    'pl': exit_pl,
                    'hold_days': hold_days,
                    'exit_reason': reason,
                })
                
                position = None
        
        # Metrics
        logger.info(f"\n{'='*70}")
        logger.info(f"Total trades: {len(trades)}")
        
        if trades:
            wins = [t for t in trades if t['pl'] > 0]
            losses = [t for t in trades if t['pl'] < 0]
            win_rate = 100 * len(wins) / len(trades)
            
            returns = np.array([t['return_pct'] / 100 for t in trades])
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
            
            total_pl = sum(t['pl'] for t in trades)
            total_ret = 100 * total_pl / self.initial_equity
            avg_hold = np.mean([t['hold_days'] for t in trades])
            
            logger.info(f"Wins: {len(wins)}/{len(trades)} ({win_rate:.1f}%) | Losses: {len(losses)}")
            logger.info(f"Avg hold: {avg_hold:.0f} days")
            logger.info(f"Total return: {total_ret:.2f}% | Avg trade: {avg_ret*100:.2f}%")
            logger.info(f"Sharpe: {sharpe:.2f} | vs SPY (87.67%): {total_ret - 87.67:.2f}%")
        else:
            logger.warning("No trades!")
            total_ret = sharpe = win_rate = avg_ret = 0
            total_pl = 0
        
        results = {
            'test_date': datetime.now().isoformat(),
            'asset': self.asset,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return_pct': total_ret,
            'trade_count': len(trades),
            'win_rate_pct': win_rate if trades else 0,
            'sharpe_ratio': sharpe,
            'avg_trade_return_pct': avg_ret * 100 if trades else 0,
            'trading_mode': f'aggressive_holding_entry{self.entry_score}_exit{self.exit_score}',
            'entry_score': self.entry_score,
            'exit_score': self.exit_score,
            'position_size_pct': self.position_size_pct,
            'trades': trades,
            'benchmark_spy': 87.67,
        }
        
        return results
    
    def run(self, output_file: Optional[str] = None):
        df = self.load_data()
        results = self.backtest(df)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', default='SPY')
    parser.add_argument('--entry', type=float, default=1.5, help='Entry score')
    parser.add_argument('--exit', type=float, default=-3.0, help='Exit score')
    parser.add_argument('--size', type=float, default=15.0, help='Position size %')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    bt = AggressiveHoldingBacktest(
        asset=args.asset,
        entry_score=args.entry,
        exit_score=args.exit,
        position_size_pct=args.size,
    )
    results = bt.run(output_file=args.output)
    
    print()
    print("="*70)
    print(f"AGGRESSIVE HOLDING: {args.asset} (Entry={args.entry}, Exit={args.exit})")
    print("="*70)
    print(f"Return:    {results['total_return_pct']:>7.2f}%  (vs SPY: 87.67%)")
    print(f"Sharpe:    {results['sharpe_ratio']:>7.2f}")
    print(f"Win Rate:  {results['win_rate_pct']:>7.1f}%")
    print(f"Trades:    {results['trade_count']:>7.0f}")
    if results['total_return_pct'] > 0:
        print(f"✓ POSITIVE RETURN")
    if results['sharpe_ratio'] > 3.0:
        print(f"✓ SHARPE > 3.0")
    gap = results['total_return_pct'] - 87.67
    if gap > 0:
        print(f"✓ BEATING SPY by {gap:.2f}%")
    else:
        print(f"Gap to SPY: {gap:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
