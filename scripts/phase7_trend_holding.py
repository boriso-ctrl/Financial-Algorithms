#!/usr/bin/env python
"""
Trend Holding Strategy: Hold on Strength, Exit on Reversal
Matches market regime: In 58.6% of the time with buy signal, out when reversal detected.
No micro-managing with small TP targets - let trends run.

Usage:
    python scripts/phase7_trend_holding.py --asset SPY --output holding_spy.json
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from financial_algorithms.strategies.voting_aggressive_growth import AggressiveGrowthVotingStrategy


class TrendHoldingBacktest:
    """Hold on signal strength, exit on signal reversal."""
    
    def __init__(
        self,
        asset: str,
        start_date: str = '2023-01-01',
        end_date: str = '2025-12-31',
        initial_equity: float = 100000,
        position_size_pct: float = 15.0,  # Aggressive: 15% per signal
    ):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.position_size_pct = position_size_pct
        
        self.strategy = AggressiveGrowthVotingStrategy(
            min_buy_score=2.0,
            max_sell_score=-2.0,
        )
        
        logger.info(f"Trend Holding Setup: Hold on +2.0 signal, exit on -2.0 reversal")
        logger.info(f"Position sizing: {position_size_pct}% of equity")
    
    def load_data(self) -> pd.DataFrame:
        df = yf.download(self.asset, start=self.start_date, end=self.end_date, progress=False, interval='1d')
        df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume', 'Open': 'open'})
        df = df.fillna(method='ffill')
        logger.info(f"Loaded {len(df)} bars")
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """Run holding-based backtest."""
        logger.info("Starting backtest - hold on signal, exit on reversal")
        
        equity = self.initial_equity
        position = None
        trades = []
        daily_returns = []
        
        # Weekly summary
        weekly_summary = {'buy_weeks': 0, 'high_return': 0, 'entry_count': 0, 'exit_count': 0}
        
        for idx in range(50, len(df)):
            date = df.index[idx]
            row = df.iloc[idx]
            
            # Calculate score
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
            
            # ENTER: Score >= +2.0 and no current position
            if position is None and score >= 2.0:
                position_size = self.initial_equity * self.position_size_pct / 100
                quantity = position_size / current_price
                
                position = {
                    'entry_price': current_price,
                    'entry_date': date,
                    'entry_score': score,
                    'quantity': quantity,
                    'position_size': position_size,
                    'high_water': current_price,
                }
                
                weekly_summary['entry_count'] += 1
                logger.info(f"{date.date()} ENTER: {current_price:.2f}, {position_size/1000:.0f}k, score={score:.2f}")
            
            # UPDATE: Track water mark
            if position is not None:
                position['high_water'] = max(position['high_water'], current_price)
                ret = (current_price - position['entry_price']) / position['entry_price'] * 100
                if ret > weekly_summary['high_return']:
                    weekly_summary['high_return'] = ret
            
            # EXIT: Score <= -2.0 or end of data
            if position is not None and (score <= -2.0 or idx == len(df) - 1):
                exit_price = current_price
                exit_ret = (exit_price - position['entry_price']) / position['entry_price']
                exit_pl = position['quantity'] * (exit_price - position['entry_price'])
                equity += exit_pl
                
                hold_days = (date - position['entry_date']).days
                reason = 'REVERSAL' if score <= -2.0 else 'END'
                
                weekly_summary['exit_count'] += 1
                
                logger.info(
                    f"{date.date()} EXIT ({reason}): {exit_price:.2f}, "
                    f"PL ${exit_pl:,.0f}, Ret {exit_ret*100:+.2f}%, "
                    f"Held {hold_days}d, Score={score:.2f}"
                )
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'exit_date': date,
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'return_pct': exit_ret * 100,
                    'pl': exit_pl,
                    'hold_days': hold_days,
                    'exit_reason': reason,
                    'entry_score': position['entry_score'],
                    'exit_score': score,
                })
                
                position = None
        
        # Metrics
        logger.info(f"\n=== RESULTS ===")
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
            
            logger.info(f"Wins: {len(wins)}/{len(trades)} ({win_rate:.1f}%)")
            logger.info(f"Losses: {len(losses)}")
            logger.info(f"Avg hold: {avg_hold:.0f} days")
            logger.info(f"Total return: {total_ret:.2f}%")
            logger.info(f"Avg trade: {avg_ret*100:.2f}%")
            logger.info(f"Sharpe: {sharpe:.2f}")
            logger.info(f"vs SPY (87.67%): {total_ret - 87.67:.2f}%")
            
            if total_ret > 0:
                logger.info(f"✓ POSITIVE RETURN!")
            if sharpe > 3.0:
                logger.info(f"✓ GOOD SHARPE RATIO (>{sharpe:.1f}%)")
        else:
            logger.warning("No trades completed")
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
            'total_pl': total_pl,
            'trading_mode': 'trend_holding',
            'strategy_description': 'Hold on +2.0 signal, exit on -2.0 reversal',
            'position_size_pct': self.position_size_pct,
            'signal_entry': 2.0,
            'signal_exit': -2.0,
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
            logger.info(f"Saved to {output_file}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', default='SPY')
    parser.add_argument('--output', default=None)
    parser.add_argument('--position-size', type=float, default=15.0)
    args = parser.parse_args()
    
    bt = TrendHoldingBacktest(asset=args.asset, position_size_pct=args.position_size)
    results = bt.run(output_file=args.output)
    
    print()
    print("="*70)
    print(f"TREND HOLDING BACKTEST: {args.asset}")
    print("="*70)
    print(f"Return:    {results['total_return_pct']:>7.2f}%  (SPY: 87.67%)")
    print(f"Sharpe:    {results['sharpe_ratio']:>7.2f}x")
    print(f"Win Rate:  {results['win_rate_pct']:>7.1f}%")
    print(f"Trades:    {results['trade_count']:>7.0f}")
    print(f"Avg Trade: {results['avg_trade_return_pct']:>7.2f}%")
    print()
    gap = results['total_return_pct'] - 87.67
    if gap > 0:
        print(f"✓ OUTPERFORMING by {gap:.2f}%")
    else:
        print(f"✗ Underperforming by {abs(gap):.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
