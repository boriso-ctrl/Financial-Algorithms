#!/usr/bin/env python
"""
Trend Following Aggressive Backtest
Simplified aggressive approach: Single larger position, extended targets (6%), no tight trailing stops.
Targets bull market uptrends (like 2023-2025).

Usage:
    python scripts/phase7_trend_following.py --asset SPY --output trend_spy.json
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_aggressive_growth import AggressiveGrowthVotingStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrendFollowingBacktest:
    """Single-position trend following with extended targets."""
    
    def __init__(
        self,
        asset: str,
        start_date: str = '2023-01-01',
        end_date: str = '2025-12-31',
        initial_equity: float = 100000,
    ):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        
        self.strategy = AggressiveGrowthVotingStrategy(
            risk_pct=2.0,
            tp1_pct=3.0,       # Larger initial target
            tp2_pct=6.0,       # Extended final target
            min_buy_score=2.0,
            max_sell_score=-2.0,
            trailing_stop_pct=3.0,  # Much wider trailing stop (only on retrace)
            max_stacked_positions=1,  # Single position only
        )
        
        self.trades_log = []
        self.equity_curve = []
        
        logger.info(f"Trend Following backtest for {asset}")
        logger.info(f"Single position | TP1=3% (exit 50%) | TP2=6% (exit 50%) | Wide trailing stop=3%")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from yfinance."""
        logger.info(f"Loading {self.asset} data")
        
        df = yf.download(
            self.asset,
            start=self.start_date,
            end=self.end_date,
            interval='1d',
            progress=False,
        )
        
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        })
        
        df = df.fillna(method='ffill')
        logger.info(f"Loaded {len(df)} bars")
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """Run backtest with single position per signal."""
        logger.info("Starting backtest")
        
        equity = self.initial_equity
        position = None
        trades_closed = []
        daily_returns = []
        
        for idx in range(50, len(df)):
            date = df.index[idx]
            row = df.iloc[idx]
            
            close_history = df['close'].iloc[:idx+1].values
            high_history = df['high'].iloc[:idx+1].values
            low_history = df['low'].iloc[:idx+1].values
            volume_history = df['volume'].iloc[:idx+1].values
            
            score = self.strategy.calculate_voting_score(
                close=close_history,
                high=high_history,
                low=low_history,
                volume=volume_history,
            )
            
            current_price = float(row['close'])
            
            # Enter on signal if no position
            if position is None and self.strategy.should_enter(score):
                position_size = self.strategy.calculate_position_size(score, equity)
                if position_size > 0:
                    quantity = position_size / current_price
                    position = {
                        'entry_price': current_price,
                        'entry_date': date,
                        'quantity': quantity,
                        'position_size': position_size,
                        'entry_score': score,
                        'tp1_hit': False,
                        'high_water': current_price,
                    }
                    logger.info(f"{date.date()} ENTRY: {current_price:.2f}, size ${position_size:.0f}, score {score:.2f}")
            
            elif position is not None:
                # Update high water mark
                position['high_water'] = max(position['high_water'], current_price)
                
                ret = (current_price - position['entry_price']) / position['entry_price']
                
                # Exit 50% at TP1 (3%)
                if not position['tp1_hit'] and ret >= 0.03:
                    position['tp1_hit'] = True
                    exit_qty = position['quantity'] * 0.5
                    exit_pl = exit_qty * (current_price - position['entry_price'])
                    equity += exit_pl
                    position['quantity'] *= 0.5
                    logger.info(f"{date.date()} TP1 (3%): exit 50%, PL ${exit_pl:.0f}")
                
                # Exit remaining 50% at TP2 (6%)
                if ret >= 0.06:
                    exit_pl = position['quantity'] * (current_price - position['entry_price'])
                    equity += exit_pl
                    
                    trades_closed.append({
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': date,
                        'exit_price': current_price,
                        'return_pct': ret * 100,
                        'pl': exit_pl,
                        'exit_reason': 'TP2',
                        'entry_score': position['entry_score'],
                    })
                    logger.info(f"{date.date()} TP2 (6%): exit 50%, PL ${exit_pl:.0f}, total return {ret*100:.2f}%")
                    position = None
                
                # Stop loss or emergency exit at -2 score
                elif ret < -0.02 or self.strategy.should_exit_all(score):
                    if self.strategy.should_exit_all(score):
                        reason = 'EMERGENCY_EXIT'
                    else:
                        reason = 'STOP_LOSS'
                    
                    exit_pl = position['quantity'] * (current_price - position['entry_price'])
                    equity += exit_pl
                    
                    trades_closed.append({
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': date,
                        'exit_price': current_price,
                        'return_pct': ret * 100,
                        'pl': exit_pl,
                        'exit_reason': reason,
                        'entry_score': position['entry_score'],
                    })
                    logger.info(f"{date.date()} {reason}: exit at {current_price:.2f}, PL ${exit_pl:.0f}")
                    position = None
            
            # Equity curve
            if position is not None:
                pos_pl = (current_price - position['entry_price']) * position['quantity']
                current_equity = equity + pos_pl
            else:
                current_equity = equity
            
            self.equity_curve.append(current_equity)
            daily_returns.append(0.0)
        
        # Metrics
        logger.info(f"\n=== RESULTS ===")
        logger.info(f"Total trades: {len(trades_closed)}")
        
        if len(trades_closed) > 0:
            wins = [t for t in trades_closed if t['pl'] > 0]
            losses = [t for t in trades_closed if t['pl'] < 0]
            win_rate = 100 * len(wins) / len(trades_closed)
            
            returns = np.array([t['return_pct'] / 100 for t in trades_closed])
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
            
            total_pl = sum(t['pl'] for t in trades_closed)
            total_ret = 100 * total_pl / self.initial_equity
            
            logger.info(f"Wins: {len(wins)}/{len(trades_closed)} ({win_rate:.1f}%)")
            logger.info(f"Return: {total_ret:.2f}%")
            logger.info(f"Sharpe: {sharpe:.2f}")
            logger.info(f"Avg trade: {avg_ret*100:.2f}%")
            logger.info(f"vs SPY (87.67%): {total_ret - 87.67:.2f}%")
        else:
            win_rate = sharpe = total_ret = avg_ret = 0
            total_pl = 0
        
        results = {
            'test_date': datetime.now().isoformat(),
            'asset': self.asset,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return_pct': total_ret,
            'trade_count': len(trades_closed),
            'win_rate_pct': win_rate,
            'sharpe_ratio': sharpe,
            'avg_trade_return_pct': avg_ret * 100,
            'total_pl': total_pl,
            'trades': trades_closed,
            'trading_mode': 'trend_following_single_position',
            'position_size': '10-15% per signal',
            'targets': 'TP1=3% (50%), TP2=6% (50%)',
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', type=str, default='SPY')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    bt = TrendFollowingBacktest(asset=args.asset)
    results = bt.run(output_file=args.output)
    
    print()
    print("="*60)
    print(f"TREND FOLLOWING: {args.asset}")
    print("="*60)
    print(f"Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"Trades: {results['trade_count']}")
    print(f"vs SPY: {results['total_return_pct'] - 87.67:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
