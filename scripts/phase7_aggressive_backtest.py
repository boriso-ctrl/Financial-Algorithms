#!/usr/bin/env python
"""
Aggressive Growth Backtest Runner
Tests aggressive voting strategy with stacked entries on single asset.
Target: 15-20%+ returns, 3+ Sharpe ratio, beat SPY (87.67% in test period)

Usage:
    python scripts/phase7_aggressive_backtest.py --asset SPY --output aggressive_spy.json
    python scripts/phase7_aggressive_backtest.py --asset AAPL --output aggressive_aapl.json
    python scripts/phase7_aggressive_backtest.py --asset NVDA --output aggressive_nvda.json

Expected output: JSON with Sharpe, return %, trades count, win rate
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_aggressive_growth import AggressiveGrowthVotingStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StackedPosition:
    """Represents a single stacked position."""
    
    def __init__(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        quantity: float,
        position_size_dollars: float,
        stack_index: int,
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.quantity = quantity
        self.position_size = position_size_dollars
        self.stack_index = stack_index
        self.tp1_hit = False
        self.tp1_exit_price = None
        self.high_water_mark = entry_price
        self.trailing_stop = entry_price * 0.99
    
    def update_high_water_mark(self, current_price: float, tp2_pct: float):
        """Update trailing stop based on high water mark."""
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
            # Trailing stop at 1.5% below high water mark
            self.trailing_stop = current_price * (1 - 0.015)


class Phase7AggressiveBacktest:
    """Aggressive growth backtest with stacked positions."""
    
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
            tp1_pct=2.5,      # Extended from 1.5%
            tp2_pct=6.0,      # Extended from 3.0%
            min_buy_score=2.0,
            max_sell_score=-2.0,
            trailing_stop_pct=1.5,
            max_stacked_positions=3,
        )
        
        self.trades_log = []
        self.equity_curve = []
        self.positions: List[StackedPosition] = []
        
        logger.info(f"Initialized Aggressive backtest for {asset}")
        logger.info(f"Extended targets: TP1={self.strategy.tp1_pct}%, TP2={self.strategy.tp2_pct}%")
        logger.info(f"Aggressive sizing: 10-15% per position, max {self.strategy.max_stacked_positions} stacked")
    
    def load_data(self) -> pd.DataFrame:
        """Load OHLCV data from yfinance."""
        logger.info(f"Loading {self.asset} data from {self.start_date} to {self.end_date}")
        
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
        
        # Forward fill any NaNs
        df = df.fillna(method='ffill')
        
        logger.info(f"Loaded {len(df)} bars for {self.asset}")
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run backtest with stacked position support.
        
        Returns:
            Dict with metrics
        """
        logger.info(f"Starting aggressive backtest on {len(df)} bars")
        
        equity = self.initial_equity
        trades_closed = []
        daily_returns = []
        score_history = []
        
        for idx in range(50, len(df)):
            date = df.index[idx]
            row = df.iloc[idx]
            
            close_history = df['close'].iloc[:idx+1].values
            high_history = df['high'].iloc[:idx+1].values
            low_history = df['low'].iloc[:idx+1].values
            volume_history = df['volume'].iloc[:idx+1].values
            
            # Calculate signal
            score = self.strategy.calculate_voting_score(
                close=close_history,
                high=high_history,
                low=low_history,
                volume=volume_history,
            )
            score_history.append(score)
            
            current_price = float(row['close'])
            daily_return = 0.0
            
            # Check for new entry (stacked entry allowed)
            if self.strategy.should_enter(score) and len(self.positions) < self.strategy.max_stacked_positions:
                # Calculate available equity (exclude margin in positions)
                used_capital = sum(p.position_size for p in self.positions)
                available_equity = equity  # Assume we can still trade
                
                position_size = self.strategy.calculate_position_size(score, available_equity)
                if position_size > 0:
                    quantity = position_size / current_price if current_price > 0 else 0
                    
                    if quantity > 0:
                        pos = StackedPosition(
                            symbol=self.asset,
                            entry_price=current_price,
                            entry_time=date,
                            quantity=quantity,
                            position_size_dollars=position_size,
                            stack_index=len(self.positions),
                        )
                        self.positions.append(pos)
                        
                        logger.info(
                            f"{self.asset} | {date.date()} | ENTRY (Stack #{pos.stack_index+1}) | "
                            f"Price: {current_price:.2f} | Size: ${position_size:.0f} | "
                            f"Score: {score:.2f} | Active positions: {len(self.positions)}"
                        )
            
            # Update and check exit conditions for each stacked position
            positions_to_remove = []
            
            for pos in self.positions:
                current_return = (current_price - pos.entry_price) / pos.entry_price
                current_pl = current_return * pos.position_size
                
                # Update high water mark for trailing stop
                pos.update_high_water_mark(current_price, self.strategy.tp2_pct)
                
                # Check TP1 (2.5%) - exit 50% of position
                if not pos.tp1_hit and current_return >= (self.strategy.tp1_pct / 100):
                    pos.tp1_hit = True
                    pos.tp1_exit_price = current_price
                    exit_quantity = pos.quantity * 0.5
                    exit_pl = exit_quantity * (current_price - pos.entry_price)
                    equity += exit_pl
                    pos.quantity *= 0.5  # Remaining 50%
                    
                    logger.info(
                        f"{self.asset} | {date.date()} | TP1 EXIT (50%) | "
                        f"Price: {current_price:.2f} | PL: ${exit_pl:.2f} | "
                        f"Remaining: {pos.quantity:.2f} shares"
                    )
                
                # Check TP2 (6.0%) - exit remaining
                if current_return >= (self.strategy.tp2_pct / 100):
                    exit_pl = pos.quantity * (current_price - pos.entry_price)
                    equity += exit_pl
                    
                    logger.info(
                        f"{self.asset} | {date.date()} | TP2 EXIT (100%) | "
                        f"Price: {current_price:.2f} | PL: ${exit_pl:.2f} | "
                        f"Total return: {current_return*100:.2f}%"
                    )
                    
                    trades_closed.append({
                        'entry_date': pos.entry_time,
                        'entry_price': pos.entry_price,
                        'exit_date': date,
                        'exit_price': current_price,
                        'quantity': pos.quantity,
                        'position_size': pos.position_size,
                        'pl': exit_pl,
                        'return_pct': current_return * 100,
                        'exit_reason': 'TP2',
                    })
                    
                    positions_to_remove.append(pos)
                
                # Check trailing stop
                elif current_price < pos.trailing_stop:
                    exit_pl = pos.quantity * (current_price - pos.entry_price)
                    equity += exit_pl
                    
                    logger.info(
                        f"{self.asset} | {date.date()} | TRAILING STOP EXIT | "
                        f"Price: {current_price:.2f} | Stop: {pos.trailing_stop:.2f} | "
                        f"PL: ${exit_pl:.2f}"
                    )
                    
                    trades_closed.append({
                        'entry_date': pos.entry_time,
                        'entry_price': pos.entry_price,
                        'exit_date': date,
                        'exit_price': current_price,
                        'quantity': pos.quantity,
                        'position_size': pos.position_size,
                        'pl': exit_pl,
                        'return_pct': (current_price - pos.entry_price) / pos.entry_price * 100,
                        'exit_reason': 'TRAILING_STOP',
                    })
                    
                    positions_to_remove.append(pos)
            
            # Check for emergency exit on ALL positions
            if self.strategy.should_exit_all(score):
                logger.info(f"{self.asset} | {date.date()} | EMERGENCY EXIT (ALL) | Score: {score:.2f}")
                
                for pos in self.positions:
                    exit_pl = pos.quantity * (current_price - pos.entry_price)
                    equity += exit_pl
                    
                    trades_closed.append({
                        'entry_date': pos.entry_time,
                        'entry_price': pos.entry_price,
                        'exit_date': date,
                        'exit_price': current_price,
                        'quantity': pos.quantity,
                        'position_size': pos.position_size,
                        'pl': exit_pl,
                        'return_pct': (current_price - pos.entry_price) / pos.entry_price * 100,
                        'exit_reason': 'EMERGENCY_EXIT',
                    })
                    
                    logger.info(
                        f"  → Stack #{pos.stack_index+1} exited at {current_price:.2f}, PL: ${exit_pl:.2f}"
                    )
                
                positions_to_remove = self.positions[:]
            
            # Remove exited positions
            for pos in positions_to_remove:
                self.positions.remove(pos)
            
            # Calculate equity curve
            position_pl = sum((current_price - p.entry_price) * p.quantity for p in self.positions)
            current_equity = equity + position_pl
            self.equity_curve.append(current_equity)
            
            daily_returns.append(0.0)  # Basic return tracking
        
        # Log any unclosed positions
        for pos in self.positions:
            logger.warning(f"Position unclosed at end: {self.asset}, entry {pos.entry_price:.2f}, qty {pos.quantity:.2f}")
        
        # Metrics calculation
        logger.info(f"\n=== RESULTS ===")
        logger.info(f"Total trades closed: {len(trades_closed)}")
        
        if score_history:
            logger.info(f"Score stats - Min: {min(score_history):.2f}, Max: {max(score_history):.2f}, Mean: {np.mean(score_history):.2f}")
        
        if len(trades_closed) > 0:
            winning_trades = [t for t in trades_closed if t.get('pl', 0) > 0]
            losing_trades = [t for t in trades_closed if t.get('pl', 0) < 0]
            win_rate = len(winning_trades) / len(trades_closed) * 100
            
            returns = np.array([t.get('return_pct', 0) / 100 for t in trades_closed])
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio
            if std_return > 0:
                sharpe = (avg_return / std_return) * np.sqrt(252)
            else:
                sharpe = 0
            
            total_pl = sum(t.get('pl', 0) for t in trades_closed)
            total_return = (total_pl / self.initial_equity) * 100
            
            logger.info(f"Winning trades: {len(winning_trades)} ({win_rate:.1f}%)")
            logger.info(f"Losing trades: {len(losing_trades)}")
            logger.info(f"Average trade return: {avg_return*100:.2f}%")
            logger.info(f"Total PL: ${total_pl:.2f}")
            logger.info(f"Total return: {total_return:.2f}%")
            logger.info(f"Sharpe ratio: {sharpe:.2f}")
            logger.info(f"SPY benchmark return: 87.67%")
            logger.info(f"Outperformance: {total_return - 87.67:.2f}%")
            
        else:
            win_rate = 0
            sharpe = 0
            total_return = 0
            avg_return = 0
            total_pl = 0
        
        results = {
            'test_date': datetime.now().isoformat(),
            'asset': self.asset,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_equity': self.initial_equity,
            'final_equity': equity,
            'total_pl': total_pl,
            'total_return_pct': total_return,
            'trade_count': len(trades_closed),
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_return * 100,
            'sharpe_ratio': sharpe,
            'std_return_pct': np.std([t.get('return_pct', 0) for t in trades_closed]) if trades_closed else 0,
            'trades': trades_closed,
            'trading_mode': 'aggressive_growth',
            'position_sizing': '10-15%',
            'exit_targets': 'TP1=2.5% TP2=6.0%',
            'max_stacked_positions': self.strategy.max_stacked_positions,
            'benchmark_spy_return_pct': 87.67,
        }
        
        return results
    
    def run(self, output_file: Optional[str] = None) -> Dict[str, any]:
        """Run full backtest and optionally save results."""
        df = self.load_data()
        results = self.backtest(df)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Phase 7: Aggressive Growth Backtest')
    parser.add_argument('--asset', type=str, default='SPY', help='Asset to backtest')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2025-12-31', help='End date')
    
    args = parser.parse_args()
    
    backtest = Phase7AggressiveBacktest(
        asset=args.asset,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    results = backtest.run(output_file=args.output)
    
    print("\n" + "="*60)
    print(f"AGGRESSIVE GROWTH BACKTEST: {args.asset}")
    print("="*60)
    print(f"Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"Trades: {results['trade_count']}")
    print(f"vs SPY Benchmark: {results['total_return_pct'] - results['benchmark_spy_return_pct']:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
