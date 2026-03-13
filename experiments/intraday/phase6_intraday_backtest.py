#!/usr/bin/env python
"""
Phase 6: Intraday Multi-Timeframe Backtest Runner
Tests enhanced weighted voting system across multiple timeframes (5m, 15m, 1h, 4h).

Usage:
    python scripts/phase6_intraday_backtest.py --asset AAPL --timeframe 1h --output results_aapl_1h.json
    python scripts/phase6_intraday_backtest.py --asset SPY --timeframe 15m --output results_spy_15m.json
    python scripts/phase6_intraday_backtest.py --asset AAPL --timeframe all --output results_aapl_all_tf.json

Supported timeframes: 5m, 15m, 1h, 4h, all (runs all 4)
Expected output: JSON with Sharpe, return %, trades count, win rate, etc.
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_enhanced_weighted import EnhancedWeightedVotingStrategy
from financial_algorithms.strategies.voting_intraday_optimized import IntradayOptimizedVotingStrategy
from financial_algorithms.signals.enhanced_indicators import EnhancedIndicators
from financial_algorithms.backtest.tiered_exits import TieredExitManager, ExitScenario

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntradayBacktest:
    """Multi-timeframe intraday backtest for enhanced weighted voting system."""
    
    VALID_TIMEFRAMES = ['5m', '15m', '1h', '4h']
    
    def __init__(
        self,
        asset: str,
        timeframe: str = '1h',
        start_date: str = '2024-01-01',
        end_date: str = '2026-03-12',
        initial_equity: float = 100000,
    ):
        self.asset = asset
        self.timeframe = timeframe if timeframe != 'all' else '1h'  # Start with 1h for 'all' mode
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        
        # Use intraday-optimized strategy with shorter indicator periods
        self.strategy = IntradayOptimizedVotingStrategy(
            timeframe=timeframe,
            risk_pct=2.0,
            tp1_pct=1.5,
            tp2_pct=3.0,
        )
        
        self.exit_manager = TieredExitManager(
            risk_pct=2.0,
            tp1_pct=1.5,
            tp2_pct=3.0,
            trailing_stop_pct=1.0,
        )
        
        self.trades_log = []
        self.equity_curve = []
        
        logger.info(f"Initialized intraday backtest for {asset} on {self.timeframe}")
    
    def load_data(self, timeframe: str = None) -> pd.DataFrame:
        """Load OHLCV data from yfinance for specified timeframe."""
        if timeframe is None:
            timeframe = self.timeframe
        
        # Map timeframe to yfinance interval
        interval_map = {
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
        }
        
        if timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        interval = interval_map[timeframe]
        
        logger.info(f"Loading {self.asset} data from {self.start_date} to {self.end_date} | Interval: {interval}")
        
        try:
            df = yf.download(
                self.asset,
                start=self.start_date,
                end=self.end_date,
                interval=interval,
                progress=False,
            )
            
            if df.empty:
                logger.warning(f"No data found for {self.asset} on {timeframe}")
                return pd.DataFrame()
            
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
            
            logger.info(f"Loaded {len(df)} bars for {self.asset} on {timeframe}")
            
            # For intraday, need fewer bars for indicator warmup (50 is overkill on 5m)
            min_bars_needed = 50
            if len(df) < min_bars_needed:
                logger.warning(
                    f"Only {len(df)} bars available, need at least {min_bars_needed} "
                    f"for indicator warmup. Skipping {timeframe}."
                )
                return pd.DataFrame()
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data for {self.asset} on {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def backtest(self, df: pd.DataFrame, timeframe: str = None) -> Dict:
        """
        Run backtest on OHLCV data.
        
        Returns:
            Dict with metrics: sharpe, total_return, trades, win_rate, etc.
        """
        if timeframe is None:
            timeframe = self.timeframe
        
        if df.empty:
            logger.warning(f"Empty dataframe for {timeframe}, skipping backtest")
            return {
                'asset': self.asset,
                'timeframe': timeframe,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'status': 'FAILED',
                'error': 'No data available for timeframe',
            }
        
        logger.info(f"Starting backtest on {len(df)} bars ({timeframe})")
        
        equity = self.initial_equity
        position = None
        trades = []
        daily_returns = []
        score_history = []
        
        # Need 50 bars minimum for indicator warmup
        warmup_bars = min(50, len(df) // 4)
        
        for idx in range(warmup_bars, len(df)):
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
            
            daily_return = 0.0
            
            if position is None:
                # Check for entry
                if self.strategy.should_enter(score):
                    entry_price = float(row['close'])
                    position_size = self.strategy.calculate_position_size(score, equity)
                    quantity = position_size / entry_price if entry_price > 0 else 0
                    
                    if quantity > 0:
                        position = self.exit_manager.create_trade(
                            symbol=self.asset,
                            entry_price=entry_price,
                            entry_time=date,
                            quantity=quantity,
                        )
                        
                        trades.append({
                            'entry_date': date,
                            'entry_price': entry_price,
                            'entry_signal': score,
                            'quantity': quantity,
                            'position_value': position_size,
                        })
            
            else:
                # Update trailing stop if in Scenario B
                if position.scenario == ExitScenario.SCENARIO_B:
                    self.exit_manager.update_trailing_stop(position, float(row['close']))
                
                # Check for TP1
                if not position.tp1_hit:
                    self.exit_manager.evaluate_at_tp1(
                        position, float(row['close']), date, score
                    )
                
                # Check for exit
                exit_cond = self.exit_manager.check_exit_conditions(
                    position, float(row['close']), date, score
                )
                
                if exit_cond:
                    exit_price = exit_cond['exit_price']
                    pl = exit_cond['pl']
                    
                    equity += pl
                    daily_return = (exit_price - position.entry_price) / position.entry_price
                    
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': exit_price,
                        'exit_reason': exit_cond['exit_reason'],
                        'pl': pl,
                        'return_pct': daily_return * 100,
                        'scenario': position.scenario.value if position.scenario else 'UNKNOWN',
                    })
                    
                    if len(trades) % 10 == 0:  # Log every 10th trade
                        logger.info(
                            f"{self.asset} ({timeframe}) | {date} | "
                            f"Entry: {position.entry_price:.2f} → Exit: {exit_price:.2f} | "
                            f"Return: {daily_return*100:.2f}% | Reason: {exit_cond['exit_reason']}"
                        )
                    
                    position = None
            
            if position is not None:
                current_price = float(row['close'])
                mark_pl = (current_price - position.entry_price) * position.quantity
                current_equity = self.initial_equity - position.entry_price * position.quantity + current_price * position.quantity
            else:
                current_equity = equity
                mark_pl = 0
            
            daily_returns.append(daily_return)
            self.equity_curve.append(current_equity)
        
        # Metrics calculation
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.get('return_pct', 0) > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            returns = np.array([t.get('return_pct', 0) / 100 for t in trades if 'return_pct' in t])
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            
            # Sharpe calculation
            if np.std(returns) > 0:
                sharpe = (avg_return / np.std(returns)) * np.sqrt(252)  # Annualized
            else:
                sharpe = 0
            
            total_pl = sum(t.get('pl', 0) for t in trades)
            total_return = (total_pl / self.initial_equity) * 100
            best_trade = max([t.get('return_pct', 0) for t in trades])
            worst_trade = min([t.get('return_pct', 0) for t in trades])
        else:
            win_rate = 0
            sharpe = 0
            total_return = 0
            total_pl = 0
            avg_return = 0
            best_trade = 0
            worst_trade = 0
        
        results = {
            'asset': self.asset,
            'timeframe': timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_equity': self.initial_equity,
            'final_equity': equity,
            'total_pl': total_pl,
            'total_return_pct': total_return,
            'trade_count': len(trades),
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_return * 100,
            'sharpe_ratio': sharpe,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,
            'trades': trades,
        }
        
        logger.info(
            f"Results ({timeframe}): Sharpe={sharpe:.2f}, Return={total_return:.2f}%, "
            f"Trades={len(trades)}, WR={win_rate:.1f}%"
        )
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Intraday multi-timeframe backtest for enhanced weighted voting system'
    )
    parser.add_argument('--asset', type=str, required=True, help='Asset ticker (e.g., AAPL, SPY)')
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['5m', '15m', '1h', '4h', 'all'],
        help='Timeframe to test: 5m, 15m, 1h, 4h, or all'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date (YYYY-MM-DD), default: 2024-01-01'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2026-03-12',
        help='End date (YYYY-MM-DD), default: 2026-03-12'
    )
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--initial-equity', type=float, default=100000, help='Initial equity (default: 100000)')
    
    args = parser.parse_args()
    
    backtest = IntradayBacktest(
        asset=args.asset,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=args.initial_equity,
    )
    
    # Determine which timeframes to run
    timeframes = (
        IntradayBacktest.VALID_TIMEFRAMES
        if args.timeframe == 'all'
        else [args.timeframe]
    )
    
    all_results = {
        'asset': args.asset,
        'date_range': f"{args.start_date} to {args.end_date}",
        'timeframes_run': timeframes,
        'results': [],
    }
    
    for tf in timeframes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running backtest for {args.asset} on {tf}")
        logger.info(f"{'='*60}\n")
        
        df = backtest.load_data(tf)
        if not df.empty:
            results = backtest.backtest(df, tf)
            all_results['results'].append(results)
        else:
            logger.warning(f"Skipping {tf} due to insufficient data")
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {args.output}")
    else:
        print(json.dumps(all_results, indent=2, default=str))


if __name__ == '__main__':
    main()
