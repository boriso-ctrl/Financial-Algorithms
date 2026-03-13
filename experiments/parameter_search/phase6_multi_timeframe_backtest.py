#!/usr/bin/env python
"""
Multi-Timeframe Backtest Runner (15M + 1H)
Tests multi-timeframe confluence strategy for 2-5 daily trades.

Usage:
    python scripts/phase6_multi_timeframe_backtest.py --asset AAPL --output results_aapl_mtf.json
    python scripts/phase6_multi_timeframe_backtest.py --asset SPY --output results_spy_mtf.json

Expected output: JSON with entry signals, trades, and performance metrics
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_multi_timeframe import MultiTimeframeVotingStrategy
from financial_algorithms.backtest.tiered_exits import TieredExitManager, ExitScenario

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTimeframeBacktest:
    """Multi-timeframe backtest engine for 15M + 1H confluence strategy."""
    
    def __init__(
        self,
        asset: str,
        start_date: str = '2025-01-01',
        end_date: str = '2026-03-12',
        initial_equity: float = 100000,
    ):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        
        self.strategy = MultiTimeframeVotingStrategy(
            risk_pct=1.5,  # 1.5% per trade
            tp1_pct=1.0,   # 1% quick target
            tp2_pct=2.5,   # 2.5% full target
            min_confluence=4,  # Require 4/5 conditions
        )
        
        self.exit_manager = TieredExitManager(
            risk_pct=1.5,
            tp1_pct=1.0,
            tp2_pct=2.5,
        )
        
        self.trades = []
        self.signals = []
        self.equity_curve = []
        
        logger.info(f"Initialized multi-timeframe backtest for {asset}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load 15M and 1H data from yfinance."""
        logger.info(f"Loading {self.asset} 15M data...")
        
        df_15m = yf.download(
            self.asset,
            start=self.start_date,
            end=self.end_date,
            interval='15m',
            progress=False,
        )
        
        logger.info(f"Loading {self.asset} 1H data...")
        
        df_1h = yf.download(
            self.asset,
            start=self.start_date,
            end=self.end_date,
            interval='60m',
            progress=False,
        )
        
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
            df.fillna(method='ffill', inplace=True)
        
        logger.info(f"Loaded {len(df_15m)} 15M bars and {len(df_1h)} 1H bars")
        
        if df_15m.empty or df_1h.empty:
            raise ValueError("No data available for one or both timeframes")
        
        return df_15m, df_1h
    
    def backtest(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Dict:
        """
        Run multi-timeframe backtest.
        
        Synchronizes 15M and 1H data to generate signals at aligned times.
        """
        logger.info(f"Starting backtest on {len(df_15m)} 15M bars and {len(df_1h)} 1H bars")
        
        equity = self.initial_equity
        position = None
        trades = []
        signals = []
        daily_returns = []
        
        # Need warmup bars
        min_warmup_15m = 50
        min_warmup_1h = 35
        
        if len(df_15m) < min_warmup_15m or len(df_1h) < min_warmup_1h:
            logger.error("Insufficient data for warmup period")
            return {
                'status': 'FAILED',
                'error': 'Insufficient data',
                'asset': self.asset,
                'start_date': self.start_date,
                'end_date': self.end_date,
            }
        
        # Iterate through 15M bars (faster timeframe)
        for idx_15m in range(min_warmup_15m, len(df_15m)):
            timestamp_15m = df_15m.index[idx_15m]
            
            # Find corresponding 1H bar (use most recent completed 1H)
            idx_1h = None
            for i in range(len(df_1h)-1, -1, -1):
                if df_1h.index[i] <= timestamp_15m:
                    idx_1h = i
                    break
            
            if idx_1h is None or idx_1h < min_warmup_1h:
                continue
            
            # Prepare data arrays
            close_15m = df_15m['close'].iloc[:idx_15m+1].values
            high_15m = df_15m['high'].iloc[:idx_15m+1].values
            low_15m = df_15m['low'].iloc[:idx_15m+1].values
            volume_15m = df_15m['volume'].iloc[:idx_15m+1].values
            
            close_1h = df_1h['close'].iloc[:idx_1h+1].values
            high_1h = df_1h['high'].iloc[:idx_1h+1].values
            low_1h = df_1h['low'].iloc[:idx_1h+1].values
            
            current_price_15m = float(df_15m.iloc[idx_15m]['close'])
            daily_return = 0.0
            
            # Check for entry signal
            if position is None:
                should_enter, signal = self.strategy.generate_entry_signal(
                    close_15m, high_15m, low_15m, volume_15m,
                    close_1h, high_1h, low_1h,
                )
                
                if should_enter:
                    position_size = self.strategy.calculate_position_size(
                        signal.signal_strength,
                        equity
                    )
                    quantity = position_size / current_price_15m if current_price_15m > 0 else 0
                    
                    if quantity > 0:
                        position = self.exit_manager.create_trade(
                            symbol=self.asset,
                            entry_price=current_price_15m,
                            entry_time=timestamp_15m,
                            quantity=quantity,
                        )
                        
                        trades.append({
                            'entry_date': timestamp_15m,
                            'entry_price': current_price_15m,
                            'confluence_score': signal.confluence_score,
                            'm15_reversal': signal.m15_reversal,
                            'm15_momentum': signal.m15_momentum,
                            'h1_divergence': signal.h1_divergence,
                            'h1_trend': signal.h1_trend,
                            'h1_momentum': signal.h1_momentum,
                            'quantity': quantity,
                            'position_value': position_size,
                        })
                        
                        signals.append({
                            'timestamp': timestamp_15m,
                            'entry_price': current_price_15m,
                            'confluence': signal.confluence_score,
                        })
                        
                        logger.info(
                            f"{self.asset} | {timestamp_15m} | "
                            f"ENTRY @ ${current_price_15m:.2f} | "
                            f"Confluence {signal.confluence_score}/5"
                        )
            
            else:
                # Check for exit conditions
                if position.scenario == ExitScenario.SCENARIO_B:
                    self.exit_manager.update_trailing_stop(position, current_price_15m)
                
                if not position.tp1_hit:
                    self.exit_manager.evaluate_at_tp1(
                        position, current_price_15m, timestamp_15m, 0  # score=0
                    )
                
                exit_cond = self.exit_manager.check_exit_conditions(
                    position, current_price_15m, timestamp_15m, 0
                )
                
                if exit_cond:
                    exit_price = exit_cond['exit_price']
                    pl = exit_cond['pl']
                    
                    equity += pl
                    daily_return = (exit_price - position.entry_price) / position.entry_price
                    
                    trades[-1].update({
                        'exit_date': timestamp_15m,
                        'exit_price': exit_price,
                        'exit_reason': exit_cond['exit_reason'],
                        'pl': pl,
                        'return_pct': daily_return * 100,
                    })
                    
                    if len(trades) % 5 == 0:
                        logger.info(
                            f"{self.asset} | {timestamp_15m} | "
                            f"EXIT @ ${exit_price:.2f} | "
                            f"Return {daily_return*100:.2f}% | {exit_cond['exit_reason']}"
                        )
                    
                    position = None
            
            # Update equity curve
            if position is not None:
                mark_equity = self.initial_equity + sum(
                    t.get('pl', 0) for t in trades[:-1]
                ) + (current_price_15m - position.entry_price) * position.quantity
            else:
                mark_equity = equity
            
            self.equity_curve.append(mark_equity)
            daily_returns.append(daily_return)
        
        # Calculate metrics
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtest Results for {self.asset}")
        logger.info(f"{'='*60}")
        logger.info(f"Total signals: {len(signals)}")
        logger.info(f"Total trades: {len(trades)}")
        
        if len(trades) > 0:
            won = sum(1 for t in trades if t.get('return_pct', 0) > 0)
            win_rate = won / len(trades) * 100
            
            returns = np.array([t.get('return_pct', 0) / 100 for t in trades if 'return_pct' in t])
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            
            if np.std(returns) > 0:
                sharpe = (avg_return / np.std(returns)) * np.sqrt(252)
            else:
                sharpe = 0
            
            total_pl = sum(t.get('pl', 0) for t in trades)
            total_return = (total_pl / self.initial_equity) * 100
            best_trade = max([t.get('return_pct', 0) for t in trades]) if trades else 0
            worst_trade = min([t.get('return_pct', 0) for t in trades]) if trades else 0
        else:
            win_rate = 0
            sharpe = 0
            total_return = 0
            total_pl = 0
            avg_return = 0
            best_trade = 0
            worst_trade = 0
        
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Avg return per trade: {avg_return*100:.2f}%")
        logger.info(f"Sharpe ratio: {sharpe:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        logger.info(f"Total P&L: ${total_pl:.2f}")
        
        results = {
            'asset': self.asset,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_equity': self.initial_equity,
            'final_equity': equity,
            'total_pl': total_pl,
            'total_return_pct': total_return,
            'trade_count': len(trades),
            'signal_count': len(signals),
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_return * 100,
            'sharpe_ratio': sharpe,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,
            'trades': trades,
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-timeframe backtest (15M + 1H confluence)'
    )
    parser.add_argument('--asset', type=str, required=True, help='Asset ticker')
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2026-03-12',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    try:
        backtest = MultiTimeframeBacktest(
            asset=args.asset,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        
        df_15m, df_1h = backtest.load_data()
        results = backtest.backtest(df_15m, df_1h)
        
        # Save results
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
