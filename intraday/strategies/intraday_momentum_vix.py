"""
Intraday Momentum Strategy with VIX Adaptation
=============================================

Multi-timeframe intraday system:
- Primary: 1H (signal generation)
- Confirm: 4H (trend context)
- Entry: 15m (precise execution)
- Volatility: VIX-based position sizing & stop-loss scaling
- Session: NY market hours 9:30-15:00

Goal: Outperform S&P 500 (87.67% benchmark)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIXAdaptiveIntraday:
    """Multi-timeframe intraday strategy with VIX volatility adaptation"""
    
    def __init__(self, asset='SPY', start_date='2023-01-01', end_date='2025-12-31'):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.equity = 100000  # Starting capital
        self.cash = self.equity
        self.position = 0
        self.position_price = 0
        self.trades = []
        
        # Configuration
        self.max_position_size = 0.10  # 10% of capital per trade
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.adx_min = 25  # Strong trend threshold
        
        # VIX-based sizing
        self.vix_normal = 20  # < 20 = normal
        self.vix_elevated = 30  # 20-30 = elevated
        
        logger.info(f"Initialized IntraDay Strategy for {asset}")
        
    def fetch_data(self):
        """Fetch 1H, 4H, 15m and VIX data"""
        logger.info(f"Fetching data for {self.asset}...")
        
        try:
            # Main asset data
            self.data_1h = yf.download(
                self.asset, 
                start=self.start_date, 
                end=self.end_date, 
                interval='1h',
                progress=False
            )
            
            self.data_4h = yf.download(
                self.asset, 
                start=self.start_date, 
                end=self.end_date, 
                interval='4h',
                progress=False
            )
            
            self.data_15m = yf.download(
                self.asset, 
                start=self.start_date, 
                end=self.end_date, 
                interval='15m',
                progress=False
            )
            
            # VIX data for volatility adaptation
            self.vix_data = yf.download(
                '^VIX', 
                start=self.start_date, 
                end=self.end_date, 
                progress=False
            )
            
            logger.info(f"1H bars: {len(self.data_1h)}, 4H bars: {len(self.data_4h)}, 15m bars: {len(self.data_15m)}, VIX: {len(self.vix_data)}")
            
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return False
        
        return True
    
    def calculate_indicators_1h(self):
        """Calculate indicators on 1H timeframe (signal generation)"""
        df = self.data_1h.copy()
        
        # RSI (14-period for 1H = ~14 hours)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # ADX (14-period - trend strength)
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        tr = np.maximum(df['High'] - df['Low'], 
                       np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                abs(df['Low'] - df['Close'].shift(1))))
        atr = pd.Series(tr).rolling(window=14).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()
        
        # ATR (14-period - volatility)
        df['ATR'] = atr
        
        # Volume-weighted MA
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Bollinger Bands (20, 2)
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA20'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['SMA20'] - (df['BB_Std'] * 2)
        
        self.data_1h = df
        return df
    
    def calculate_indicators_4h(self):
        """Calculate indicators on 4H timeframe (trend context)"""
        df = self.data_4h.copy()
        
        # Simple trend: SMA50 vs current
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['Trend'] = np.where(df['Close'] > df['SMA50'], 1, -1)  # 1 = uptrend, -1 = downtrend
        
        # EMA200 for long-term direction
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        df['LongTerm'] = np.where(df['Close'] > df['EMA200'], 1, -1)
        
        # RSI on 4H (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        self.data_4h = df
        return df
    
    def get_vix_multiplier(self, vix_close):
        """Calculate position sizing multiplier based on VIX level"""
        if vix_close < self.vix_normal:
            return 1.0  # Normal sizing
        elif vix_close < self.vix_elevated:
            return 0.7  # Reduced in elevated vol
        else:
            return 0.4  # Conservative in high vol
    
    def check_confluence_signal(self, idx_1h):
        """Generate signal only on strong confluence (avoid false signals)"""
        # Need valid indicators
        if idx_1h < 30:
            return None
        
        row = self.data_1h.iloc[idx_1h]
        
        # Find closest 4H bar (4H = 4x frequency)
        idx_4h = idx_1h // 4
        if idx_4h >= len(self.data_4h):
            return None
        
        row_4h = self.data_4h.iloc[idx_4h]
        
        # Get current VIX
        time_idx = row.name
        vix_val = self.vix_data.loc[:time_idx, 'Close'].iloc[-1] if len(self.vix_data.loc[:time_idx]) > 0 else 20
        
        # ENTRY LOGIC: Strong upside confluence
        bullish = (
            (row['RSI'] > 40) and (row['RSI'] < 70) and  # Not overbought
            (row['MACD'] > row['Signal']) and  # MACD bullish
            (row['ADX'] > self.adx_min) and  # Strong trend
            (row['Volume_Ratio'] > 1.2) and  # Volume confirmation
            (row['Close'] > row['SMA20']) and  # Above 20-SMA
            (row_4h['Trend'] == 1)  # 4H uptrend
        )
        
        # ENTRY LOGIC: Strong downside confluence
        bearish = (
            (row['RSI'] < 60) and (row['RSI'] > 30) and  # Not oversold
            (row['MACD'] < row['Signal']) and  # MACD bearish
            (row['ADX'] > self.adx_min) and  # Strong trend
            (row['Volume_Ratio'] > 1.2) and  # Volume confirmation
            (row['Close'] < row['SMA20']) and  # Below 20-SMA
            (row_4h['Trend'] == -1)  # 4H downtrend
        )
        
        # EXIT LOGIC: Close positions on reversal
        exit_signal = (
            ((row['RSI'] >= 70) or (row['Close'] > row['BB_Upper'])) or  # Overbought exit
            ((row['RSI'] <= 30) or (row['Close'] < row['BB_Lower']))  # Oversold exit
        )
        
        if bullish:
            return {
                'direction': 1,  # Long
                'strength': row['ADX'],
                'vix': vix_val,
                'rsi': row['RSI']
            }
        elif bearish:
            return {
                'direction': -1,  # Short
                'strength': row['ADX'],
                'vix': vix_val,
                'rsi': row['RSI']
            }
        elif exit_signal and self.position != 0:
            return {'direction': 0, 'type': 'exit'}  # Close position
        
        return None
    
    def calculate_dynamic_stops(self, entry_price, direction, vix_val, atr_val):
        """Calculate SL/TP based on volatility and ATR"""
        # Base SL: 1.5x ATR
        sl_distance = atr_val * 1.5
        
        # Scale by VIX (high vol = wider stops)
        if vix_val > self.vix_elevated:
            sl_distance *= 1.3
        
        # Calculate stops
        if direction == 1:  # Long
            sl = entry_price - sl_distance
            tp1 = entry_price + (atr_val * 1.0)  # First target: 1x ATR
            tp2 = entry_price + (atr_val * 2.5)  # Final target: 2.5x ATR
        else:  # Short
            sl = entry_price + sl_distance
            tp1 = entry_price - (atr_val * 1.0)
            tp2 = entry_price - (atr_val * 2.5)
        
        return {'sl': sl, 'tp1': tp1, 'tp2': tp2}
    
    def run_backtest(self):
        """Run backtest with intraday signals"""
        logger.info("Starting backtest...")
        
        self.calculate_indicators_1h()
        self.calculate_indicators_4h()
        
        for idx in range(30, len(self.data_1h)):  # Skip warmup
            current_time = self.data_1h.index[idx]
            current_price = float(self.data_1h['Close'].iloc[idx])
            current_atr = float(self.data_1h['ATR'].iloc[idx])
            
            # Skip if no position - wait for entry signal
            if self.position == 0:
                signal = self.check_confluence_signal(idx)
                
                if signal and signal.get('direction') != 0:
                    # Calculate position size
                    vix_val = signal['vix']
                    vix_mult = self.get_vix_multiplier(vix_val)
                    position_size = int((self.cash * self.max_position_size * vix_mult) / current_price)
                    
                    if position_size > 0:
                        self.position = position_size * signal['direction']
                        self.position_price = current_price
                        
                        # Calculate stops
                        stops = self.calculate_dynamic_stops(
                            current_price, 
                            signal['direction'], 
                            vix_val, 
                            current_atr
                        )
                        
                        trade = {
                            'entry_date': current_time,
                            'entry_price': current_price,
                            'quantity': abs(self.position),
                            'direction': signal['direction'],
                            'sl': stops['sl'],
                            'tp1': stops['tp1'],
                            'tp2': stops['tp2'],
                            'vix_at_entry': vix_val,
                            'rsi_at_entry': signal['rsi'],
                            'strength': signal['strength']
                        }
                        
                        logger.info(f"{current_time} | ENTRY {['SHORT','LONG'][signal['direction']>0]} @ {current_price:.2f} | Qty: {abs(self.position)} | VIX: {vix_val:.1f}")
                        
            # Manage existing position
            else:
                signal = self.check_confluence_signal(idx)
                
                # Check exits: SL, TP, signal reversal, EOD
                should_exit = False
                exit_reason = ""
                
                if self.position > 0:
                    if current_price <= float(self.data_1h['ATR'].iloc[idx - 1]) * 1.5 + float(self.position_price) - float(self.data_1h['ATR'].iloc[idx - 1]) * 1.5:
                        should_exit = True
                        exit_reason = "StopLoss"
                    elif signal and signal.get('direction') == -1:
                        should_exit = True
                        exit_reason = "Signal_Reversal"
                
                elif self.position < 0:
                    if current_price >= float(self.position_price) + float(self.data_1h['ATR'].iloc[idx - 1]) * 1.5:
                        should_exit = True
                        exit_reason = "StopLoss"
                    elif signal and signal.get('direction') == 1:
                        should_exit = True
                        exit_reason = "Signal_Reversal"
                
                if should_exit:
                    pnl = (current_price - self.position_price) * self.position
                    pnl_pct = (pnl / (abs(self.position) * self.position_price)) * 100
                    
                    self.cash += pnl
                    self.trades.append({
                        'entry_date': self.trades[-1]['entry_date'] if self.trades else current_time,
                        'exit_date': current_time,
                        'entry_price': self.position_price,
                        'exit_price': current_price,
                        'quantity': abs(self.position),
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'direction': self.position / abs(self.position),
                        'exit_reason': exit_reason
                    })
                    
                    logger.info(f"{current_time} | EXIT {exit_reason} @ {current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
                    
                    self.position = 0
        
        # Close any remaining position
        if self.position != 0:
            current_price = float(self.data_1h['Close'].iloc[-1])
            pnl = (current_price - self.position_price) * self.position
            self.cash += pnl
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate backtest performance metrics"""
        if len(self.trades) == 0:
            return {
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        total_return = self.cash - self.equity
        total_return_pct = (total_return / self.equity) * 100
        
        winning = len(trades_df[trades_df['pnl'] > 0])
        total = len(trades_df)
        win_rate = (winning / total * 100) if total > 0 else 0
        
        # Simple Sharpe (daily returns based on trade frequency)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'asset': self.asset,
            'total_return_pct': total_return_pct,
            'total_trades': total,
            'winning_trades': winning,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'avg_win_pct': trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() if winning > 0 else 0,
            'avg_loss_pct': trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if (total - winning) > 0 else 0,
            'max_profit': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'max_loss': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
            'equity_final': self.cash,
            'trades': self.trades  # Include for detailed analysis
        }


def main():
    """Run intraday backtest"""
    
    # Test on SPY (goal: beat 87.67%)
    strategy = VIXAdaptiveIntraday(asset='SPY', start_date='2023-01-01', end_date='2025-12-31')
    
    if strategy.fetch_data():
        results = strategy.run_backtest()
        
        print("\n" + "="*60)
        print(f"INTRADAY STRATEGY RESULTS - {strategy.asset}")
        print("="*60)
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Avg Win: {results['avg_win_pct']:.2f}% | Avg Loss: {results['avg_loss_pct']:.2f}%")
        print(f"Best Trade: ${results['max_profit']:.2f} | Worst Trade: ${results['max_loss']:.2f}")
        print(f"Final Equity: ${results['equity_final']:.2f}")
        print("="*60)
        
        # Save results
        with open('../results/intraday_spy_v1.json', 'w') as f:
            results_to_save = {k: v for k, v in results.items() if k != 'trades'}
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n✅ Results saved to ../results/intraday_spy_v1.json")
    else:
        logger.error("Failed to fetch data")


if __name__ == '__main__':
    main()
