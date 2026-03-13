"""
Intraday-Style High-Frequency Daily Trader v2
=============================================

Simplified but powerful approach:
- Uses daily bars with intraday signal logic
- Multiple entry/exit opportunities per day  
- VIX-based risk management
- Goal: Beat SPY 87.67% baseline

Robust implementation that handles yfinance data properly.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HighFreqDailyTrader:
    """Intraday-style signals on daily bars"""
    
    def __init__(self, ticker='SPY'):
        self.ticker = ticker
        self.data = None
        self.vix = None
        self.equity = 100000
        self.positions = []
        self.closed_trades = []
        self.equity_curve = [100000]
        
    def fetch_data(self):
        """Fetch data cleanly"""
        try:
            logger.info(f"Fetching {self.ticker}...")
            self.data = yf.download(self.ticker, start='2023-01-01', end='2025-12-31', progress=False)
            
            # Ensure single index
            if isinstance(self.data.index, pd.MultiIndex):
                self.data = self.data.reset_index(level=0, drop=True)
            
            # Ensure columns aren't MultiIndex
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)
            
            self.vix = yf.download('^VIX', start='2023-01-01', end='2025-12-31', progress=False)
            if isinstance(self.vix.columns, pd.MultiIndex):
                self.vix.columns = self.vix.columns.get_level_values(0)
            
            logger.info(f"Loaded {len(self.data)} bars")
            return True
        except Exception as e:
            logger.error(f"Data error: {e}")
            return False
    
    def prepare_indicators(self):
        """Prepare all indicators"""
        df = self.data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['Close'].ewm(12).mean()
        ema26 = df['Close'].ewm(26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(9).mean()
        
        # SMAs
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        # ATR
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # Volume
        df['Vol_Avg'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_Avg']
        
        # VIX
        df['VIX'] = self.vix['Close']
        df['VIX'] = df['VIX'].fillna(method='ffill').fillna(20)
        
        self.data = df
        return df
    
    def get_signals(self, idx):
        """Generate buy/sell signals"""
        if idx < 50:
            return []
        
        row = self.data.iloc[idx]
        signals = []
        
        # Bullish signals
        # 1. RSI oversold reversal
        if self.data.iloc[idx-1]['RSI'] < 35 < row['RSI'] < 50:
            signals.append(('BUY', 'RSI_Reversal', 0.7))
        
        # 2. MACD bullish crossover
        if self.data.iloc[idx-1]['MACD'] < self.data.iloc[idx-1]['Signal'] and row['MACD'] > row['Signal']:
            signals.append(('BUY', 'MACD_Cross', 0.8))
        
        # 3. Price above SMA20 with volume
        if row['Close'] > row['SMA20'] and self.data.iloc[idx-1]['Close'] <= self.data.iloc[idx-1]['SMA20']:
            if row['Vol_Ratio'] > 1.2:
                signals.append(('BUY', 'MA_Breakout', 0.75))
        
        # Bearish signals
        # 1. RSI overbought reversal
        if self.data.iloc[idx-1]['RSI'] > 65 > row['RSI'] > 50:
            signals.append(('SELL', 'RSI_Reversal', 0.7))
        
        # 2. MACD bearish crossover
        if self.data.iloc[idx-1]['MACD'] > self.data.iloc[idx-1]['Signal'] and row['MACD'] < row['Signal']:
            signals.append(('SELL', 'MACD_Cross', 0.8))
        
        # 3. Price below SMA20 with volume
        if row['Close'] < row['SMA20'] and self.data.iloc[idx-1]['Close'] >= self.data.iloc[idx-1]['SMA20']:
            if row['Vol_Ratio'] > 1.2:
                signals.append(('SELL', 'MA_Breakdown', 0.75))
        
        return signals
    
    def vix_position_size(self, base_risk, vix):
        """VIX-adjusted sizing"""
        if vix < 15:
            return base_risk * 1.3
        elif vix < 20:
            return base_risk * 1.0
        elif vix < 30:
            return base_risk * 0.7
        else:
            return base_risk * 0.4
    
    def backtest(self):
        """Run backtest"""
        logger.info("Starting backtest...")
        self.prepare_indicators()
        
        for idx in range(50, len(self.data)):
            date = self.data.index[idx]
            price = float(self.data['Close'].iloc[idx])
            vix = float(self.data['VIX'].iloc[idx])
            atr = float(self.data['ATR'].iloc[idx])
            
            # Manage open positions
            for pos in list(self.positions):
                if pos['dir'] == 1:  # Long
                    if price <= pos['sl'] or price >= pos['tp']:
                        exit_p = min(price, pos['sl']) if price <= pos['sl'] else pos['tp']
                        pnl = (exit_p - pos['entry']) * pos['qty']
                        self.equity += pnl
                        self.closed_trades.append({
                            'symbol': self.ticker,
                            'entry_date': pos['date'],
                            'exit_date': date,
                            'entry_price': pos['entry'],
                            'exit_price': exit_p,
                            'qty': pos['qty'],
                            'side': 'LONG',
                            'pnl': pnl,
                            'pnl_pct': (pnl / (pos['entry'] * pos['qty'])) * 100,
                            'reason': pos['reason']
                        })
                        self.positions.remove(pos)
                else:  # Short
                    if price >= pos['sl'] or price <= pos['tp']:
                        exit_p = max(price, pos['sl']) if price >= pos['sl'] else pos['tp']
                        pnl = (pos['entry'] - exit_p) * pos['qty']
                        self.equity += pnl
                        self.closed_trades.append({
                            'symbol': self.ticker,
                            'entry_date': pos['date'],
                            'exit_date': date,
                            'entry_price': pos['entry'],
                            'exit_price': exit_p,
                            'qty': pos['qty'],
                            'side': 'SHORT',
                            'pnl': pnl,
                            'pnl_pct': (pnl / (pos['entry'] * pos['qty'])) * 100,
                            'reason': pos['reason']
                        })
                        self.positions.remove(pos)
            
            # Generate new signals (max 2 concurrent)
            if len(self.positions) < 2:
                signals = self.get_signals(idx)
                for sig_dir, sig_reason, sig_strength in signals:
                    
                    # Position sizing with VIX
                    base_risk = self.equity * 0.02
                    risk_amount = self.vix_position_size(base_risk, vix)
                    
                    if sig_dir == 'BUY':
                        sl = price - (atr * 1.5)
                        tp = price + (atr * 2.5)
                        qty = int(risk_amount / (price - sl))
                    else:  # SELL
                        sl = price + (atr * 1.5)
                        tp = price - (atr * 2.5)
                        qty = int(risk_amount / (sl - price))
                    
                    if qty > 0:
                        self.positions.append({
                            'date': date,
                            'entry': price,
                            'qty': qty,
                            'sl': sl,
                            'tp': tp,
                            'dir': 1 if sig_dir == 'BUY' else -1,
                            'reason': sig_reason,
                            'vix': vix
                        })
                        logger.info(f"{date.date()} | {sig_dir} @ {price:.2f} | {sig_reason} | VIX:{vix:.1f}")
            
            self.equity_curve.append(self.equity)
        
        # Close remaining
        for pos in list(self.positions):
            price = float(self.data['Close'].iloc[-1])
            if pos['dir'] == 1:
                pnl = (price - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - price) * pos['qty']
            
            self.equity += pnl
            self.closed_trades.append({
                'symbol': self.ticker,
                'entry_date': pos['date'],
                'exit_date': self.data.index[-1],
                'entry_price': pos['entry'],
                'exit_price': price,
                'qty': pos['qty'],
                'side': 'LONG' if pos['dir'] == 1 else 'SHORT',
                'pnl': pnl,
                'pnl_pct': (pnl / (pos['entry'] * pos['qty'])) * 100,
                'reason': 'EOB_Close'
            })
            self.positions = []
        
        return self.metrics()
    
    def metrics(self):
        """Calculate metrics"""
        if len(self.closed_trades) == 0:
            return {'error': 'No trades'}
        
        trades = pd.DataFrame(self.closed_trades)
        ret_pct = ((self.equity - 100000) / 100000) * 100
        wins = len(trades[trades['pnl'] > 0])
        total = len(trades)
        
        if total > 1:
            pnl_pcts = trades['pnl_pct'].values
            sharpe = np.mean(pnl_pcts) / (np.std(pnl_pcts) + 0.001)
        else:
            sharpe = 0
        
        # Drawdown
        peak = 100000
        max_dd = 0
        for e in self.equity_curve:
            if e > peak:
                peak = e
            dd = ((peak - e) / peak) * 100
            max_dd = max(max_dd, dd)
        
        return {
            'symbol': self.ticker,
            'return_pct': ret_pct,
            'initial': 100000,
            'final': self.equity,
            'trades': total,
            'wins': wins,
            'win_rate': (wins / total) * 100 if total > 0 else 0,
            'sharpe': sharpe,
            'avg_win': trades[trades['pnl'] > 0]['pnl_pct'].mean() if wins > 0 else 0,
            'avg_loss': trades[trades['pnl'] < 0]['pnl_pct'].mean() if (total - wins) > 0 else 0,
            'max_profit': trades['pnl'].max(),
            'max_loss': trades['pnl'].min(),
            'max_dd': max_dd,
            'profit_factor': (abs(trades[trades['pnl'] > 0]['pnl'].sum()) / abs(trades[trades['pnl'] < 0]['pnl'].sum() + 1))
        }


# Main
logger.info("=" * 70)
logger.info("INTRADAY-STYLE TRADER V2 - SPY")
logger.info("=" * 70)

trader = HighFreqDailyTrader('SPY')
if trader.fetch_data():
    results = trader.backtest()
    
    print("\n" + "="*70)
    print(f"RESULTS: {results['symbol']}")
    print("="*70)
    print(f"SPY Benchmark: 87.67% over 3 years")
    print(f"Strategy Return: {results['return_pct']:.2f}%")
    print(f"CAGR: {(((results['final']/results['initial'])**(1/3))-1)*100:.2f}%")
    print(f"Trades: {results['trades']} | Wins: {results['wins']} | WR: {results['win_rate']:.1f}%")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"Max DD: {results['max_dd']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Avg Win: {results['avg_win']:.2f}% | Avg Loss: {results['avg_loss']:.2f}%")
    print("="*70)
    
    # Save
    with open('intraday/results/high_freq_daily_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Saved to intraday/results/high_freq_daily_v2.json")
