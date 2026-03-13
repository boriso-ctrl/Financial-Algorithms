"""
Smart Daily Trader with Intraday-Style Signals
===============================================

Uses daily bars with advanced entry/exit logic to simulate intraday performance:
- Multiple signals per daily bar (open, mid-day, EOD)
- VIX-adaptive position sizing & stops
- Session bias (opening gaps, end-of-day)
- Reversal + trend-following fusion

Goal: Outperform S&P 500 (87.67% return over 3 years = 23.35% CAGR)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartDailyTrader:
    """High-frequency daily signals with VIX adaptation for outperformance"""
    
    def __init__(self, asset='SPY', start_date='2023-01-01', end_date='2025-12-31'):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.equity_curve = [100000]
        self.current_equity = 100000
        self.positions = []  # Active positions
        self.closed_trades = []
        
        # Configuration
        self.per_trade_risk = 0.02  # Risk 2% per trade
        self.max_concurrent = 3  # Max 3 concurrent positions
        
        logger.info(f"Initialized Smart Daily Trader for {asset}")
    
    def fetch_data(self):
        """Fetch daily data and VIX"""
        logger.info(f"Fetching daily data for {self.asset}...")
        
        try:
            self.data = yf.download(self.asset, start=self.start_date, end=self.end_date, progress=False)
            self.vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
            
            logger.info(f"Loaded {len(self.data)} daily bars for {self.asset}")
            return True
            
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return False
    
    def calculate_indicators(self):
        """Calculate all indicators for decision making"""
        df = self.data.copy()
        
        # === MOMENTUM INDICATORS ===
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Stochastic (14, 3, 3)
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        df['K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        df['D'] = df['K'].rolling(window=3).mean()
        
        # === TREND INDICATORS ===
        # ATR (14) - volatility
        tr = np.maximum(df['High'] - df['Low'], 
                       np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                abs(df['Low'] - df['Close'].shift(1))))
        atr_series = tr.rolling(window=14).mean()
        df['ATR'] = atr_series.values
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # Moving averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        
        # Trend strength via ADX
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        atr = df['ATR'].values
        atr_safe = np.clip(atr, 0.0001, None)  # Avoid division by zero
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean().values / atr_safe)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean().values / atr_safe)
        denominator = plus_di + minus_di
        denominator = np.clip(denominator, 0.0001, None)
        dx = 100 * abs(plus_di - minus_di) / denominator
        df['ADX'] = pd.Series(dx).rolling(window=14).mean().values
        
        # === VOLUME & BREADTH ===
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=14).mean()
        df['OBV_Signal'] = np.where(df['OBV'] > df['OBV_MA'], 1, -1)
        
        # === GAP DETECTION ===
        df['Gap'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
        df['Gap_Direction'] = np.sign(df['Gap'])
        
        # === VOLATILITY (VIX integration) ===
        df['VIX'] = np.nan
        for date in df.index:
            if date in self.vix_data.index:
                df.loc[date, 'VIX'] = self.vix_data.loc[date, 'Close']
        df['VIX'] = df['VIX'].fillna(method='ffill').fillna(20)
        
        # === PRICE ACTION ===
        df['High_Low_Ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])  # Close position in range
        df['Body_Ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])  # Body size
        
        self.data = df
        return df
    
    def generate_signals(self, idx):
        """Generate buy/sell signals using multiple timeframe analysis"""
        if idx < 50:
            return []
        
        row = self.data.iloc[idx]
        signals = []
        
        # === BULLISH SIGNALS ===
        
        # Signal 1: Opening gap up + bullish follow-through
        if idx > 0:
            gap = float(row['Gap'])
            if gap > 0.5:  # 0.5% gap up
                # Check for bullish confirmation
                if (float(row['Close']) > float(row['Open']) and 
                    float(row['Volume']) > float(row['Volume_MA']) * 1.2):
                    signals.append({
                        'type': 'buy',
                        'reason': 'Gap_Up_Breakout',
                        'strength': min(gap / 2, 1.0),  # Strength proportional to gap
                        'vix': float(row['VIX'])
                    })
            
            # Signal 2: RSI reversal from oversold
            prev_rsi = float(self.data['RSI'].iloc[idx - 1])
            if (prev_rsi < 35 and float(row['RSI']) > prev_rsi and float(row['RSI']) < 50):
                signals.append({
                    'type': 'buy',
                    'reason': 'RSI_Oversold_Reversal',
                    'strength': (50 - float(row['RSI'])) / 50,
                    'vix': float(row['VIX'])
                })
            
            # Signal 3: EMA9 crosses above EMA21 on volume
            if (float(self.data['EMA9'].iloc[idx - 1]) <= float(self.data['EMA21'].iloc[idx - 1]) and
                float(row['EMA9']) > float(row['EMA21']) and
                float(row['Volume_Ratio']) > 1.1):
                signals.append({
                    'type': 'buy',
                    'reason': 'EMA_Golden_Cross',
                    'strength': 0.8,
                    'vix': float(row['VIX'])
                })
            
            # Signal 4: Price breaks above resistance (SMA20) with strong ADX
            if (float(self.data['Close'].iloc[idx - 1]) <= float(row['SMA20']) and
                float(row['Close']) > float(row['SMA20']) and
                float(row['ADX']) > 30):
                signals.append({
                    'type': 'buy',
                    'reason': 'Resistance_Breakout',
                    'strength': 0.7,
                    'vix': float(row['VIX'])
                })
            
            # Signal 5: MACD histogram turns positive
            prev_hist = float(self.data['MACD_Histogram'].iloc[idx - 1])
            if (prev_hist < 0 and float(row['MACD_Histogram']) > 0):
                signals.append({
                    'type': 'buy',
                    'reason': 'MACD_Histogram_Bullish',
                    'strength': 0.6,
                    'vix': float(row['VIX'])
                })
        
        # === BEARISH SIGNALS ===
        
        if idx > 0:
            # Signal 1: Opening gap down + bearish follow-through
            gap = float(row['Gap'])
            if gap < -0.5:  # 0.5% gap down
                if (float(row['Close']) < float(row['Open']) and 
                    float(row['Volume']) > float(row['Volume_MA']) * 1.2):
                    signals.append({
                        'type': 'sell',
                        'reason': 'Gap_Down_Breakdown',
                        'strength': min(abs(gap) / 2, 1.0),
                        'vix': float(row['VIX'])
                    })
            
            # Signal 2: RSI reversal from overbought
            prev_rsi = float(self.data['RSI'].iloc[idx - 1])
            if (prev_rsi > 65 and float(row['RSI']) < prev_rsi and float(row['RSI']) > 50):
                signals.append({
                    'type': 'sell',
                    'reason': 'RSI_Overbought_Reversal',
                    'strength': (float(row['RSI']) - 50) / 50,
                    'vix': float(row['VIX'])
                })
            
            # Signal 3: EMA9 crosses below EMA21 on volume
            if (float(self.data['EMA9'].iloc[idx - 1]) >= float(self.data['EMA21'].iloc[idx - 1]) and
                float(row['EMA9']) < float(row['EMA21']) and
                float(row['Volume_Ratio']) > 1.1):
                signals.append({
                    'type': 'sell',
                    'reason': 'EMA_Death_Cross',
                    'strength': 0.8,
                    'vix': float(row['VIX'])
                })
            
            # Signal 4: Price breaks below support (SMA20) with strong ADX
            if (float(self.data['Close'].iloc[idx - 1]) >= float(row['SMA20']) and
                float(row['Close']) < float(row['SMA20']) and
                float(row['ADX']) > 30):
                signals.append({
                    'type': 'sell',
                    'reason': 'Support_Breakdown',
                    'strength': 0.7,
                    'vix': float(row['VIX'])
                })
        
        return signals
    
    def execute_trade(self, signal, price, date):
        """Execute trade based on signal"""
        vix = signal['vix']
        
        # VIX-based position sizing
        if vix < 15:
            size_mult = 1.2
        elif vix < 20:
            size_mult = 1.0
        elif vix < 30:
            size_mult = 0.7
        else:
            size_mult = 0.4
        
        # Risk calculation
        position_risk = self.current_equity * self.per_trade_risk * size_mult
        
        # Dynamic stops based on ATR
        atr_val = float(self.data['ATR'].loc[date]) if pd.notna(self.data['ATR'].loc[date]) else 0
        
        if signal['type'] == 'buy':
            stop_loss = price - (atr_val * 1.5)
            take_profit = price + (atr_val * 2.5)
            qty = int(position_risk / (price - stop_loss))
        else:  # sell
            stop_loss = price + (atr_val * 1.5)
            take_profit = price - (atr_val * 2.5)
            qty = int(position_risk / (stop_loss - price))
        
        if qty > 0:
            self.positions.append({
                'date': date,
                'signal_type': signal['type'],
                'entry_price': price,
                'qty': qty,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': signal['reason'],
                'vix': vix,
                'strength': signal['strength'],
                'atr': atr_val
            })
            
            return True
        return False
    
    def manage_positions(self, current_price, current_date):
        """Check position exits (SL, TP, reversal)"""
        closed = []
        
        for pos in list(self.positions):
            pnl = 0
            exit_reason = ""
            
            if pos['signal_type'] == 'buy':
                if current_price <= pos['stop_loss']:
                    pnl = (pos['stop_loss'] - pos['entry_price']) * pos['qty']
                    exit_reason = "StopLoss"
                elif current_price >= pos['take_profit']:
                    pnl = (pos['take_profit'] - pos['entry_price']) * pos['qty']
                    exit_reason = "TakeProfit"
            else:  # short
                if current_price >= pos['stop_loss']:
                    pnl = (pos['entry_price'] - pos['stop_loss']) * pos['qty']
                    exit_reason = "StopLoss"
                elif current_price <= pos['take_profit']:
                    pnl = (pos['entry_price'] - pos['take_profit']) * pos['qty']
                    exit_reason = "TakeProfit"
            
            if exit_reason:
                self.current_equity += pnl
                self.equity_curve.append(self.current_equity)
                
                self.closed_trades.append({
                    'entry_date': pos['date'],
                    'exit_date': current_date,
                    'signal_type': pos['signal_type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price if exit_reason != "TakeProfit" else pos['take_profit'] if pos['signal_type'] == 'buy' else pos['take_profit'],
                    'qty': pos['qty'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (pos['entry_price'] * pos['qty'])) * 100,
                    'exit_reason': exit_reason,
                    'reason': pos['reason']
                })
                
                logger.info(f"{current_date} | {exit_reason} {pos['signal_type'].upper()} @ {current_price:.2f} | PnL: ${pnl:.2f}")
                
                closed.append(pos)
        
        # Remove closed positions
        for pos in closed:
            self.positions.remove(pos)
    
    def run_backtest(self):
        """Run complete backtest"""
        logger.info("Starting backtest...")
        
        self.calculate_indicators()
        
        for idx in range(50, len(self.data)):
            date = self.data.index[idx]
            price = float(self.data['Close'].iloc[idx])
            
            # Manage existing positions
            self.manage_positions(price, date)
            
            # Generate new signals (only if room)
            if len(self.positions) < self.max_concurrent:
                signals = self.generate_signals(idx)
                
                for signal in signals:
                    if self.execute_trade(signal, price, date):
                        logger.info(f"{date} | ENTRY {signal['type'].upper()} @ {price:.2f} | {signal['reason']}")
        
        #Close remaining positions
        if self.positions:
            final_price = float(self.data['Close'].iloc[-1])
            for pos in list(self.positions):
                if pos['signal_type'] == 'buy':
                    pnl = (final_price - pos['entry_price']) * pos['qty']
                else:
                    pnl = (pos['entry_price'] - final_price) * pos['qty']
                
                self.current_equity += pnl
                self.closed_trades.append({
                    'entry_date': pos['date'],
                    'exit_date': self.data.index[-1],
                    'signal_type': pos['signal_type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'qty': pos['qty'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (pos['entry_price'] * pos['qty'])) * 100,
                    'exit_reason': 'EOB_Liquidation',
                    'reason': pos['reason']
                })
                self.positions = []
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        trades_df = pd.DataFrame(self.closed_trades)
        
        if len(trades_df) == 0:
            return {
                'asset': self.asset,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0
            }
        
        total_return = self.current_equity - 100000
        total_return_pct = (total_return / 100000) * 100
        
        winning = len(trades_df[trades_df['pnl'] > 0])
        total = len(trades_df)
        win_rate = (winning / total * 100) if total > 0 else 0
        
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe = np.mean(returns) / (np.std(returns) + 0.0001) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        equity_peak = 100000
        max_dd = 0
        for e in self.equity_curve:
            if e > equity_peak:
                equity_peak = e
            dd = ((equity_peak - e) / equity_peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return {
            'asset': self.asset,
            'total_return_pct': total_return_pct,
            'initial_equity': 100000,
            'final_equity': self.current_equity,
            'total_trades': total,
            'winning_trades': winning,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'avg_win_pct': trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() if winning > 0 else 0,
            'avg_loss_pct': trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if (total - winning) > 0 else 0,
            'max_profit': trades_df['pnl'].max(),
            'max_loss': trades_df['pnl'].min(),
            'max_drawdown_pct': max_dd,
            'avg_trade_duration_days': (trades_df['exit_date'] - trades_df['entry_date']).dt.total_seconds().mean() / 86400 if len(trades_df) > 0 else 0
        }


def main():
    """Run backtest"""
    strategy = SmartDailyTrader(asset='SPY', start_date='2023-01-01', end_date='2025-12-31')
    
    if strategy.fetch_data():
        results = strategy.run_backtest()
        
        print("\n" + "="*70)
        print(f"SMART DAILY TRADER - {strategy.asset}")
        print("="*70)
        print(f"Period: 2023-01-01 to 2025-12-31 (3 years)")
        print(f"SPY Benchmark: 87.67% | Strategy Goal: BEAT BENCHMARK")
        print("="*70)
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Annualized CAGR: {(((results['final_equity']/results['initial_equity'])**(1/3))-1)*100:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Avg Win: {results['avg_win_pct']:.2f}% | Avg Loss: {results['avg_loss_pct']:.2f}%")
        print(f"Profit Factor: {abs(results['avg_win_pct'] * results['winning_trades']) / (abs(results['avg_loss_pct'] * (results['total_trades'] - results['winning_trades'])) + 0.001):.2f}")
        print("="*70)
        
        # Save results
        with open('../../intraday/results/smart_daily_trader_v1.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved")
    else:
        logger.error("Failed to fetch data")


if __name__ == '__main__':
    main()
