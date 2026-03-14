#!/usr/bin/env python3
"""
AGGRESSIVE HYBRID TRADER V5 - MULTI-ASSET
=========================================
Applies V4 strategy logic to multiple assets:
- SPY (S&P 500, benchmark)
- QQQ (Nasdaq 100, more volatile/trendy)
- AAPL (largest tech stock)
- IWM (Russell 2000, small-cap)

Goal: Find optimal asset(s) for aggressive hybrid approach
Strategy: Same as V4 (1+ confluence, trend + mean-reversion)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class AggressiveHybridV5:
    def __init__(self, symbols, start_date='2023-01-01', end_date='2025-12-31'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}
        
    def download_data(self, symbol):
        """Download daily price data for asset"""
        try:
            df = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            # Flatten multi-index columns if downloading multiple symbols
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            logger.info(f"{symbol}: Downloaded {len(df)} bars")
            return df
        except Exception as e:
            logger.error(f"{symbol}: Download failed - {e}")
            return None
    
    def prepare_indicators(self, df):
        """Calculate all technical indicators"""
        # Handle column naming (after reset_index, column names might differ)
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})
        
        # Ensure we have the required columns
        required = ['High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing column: {col}. Available: {list(df.columns)}")
                raise KeyError(f"Column {col} not found")
        
        # EMAs
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # ADX
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = df['High'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm = -df['Low'].diff()
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        adx_raw = 100 * (di_diff / di_sum).rolling(14).mean()
        df['ADX'] = adx_raw.rolling(5).mean()
        df['ATR'] = atr
        
        # Volume
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        
        # Price levels
        df['Price_Low_5'] = df['Close'].rolling(5).min()
        df['Price_High_5'] = df['Close'].rolling(5).max()
        
        return df
    
    def generate_buy_signals(self, df):
        """Generate buy signal count for this bar"""
        signals = 0
        signal_types = []
        
        if pd.isna(df['MACD']) or pd.isna(df['RSI']):
            return signals, signal_types
        
        # MACD positive
        if df['MACD'] > df['MACD_Signal']:
            signals += 1
            signal_types.append('macd_positive')
        
        # EMA9 > EMA20
        if df['EMA9'] > df['EMA20']:
            signals += 1
            signal_types.append('ema9_above_20')
        
        # All EMAs aligned (9 > 20 > 50 > 200)
        if df['EMA9'] > df['EMA20'] > df['EMA50'] > df['EMA200']:
            signals += 1
            signal_types.append('all_emas_aligned')
        
        # Strong trend (ADX > 25 + EMA200 up)
        if df['ADX'] > 25 and df['Close'] > df['EMA200']:
            signals += 1
            signal_types.append('strong_trend')
        
        # RSI oversold bounce
        if df['RSI'] < 40:
            signals += 1
            signal_types.append('rsi_oversold')
        
        # Price low bounce (close near 5-bar low)
        if df['Close'] < df['Price_Low_5'] * 1.01:
            signals += 1
            signal_types.append('price_low_bounce')
        
        # MACD histogram positive
        if df['MACD_Hist'] > 0:
            signals += 1
            signal_types.append('macd_hist_positive')
        
        # Volume surge
        if df['Volume'] > df['Volume_MA'] * 1.5:
            signals += 1
            signal_types.append('volume_surge')
        
        # MA breakout (close > all recent MAs)
        if df['Close'] > df['EMA50'] and df['EMA50'] > df['EMA200']:
            signals += 1
            signal_types.append('ma_breakout')
        
        return signals, signal_types
    
    def generate_sell_signals(self, df):
        """Generate sell signal count for this bar"""
        signals = 0
        signal_types = []
        
        if pd.isna(df['MACD']) or pd.isna(df['RSI']):
            return signals, signal_types
        
        # MACD negative
        if df['MACD'] < df['MACD_Signal']:
            signals += 1
            signal_types.append('macd_negative')
        
        # EMA9 < EMA20
        if df['EMA9'] < df['EMA20']:
            signals += 1
            signal_types.append('ema9_below_20')
        
        # Strong downtrend (ADX > 25 + EMA200 down)
        if df['ADX'] > 25 and df['Close'] < df['EMA200']:
            signals += 1
            signal_types.append('strong_trend')
        
        # RSI overbought
        if df['RSI'] > 60:
            signals += 1
            signal_types.append('rsi_overbought')
        
        # Price high rollover
        if df['Close'] > df['Price_High_5'] * 0.99:
            signals += 1
            signal_types.append('price_high_rollover')
        
        # MACD histogram negative
        if df['MACD_Hist'] < 0:
            signals += 1
            signal_types.append('macd_hist_negative')
        
        # Volume surge (bearish context)
        if df['Volume'] > df['Volume_MA'] * 1.5 and df['Close'] < df['EMA20']:
            signals += 1
            signal_types.append('volume_surge')
        
        # MA breakdown
        if df['Close'] < df['EMA50'] and df['EMA50'] < df['EMA200']:
            signals += 1
            signal_types.append('ma_breakdown')
        
        return signals, signal_types
    
    def backtest(self, symbol):
        """Run backtest for single symbol"""
        logger.info(f"\n{'='*70}")
        logger.info(f"AGGRESSIVE HYBRID TRADER V5 - {symbol}")
        logger.info(f"{'='*70}")
        
        df = self.download_data(symbol)
        if df is None or len(df) < 100:
            logger.error(f"{symbol}: Insufficient data")
            return None
        
        # Debug: print column names before reset
        logger.info(f"Columns before reset: {list(df.columns)}")
        
        df = df.reset_index()
        logger.info(f"Columns after reset: {list(df.columns)}")
        df = self.prepare_indicators(df)
        
        # Backtest simulation
        positions = []  # Track open positions
        closed_trades = []  # Track completed trades
        capital = 100000
        equity = [capital]
        equity_dates = []
        risk_per_trade = 0.015  # 1.5% risk
        max_positions = 5
        
        for idx in range(200, len(df)):  # Start after indicators are ready
            row = df.iloc[idx]
            date = row['Date']
            close = row['Close']
            atr = row['ATR']
            
            date_str = date.strftime('%Y-%m-%d')
            
            # Close positions with SL/TP hits or time decay
            for pos in positions[:]:
                pnl = (close - pos['entry']) * pos['size']
                pnl_pct = pnl / capital if capital > 0 else 0
                
                # TP hit (ATR * 3.5)
                if pos['direction'] == 'LONG' and close >= pos['tp']:
                    closed_trades.append({
                        'entry': pos['entry'], 'exit': close, 'size': pos['size'],
                        'pnl': pnl, 'direction': 'LONG', 'exit_date': date_str
                    })
                    capital += pnl
                    positions.remove(pos)
                    logger.info(f"{date_str} | CLOSE LONG ${pnl:,.0f} (TP) @ {close:.2f}")
                
                # SL hit (ATR * 2.0)
                elif pos['direction'] == 'LONG' and close <= pos['sl']:
                    closed_trades.append({
                        'entry': pos['entry'], 'exit': close, 'size': pos['size'],
                        'pnl': pnl, 'direction': 'LONG', 'exit_date': date_str
                    })
                    capital += pnl
                    positions.remove(pos)
                    logger.info(f"{date_str} | CLOSE LONG ${pnl:,.0f} (SL) @ {close:.2f}")
                
                # Time decay (90 days)
                elif (date - pos['entry_date']).days >= 90:
                    closed_trades.append({
                        'entry': pos['entry'], 'exit': close, 'size': pos['size'],
                        'pnl': pnl, 'direction': 'LONG', 'exit_date': date_str
                    })
                    capital += pnl
                    positions.remove(pos)
                    logger.info(f"{date_str} | CLOSE LONG ${pnl:,.0f} (90d) @ {close:.2f}")
            
            # Check for entry signals
            buy_signals, buy_types = self.generate_buy_signals(row)
            sell_signals, sell_types = self.generate_sell_signals(row)
            
            # Entry logic (need 1+ signal for V5)
            if buy_signals >= 1 and len(positions) < max_positions and capital > 0:
                risk_amount = capital * risk_per_trade
                vix_multiplier = min(1.2, max(0.8, 25.0 / max(row.get('RSI', 50), 10)))
                size = max(1, int(risk_amount / (atr * 2.0) * vix_multiplier))
                
                entry_price = close
                stop_loss = close - (atr * 2.0)
                take_profit = close + (atr * 3.5)
                position_risk = size * (entry_price - stop_loss)
                
                if position_risk < risk_amount * 1.5:
                    position = {
                        'entry': entry_price, 'entry_date': date, 'size': size,
                        'sl': stop_loss, 'tp': take_profit, 'direction': 'LONG'
                    }
                    positions.append(position)
                    logger.info(f"{date_str} | BUY x{size} @ {close:.2f} | {' + '.join(buy_types[:2])}")
            
            elif sell_signals >= 1 and len(positions) < max_positions and capital > 0:
                # Short logic (reduced frequency in bull market)
                if buy_signals < sell_signals and row['RSI'] > 70:
                    risk_amount = capital * risk_per_trade * 0.5  # Half size for shorts
                    size = max(1, int(risk_amount / (atr * 2.0)))
                    
                    entry_price = close
                    stop_loss = close + (atr * 2.0)
                    take_profit = close - (atr * 3.5)
                    
                    position = {
                        'entry': entry_price, 'entry_date': date, 'size': size,
                        'sl': stop_loss, 'tp': take_profit, 'direction': 'SHORT'
                    }
                    positions.append(position)
                    logger.info(f"{date_str} | SELL x{size} @ {close:.2f} | {' + '.join(sell_types[:2])}")
            
            # Track equity
            open_pnl = sum((close - pos['entry']) * pos['size'] for pos in positions)
            equity.append(capital + open_pnl)
            equity_dates.append(date)
        
        # Close all remaining positions
        final_close = df.iloc[-1]['Close']
        for pos in positions:
            pnl = (final_close - pos['entry']) * pos['size']
            capital += pnl
            closed_trades.append({
                'entry': pos['entry'], 'exit': final_close, 'size': pos['size'],
                'pnl': pnl, 'direction': pos['direction'], 'exit_date': df.iloc[-1]['Date'].strftime('%Y-%m-%d')
            })
        
        # Calculate metrics
        equity_array = np.array(equity)
        returns = equity_array[1:] / equity_array[:-1] - 1
        total_return = (equity[-1] - capital) / 100000
        cagr = (equity[-1] / 100000) ** (1 / 3) - 1 if len(equity_dates) > 0 else 0
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        max_dd = (np.min(equity_array) - capital) / capital if capital > 0 else 0
        
        wins = [t for t in closed_trades if t['pnl'] > 0]
        win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_hold = np.mean([(datetime.strptime(t['exit_date'], '%Y-%m-%d') - 
                            df[df['Close'] == t['entry']]['Date'].iloc[0]).days 
                           for t in closed_trades if len(df[df['Close'] == t['entry']]) > 0]) if closed_trades else 0
        
        best_trade = max(t['pnl'] for t in closed_trades) if closed_trades else 0
        worst_trade = min(t['pnl'] for t in closed_trades) if closed_trades else 0
        
        results = {
            'symbol': symbol,
            'return': total_return,
            'cagr': cagr,
            'trades': len(closed_trades),
            'wins': len(wins),
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'profit_factor': profit_factor,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_hold': avg_hold,
            'final_capital': equity[-1]
        }
        
        logger.info(f"Return: {total_return*100:.2f}% | CAGR: {cagr*100:.2f}%")
        logger.info(f"Trades: {len(closed_trades)} | Wins: {len(wins)} | WR: {win_rate*100:.1f}%")
        logger.info(f"Sharpe: {sharpe:.2f} | Max DD: {max_dd*100:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}x")
        logger.info(f"Best Trade: ${best_trade:,.0f} | Worst: ${worst_trade:,.0f}")
        logger.info(f"Avg Hold: {avg_hold:.1f} days")
        
        return results
    
    def run_all(self):
        """Run backtest for all symbols"""
        for symbol in self.symbols:
            result = self.backtest(symbol)
            if result:
                self.results[symbol] = result
        
        # Print comparison table
        print(f"\n{'='*90}")
        print("MULTI-ASSET PERFORMANCE COMPARISON (V5)".center(90))
        print(f"{'='*90}")
        print(f"{'Symbol':<10} {'Return':<12} {'CAGR':<10} {'Trades':<10} {'WR':<10} {'Sharpe':<10} {'Max DD':<10}")
        print(f"{'-'*90}")
        
        ranked = sorted(self.results.items(), key=lambda x: x[1]['return'], reverse=True)
        for symbol, res in ranked:
            print(f"{symbol:<10} {res['return']*100:>10.2f}% {res['cagr']*100:>9.2f}% {res['trades']:>9} {res['win_rate']*100:>9.1f}% {res['sharpe']:>9.2f} {res['max_dd']*100:>9.2f}%")
        
        print(f"{'-'*90}")
        print(f"\nSPY Benchmark: 87.67% (goal)")
        print(f"Best Strategy: {ranked[0][0]} with {ranked[0][1]['return']*100:.2f}% return")
        
        # Save results
        with open('intraday/results/aggressive_hybrid_v5_multiasset.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to intraday/results/aggressive_hybrid_v5_multiasset.json")
        
        return self.results

if __name__ == '__main__':
    trader = AggressiveHybridV5(
        symbols=['SPY', 'QQQ', 'AAPL', 'IWM'],
        start_date='2023-01-01',
        end_date='2025-12-31'
    )
    trader.run_all()
