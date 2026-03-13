"""
Aggressive Trend + Mean-Reversion Hybrid V4
============================================

Improvements over V3:
1. Looser confluence (1+ signals, not 2+)
2. Combines trend-following WITH mean-reversion trades
3. Smaller position sizes, up to 5 concurrent positions
4. Adds more signal types (breakouts, support/resistance)
5. Dynamic sizing based on trend strength (ADX) and volatility (VIX)

Goal: Increase trade frequency while maintaining win rate > 50%
Target: Beat SPY 87.67%
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggressiveHybridTrader:
    """Trend + Mean-Reversion with aggressive sizing"""
    
    def __init__(self, ticker='SPY'):
        self.ticker = ticker
        self.data = None
        self.vix = None
        self.equity = 100000
        self.positions = []
        self.closed_trades = []
        self.equity_curve = [100000]
        
    def fetch_data(self):
        """Fetch data"""
        try:
            self.data = yf.download(self.ticker, start='2023-01-01', end='2025-12-31', progress=False)
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)
            
            self.vix = yf.download('^VIX', start='2023-01-01', end='2025-12-31', progress=False)
            if isinstance(self.vix.columns, pd.MultiIndex):
                self.vix.columns = self.vix.columns.get_level_values(0)
            
            logger.info(f"Loaded {len(self.data)} bars")
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def prepare_indicators(self):
        """Prepare indicators"""
        df = self.data.copy()
        
        # ==== MAIN TREND ====
        df['EMA200'] = df['Close'].ewm(200).mean()
        df['EMA50'] = df['Close'].ewm(50).mean()
        df['EMA20'] = df['Close'].ewm(20).mean()
        df['EMA9'] = df['Close'].ewm(9).mean()
        
        df['LT_Trend'] = np.where(df['Close'] > df['EMA200'], 1, -1)
        df['MT_Trend'] = np.where(df['EMA20'] > df['EMA50'], 1, -1)
        
        # ==== MOMENTUM ====
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        ema12 = df['Close'].ewm(12).mean()
        ema26 = df['Close'].ewm(26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # ==== VOLATILITY & TREND STRENGTH ====
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        atr_safe = np.clip(atr, 0.0001, None)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean().values / atr_safe)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean().values / atr_safe)
        denom = plus_di + minus_di
        denom = np.clip(denom, 0.0001, None)
        dx = 100 * abs(plus_di - minus_di) / denom
        df['ADX'] = pd.Series(dx).rolling(14).mean().values
        
        df['ATR'] = atr.values
        
        # ==== VOLATILITY & VOLUME ====
        df['Vol_Avg'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_Avg']
        df['VIX'] = self.vix['Close']
        df['VIX'] = df['VIX'].fillna(method='ffill').fillna(20)
        
        # ==== SUPPORT/RESISTANCE ====
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Close_Range'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'] + 1e-10)
        
        # ==== PRICE ACTION ====
        df['Body_Size'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['Upper_Wick'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        
        self.data = df
    
    def generate_buy_signals(self, idx):
        """Generate all bullish signals"""
        if idx < 200:
            return []
        
        row = self.data.iloc[idx]
        prev = self.data.iloc[idx - 1]
        signals = []
        strength = 0
        
        # === TREND-FOLLOWING SIGNALS ===
        
        # 1. MACD positive
        if row['MACD'] > row['Signal'] > 0:
            signals.append('macd_positive')
            strength += 0.2
        
        # 2. EMA9 above EMA20
        if row['EMA9'] > row['EMA20']:
            signals.append('ema9_above_20')
            strength += 0.15
        
        # 3. Price above all EMAs
        if row['Close'] > row['EMA20'] > row['EMA50'] > row['EMA200']:
            signals.append('all_emas_aligned')
            strength += 0.25
        
        # 4. ADX > 25 (strong trend)
        if row['ADX'] > 25:
            signals.append('strong_trend')
            strength += 0.15
        
        # === MEAN-REVERSION SIGNALS ===
        
        # 5. RSI oversold reversal (20-45)
        if prev['RSI'] < 35 < row['RSI'] < 45:
            signals.append('rsi_oversold')
            strength += 0.3
        
        # 6. Price near 20-day low, bouncing
        if row['Close_Range'] < 0.2 and row['Close'] > prev['Close']:
            signals.append('price_low_bounce')
            strength += 0.25
        
        # 7. MACD histogram turns positive
        if prev['MACD_Hist'] < 0 < row['MACD_Hist']:
            signals.append('macd_hist_positive')
            strength += 0.2
        
        # 8. Volume surge
        if row['Vol_Ratio'] > 1.3:
            signals.append('volume_surge')
            strength += 0.1
        
        # 9. Price breaks above resistance
        if prev['Close'] < prev['EMA20'] and row['Close'] > row['EMA20']:
            signals.append('ma_breakout')
            strength += 0.25
        
        return signals, strength if len(signals) > 0 else 0
    
    def generate_sell_signals(self, idx):
        """Generate all bearish signals"""
        if idx < 200:
            return []
        
        row = self.data.iloc[idx]
        prev = self.data.iloc[idx - 1]
        signals = []
        strength = 0
        
        # === TREND-FOLLOWING SIGNALS ===
        
        # 1. MACD negative
        if row['MACD'] < row['Signal'] < 0:
            signals.append('macd_negative')
            strength += 0.2
        
        # 2. EMA9 below EMA20
        if row['EMA9'] < row['EMA20']:
            signals.append('ema9_below_20')
            strength += 0.15
        
        # 3. ADX > 25 (strong trend)
        if row['ADX'] > 25:
            signals.append('strong_trend')
            strength += 0.15
        
        # === MEAN-REVERSION SIGNALS ===
        
        # 4. RSI overbought reversal (55-80)
        if prev['RSI'] > 65 > row['RSI'] > 55:
            signals.append('rsi_overbought')
            strength += 0.3
        
        # 5. Price near 20-day high, rolling over
        if row['Close_Range'] > 0.8 and row['Close'] < prev['Close']:
            signals.append('price_high_rollover')
            strength += 0.25
        
        # 6. MACD histogram turns negative
        if prev['MACD_Hist'] > 0 > row['MACD_Hist']:
            signals.append('macd_hist_negative')
            strength += 0.2
        
        # 7. Volume surge
        if row['Vol_Ratio'] > 1.3:
            signals.append('volume_surge')
            strength += 0.1
        
        # 8. Price breaks below support
        if prev['Close'] > prev['EMA20'] and row['Close'] < row['EMA20']:
            signals.append('ma_breakdown')
            strength += 0.25
        
        return signals, strength if len(signals) > 0 else 0
    
    def backtest(self):
        """Run backtest"""
        logger.info("Starting aggressive hybrid backtest...")
        self.prepare_indicators()
        
        for idx in range(200, len(self.data)):
            date = self.data.index[idx]
            price = float(self.data['Close'].iloc[idx])
            vix = float(self.data['VIX'].iloc[idx])
            atr = float(self.data['ATR'].iloc[idx])
            lt_trend = float(self.data['LT_Trend'].iloc[idx])
            adx = float(self.data['ADX'].iloc[idx])
            
            # Manage open positions
            for pos in list(self.positions):
                exit_price = None
                exit_reason = ""
                
                if pos['side'] == 1:  # Long
                    if price <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = "SL"
                    elif price >= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = "TP"
                    elif (date - pos['date']).days > 90:  # Max 90 day hold
                        exit_price = price
                        exit_reason = "Max_Hold"
                
                else:  # Short
                    if price >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = "SL"
                    elif price <= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = "TP"
                    elif (date - pos['date']).days > 90:
                        exit_price = price
                        exit_reason = "Max_Hold"
                
                if exit_price is not None:
                    pnl = (exit_price - pos['entry']) * pos['qty'] if pos['side'] == 1 else (pos['entry'] - exit_price) * pos['qty']
                    self.equity += pnl
                    self.closed_trades.append({
                        'entry_date': pos['date'],
                        'exit_date': date,
                        'entry_price': pos['entry'],
                        'exit_price': exit_price,
                        'qty': pos['qty'],
                        'side': 'LONG' if pos['side'] == 1 else 'SHORT',
                        'pnl': pnl,
                        'pnl_pct': (pnl / (pos['entry'] * pos['qty'])) * 100,
                        'hold_days': (date - pos['date']).days,
                        'reason': pos['reason'],
                        'reason2': pos['reason2'],
                        'exit_reason': exit_reason
                    })
                    self.positions.remove(pos)
            
            # Generate new signals
            if len(self.positions) < 5:  # Allow more concurrent positions
                
                # LONG SIGNALS
                if lt_trend > 0:  # Only in uptrend context
                    buy_sigs, buy_strength = self.generate_buy_signals(idx)
                    if len(buy_sigs) >= 1:  # Only need 1 signal now
                        risk_amount = self.equity * 0.015  # Smaller per trade (risk 1.5%)
                        
                        # VIX adjustment
                        if vix < 20:
                            risk_amount *= 1.2
                        elif vix > 30:
                            risk_amount *= 0.8
                        
                        sl = price - (atr * 2.0)
                        tp = price + (atr * 3.0)
                        qty = int(risk_amount / (price - sl))
                        
                        if qty > 0:
                            reason = ' + '.join(buy_sigs[:2])
                            self.positions.append({
                                'date': date,
                                'entry': price,
                                'qty': qty,
                                'sl': sl,
                                'tp': tp,
                                'side': 1,
                                'reason': reason,
                                'reason2': f"ADX:{adx:.0f} VIX:{vix:.0f}",
                                'strength': buy_strength
                            })
                            if len(buy_sigs) >= 2:
                                logger.info(f"{date.date()} | BUY x{len(buy_sigs)} | {reason}")
                
                # SHORT SIGNALS
                elif lt_trend < 0:  # Only in downtrend context
                    sell_sigs, sell_strength = self.generate_sell_signals(idx)
                    if len(sell_sigs) >= 1:
                        risk_amount = self.equity * 0.015
                        
                        if vix < 20:
                            risk_amount *= 1.2
                        elif vix > 30:
                            risk_amount *= 0.8
                        
                        sl = price + (atr * 2.0)
                        tp = price - (atr * 3.0)
                        qty = int(risk_amount / (sl - price))
                        
                        if qty > 0:
                            reason = ' + '.join(sell_sigs[:2])
                            self.positions.append({
                                'date': date,
                                'entry': price,
                                'qty': qty,
                                'sl': sl,
                                'tp': tp,
                                'side': -1,
                                'reason': reason,
                                'reason2': f"ADX:{adx:.0f} VIX:{vix:.0f}",
                                'strength': sell_strength
                            })
                            if len(sell_sigs) >= 2:
                                logger.info(f"{date.date()} | SELL x{len(sell_sigs)} | {reason}")
            
            self.equity_curve.append(self.equity)
        
        # Close remaining
        for pos in list(self.positions):
            price = float(self.data['Close'].iloc[-1])
            pnl = (price - pos['entry']) * pos['qty'] if pos['side'] == 1 else (pos['entry'] - price) * pos['qty']
            self.equity += pnl
            self.closed_trades.append({
                'entry_date': pos['date'],
                'exit_date': self.data.index[-1],
                'entry_price': pos['entry'],
                'exit_price': price,
                'qty': pos['qty'],
                'side': 'LONG' if pos['side'] == 1 else 'SHORT',
                'pnl': pnl,
                'pnl_pct': (pnl / (pos['entry'] * pos['qty'])) * 100,
                'hold_days': (self.data.index[-1] - pos['date']).days,
                'reason': pos['reason'],
                'reason2': pos['reason2'],
                'exit_reason': 'EOB'
            })
        
        return self.metrics()
    
    def metrics(self):
        """Calculate metrics"""
        if len(self.closed_trades) == 0:
            return {'error': 'No trades'}
        
        trades = pd.DataFrame(self.closed_trades)
        ret = ((self.equity - 100000) / 100000) * 100
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
            'return_pct': ret,
            'initial': 100000,
            'final': self.equity,
            'cagr': (((self.equity/100000)**(1/3))-1)*100,
            'trades': total,
            'wins': wins,
            'win_rate': (wins / total) * 100 if total > 0 else 0,
            'sharpe': sharpe,
            'avg_win': trades[trades['pnl'] > 0]['pnl_pct'].mean() if wins > 0 else 0,
            'avg_loss': trades[trades['pnl'] < 0]['pnl_pct'].mean() if (total - wins) > 0 else 0,
            'max_profit': trades['pnl'].max(),
            'max_loss': trades['pnl'].min(),
            'max_dd': max_dd,
            'avg_hold_days': trades['hold_days'].mean(),
            'profit_factor': (abs(trades[trades['pnl'] > 0]['pnl'].sum() + 0.1) / abs(trades[trades['pnl'] < 0]['pnl'].sum() + 0.1)),
            'best_trade': trades['pnl'].max(),
            'worst_trade': trades['pnl'].min()
        }


# Run
logger.info("=" * 70)
logger.info("AGGRESSIVE HYBRID TRADER V4")
logger.info("=" * 70)

trader = AggressiveHybridTrader('SPY')
if trader.fetch_data():
    results = trader.backtest()
    
    print("\n" + "="*70)
    print(f"AGGRESSIVE HYBRID TRADER V4 - {results['symbol']}")
    print("="*70)
    print(f"Target: Beat SPY 87.67%")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"CAGR: {results['cagr']:.2f}%")
    print(f"Trades: {results['trades']} | Wins: {results['wins']} | WR: {results['win_rate']:.1f}%")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"Max DD: {results['max_dd']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}x")
    print(f"Best Trade: ${results['best_trade']:.0f} | Worst: ${results['worst_trade']:.0f}")
    print(f"Avg Hold: {results['avg_hold_days']:.1f} days")
    print("="*70)
    
    with open('intraday/results/aggressive_hybrid_v4.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to intraday/results/aggressive_hybrid_v4.json")
    
    # Progression
    print(f"\n📈 STRATEGY EVOLUTION:")
    print(f"V2: -31.95% ❌ (fought uptrend)")
    print(f"V3: +11.77% ✓ (trend-following, few trades)")
    print(f"V4: {results['return_pct']:.2f}% {'✅' if results['return_pct'] > 20 else '🔄'} (aggressive + hybrid)")
    
    if results['return_pct'] > 87.67:
        print(f"\n🎉 BEAT SPY! {results['return_pct']:.2f}% > 87.67%")
    else:
        gap = 87.67 - results['return_pct']
        print(f"\nGap to beat SPY: {gap:.2f} percentage points")
