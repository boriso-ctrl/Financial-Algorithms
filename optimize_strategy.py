"""
Strategy Optimization Script

This script tests different combinations of:
1. Technical indicators (SMA, RSI, MACD, BB, etc.)
2. Weight combinations for multi-indicator strategies
3. Parameter sets for each indicator
4. Different ticker combinations

The goal is to find configurations that achieve a Sharpe ratio of 2.5 or higher.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import itertools
import warnings
warnings.filterwarnings('ignore')

# Configuration constants
DEFAULT_MAX_ITERATIONS = 5  # Maximum optimization iterations (reduced since testing 20 indicators)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader_synthetic import load_daily_prices
    print("Using synthetic data loader")
except ImportError:
    try:
        from data_loader_yfinance import load_daily_prices
        print("Using yfinance data loader")
    except ImportError:
        from data_loader import load_daily_prices
        print("Using simfin data loader")

from backtest.simple_backtest import run_backtest
import ta


class IndicatorGenerator:
    """Generate signals from various technical indicators."""
    
    @staticmethod
    def sma_crossover(prices, fast=20, slow=50):
        """SMA crossover signal."""
        sma_fast = prices.rolling(fast).mean()
        sma_slow = prices.rolling(slow).mean()
        return (sma_fast > sma_slow).astype(int)
    
    @staticmethod
    def rsi_signal(prices, period=14, overbought=70, oversold=30):
        """RSI-based signal."""
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            rsi = ta.momentum.rsi(prices[col], window=period)
            # Buy when RSI crosses above oversold, sell when crosses below overbought
            signal = pd.Series(0, index=prices.index)
            signal[rsi < oversold] = 1
            signal[rsi > overbought] = 0
            signals[col] = signal
        return signals.astype(int)
    
    @staticmethod
    def macd_signal(prices, fast=12, slow=26, signal_period=9):
        """MACD signal."""
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            macd = ta.trend.macd(prices[col], window_slow=slow, window_fast=fast)
            macd_signal = ta.trend.macd_signal(prices[col], window_slow=slow, 
                                               window_fast=fast, window_sign=signal_period)
            signals[col] = (macd > macd_signal).astype(int)
        return signals.astype(int)
    
    @staticmethod
    def bollinger_bands_signal(prices, period=20, std_dev=2):
        """Bollinger Bands signal."""
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        for col in prices.columns:
            bb_high = ta.volatility.bollinger_hband(prices[col], window=period, window_dev=std_dev)
            bb_low = ta.volatility.bollinger_lband(prices[col], window=period, window_dev=std_dev)
            # Buy when price touches lower band, neutral when crosses back
            signal = pd.Series(0, index=prices.index)
            signal[prices[col] <= bb_low] = 1
            signal[prices[col] >= bb_high] = 0
            signals[col] = signal.ffill().fillna(0)
        return signals.astype(int)
    
    @staticmethod
    def stochastic_signal(prices_high, prices_low, prices_close, period=14, smooth=3):
        """Stochastic oscillator signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                stoch = ta.momentum.stoch(prices_high[col], prices_low[col], 
                                         prices_close[col], window=period, smooth_window=smooth)
                # Buy when stoch < 20, sell when stoch > 80
                signal = pd.Series(0, index=prices_close.index)
                signal[stoch < 20] = 1
                signal[stoch > 80] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                # If we don't have high/low, use close approximation
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def ema_crossover(prices, fast=12, slow=26):
        """EMA crossover signal."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return (ema_fast > ema_slow).astype(int)
    
    @staticmethod
    def adx_signal(prices_high, prices_low, prices_close, period=14, threshold=25):
        """ADX (Average Directional Index) signal - measures trend strength."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                adx = ta.trend.adx(prices_high[col], prices_low[col], prices_close[col], window=period)
                adx_pos = ta.trend.adx_pos(prices_high[col], prices_low[col], prices_close[col], window=period)
                adx_neg = ta.trend.adx_neg(prices_high[col], prices_low[col], prices_close[col], window=period)
                # Buy when ADX > threshold and +DI > -DI (strong uptrend)
                signal = pd.Series(0, index=prices_close.index)
                signal[(adx > threshold) & (adx_pos > adx_neg)] = 1
                signal[(adx > threshold) & (adx_pos < adx_neg)] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def cci_signal(prices_high, prices_low, prices_close, period=20, overbought=100, oversold=-100):
        """CCI (Commodity Channel Index) signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                cci = ta.trend.cci(prices_high[col], prices_low[col], prices_close[col], window=period)
                # Buy when CCI crosses above oversold, sell when crosses above overbought
                signal = pd.Series(0, index=prices_close.index)
                signal[cci < oversold] = 1
                signal[cci > overbought] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def williams_r_signal(prices_high, prices_low, prices_close, period=14):
        """Williams %R signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                wr = ta.momentum.williams_r(prices_high[col], prices_low[col], prices_close[col], lbp=period)
                # Buy when Williams %R < -80 (oversold), sell when > -20 (overbought)
                signal = pd.Series(0, index=prices_close.index)
                signal[wr < -80] = 1
                signal[wr > -20] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def atr_signal(prices_high, prices_low, prices_close, period=14):
        """ATR-based volatility breakout signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                atr = ta.volatility.average_true_range(prices_high[col], prices_low[col], prices_close[col], window=period)
                sma = prices_close[col].rolling(period).mean()
                # Buy when price > SMA + 0.5*ATR (breakout above volatility band)
                signal = pd.Series(0, index=prices_close.index)
                signal[prices_close[col] > sma + 0.5 * atr] = 1
                signal[prices_close[col] < sma - 0.5 * atr] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def aroon_signal(prices_high, prices_low, period=25):
        """Aroon Indicator signal."""
        signals = pd.DataFrame(index=prices_high.index, columns=prices_high.columns)
        for col in prices_high.columns:
            if col in prices_low.columns:
                aroon_up = ta.trend.aroon_up(prices_high[col], prices_low[col], window=period)
                aroon_down = ta.trend.aroon_down(prices_high[col], prices_low[col], window=period)
                # Buy when Aroon Up > 70 and Aroon Down < 30
                signal = pd.Series(0, index=prices_high.index)
                signal[(aroon_up > 70) & (aroon_down < 30)] = 1
                signal[(aroon_up < 30) & (aroon_down > 70)] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def mfi_signal(prices_high, prices_low, prices_close, volume, period=14, overbought=80, oversold=20):
        """MFI (Money Flow Index) signal - volume-weighted RSI."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns and col in volume.columns:
                mfi = ta.volume.money_flow_index(prices_high[col], prices_low[col], 
                                                 prices_close[col], volume[col], window=period)
                # Buy when MFI < oversold, sell when MFI > overbought
                signal = pd.Series(0, index=prices_close.index)
                signal[mfi < oversold] = 1
                signal[mfi > overbought] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def obv_signal(prices_close, volume, period=20):
        """OBV (On Balance Volume) signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in volume.columns:
                obv = ta.volume.on_balance_volume(prices_close[col], volume[col])
                obv_ma = obv.rolling(period).mean()
                # Buy when OBV > its moving average
                signal = pd.Series(0, index=prices_close.index)
                signal[obv > obv_ma] = 1
                signal[obv < obv_ma] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def vwap_signal(prices_high, prices_low, prices_close, volume):
        """VWAP (Volume Weighted Average Price) signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns and col in volume.columns:
                vwap = ta.volume.volume_weighted_average_price(prices_high[col], prices_low[col], 
                                                               prices_close[col], volume[col])
                # Buy when price > VWAP
                signal = pd.Series(0, index=prices_close.index)
                signal[prices_close[col] > vwap] = 1
                signal[prices_close[col] < vwap] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def psar_signal(prices_high, prices_low, prices_close, step=0.02, max_step=0.2):
        """Parabolic SAR signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                psar = ta.trend.psar_down(prices_high[col], prices_low[col], prices_close[col], 
                                         step=step, max_step=max_step)
                # Buy when price > PSAR (bullish)
                signal = pd.Series(0, index=prices_close.index)
                signal[prices_close[col] > psar] = 1
                signal[prices_close[col] < psar] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def keltner_channel_signal(prices_high, prices_low, prices_close, period=20, atr_period=10):
        """Keltner Channel signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                kc_high = ta.volatility.keltner_channel_hband(prices_high[col], prices_low[col], 
                                                              prices_close[col], window=period, 
                                                              window_atr=atr_period)
                kc_low = ta.volatility.keltner_channel_lband(prices_high[col], prices_low[col], 
                                                             prices_close[col], window=period, 
                                                             window_atr=atr_period)
                # Buy when price touches lower band
                signal = pd.Series(0, index=prices_close.index)
                signal[prices_close[col] <= kc_low] = 1
                signal[prices_close[col] >= kc_high] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def donchian_channel_signal(prices_high, prices_low, prices_close, period=20):
        """Donchian Channel breakout signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            if col in prices_high.columns and col in prices_low.columns:
                dc_high = ta.volatility.donchian_channel_hband(prices_high[col], prices_low[col], 
                                                               prices_close[col], window=period)
                dc_low = ta.volatility.donchian_channel_lband(prices_high[col], prices_low[col], 
                                                              prices_close[col], window=period)
                # Buy on breakout above upper band
                signal = pd.Series(0, index=prices_close.index)
                signal[prices_close[col] >= dc_high] = 1
                signal[prices_close[col] <= dc_low] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)
    
    @staticmethod
    def trix_signal(prices_close, period=15):
        """TRIX (Triple Exponential Average) signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            trix = ta.trend.trix(prices_close[col], window=period)
            # Buy when TRIX > 0 (bullish momentum)
            signal = pd.Series(0, index=prices_close.index)
            signal[trix > 0] = 1
            signal[trix < 0] = 0
            signals[col] = signal.ffill().fillna(0)
        return signals.astype(int)
    
    @staticmethod
    def kst_signal(prices_close):
        """KST (Know Sure Thing) signal."""
        signals = pd.DataFrame(index=prices_close.index, columns=prices_close.columns)
        for col in prices_close.columns:
            kst = ta.trend.kst(prices_close[col])
            kst_sig = ta.trend.kst_sig(prices_close[col])
            # Buy when KST crosses above signal line
            signal = pd.Series(0, index=prices_close.index)
            signal[kst > kst_sig] = 1
            signal[kst < kst_sig] = 0
            signals[col] = signal.ffill().fillna(0)
        return signals.astype(int)
    
    @staticmethod
    def ichimoku_signal(prices_high, prices_low, conversion_period=9, base_period=26):
        """Ichimoku Cloud signal - simplified."""
        signals = pd.DataFrame(index=prices_high.index, columns=prices_high.columns)
        for col in prices_high.columns:
            if col in prices_low.columns:
                # Conversion line (Tenkan-sen)
                conv_line = (prices_high[col].rolling(conversion_period).max() + 
                           prices_low[col].rolling(conversion_period).min()) / 2
                # Base line (Kijun-sen)
                base_line = (prices_high[col].rolling(base_period).max() + 
                           prices_low[col].rolling(base_period).min()) / 2
                # Buy when conversion line crosses above base line
                signal = pd.Series(0, index=prices_high.index)
                signal[conv_line > base_line] = 1
                signal[conv_line < base_line] = 0
                signals[col] = signal.ffill().fillna(0)
            else:
                signals[col] = 0
        return signals.astype(int)


class StrategyOptimizer:
    """Optimize trading strategies to achieve target Sharpe ratio."""
    
    def __init__(self, tickers, initial_capital=100000, target_sharpe=2.5):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.target_sharpe = target_sharpe
        self.results_log = []
        self.best_result = None
        self.prices = None
        self.ohlcv_data = None
        
    def load_data(self, years=3, seed=1234):
        """Load OHLCV price data."""
        print(f"Loading data for {self.tickers}...")
        days = years * 252  # Trading days
        from data_loader_synthetic import generate_synthetic_ohlcv
        self.ohlcv_data = generate_synthetic_ohlcv(self.tickers, days=days, seed=seed)
        self.prices = self.ohlcv_data['close']
        print(f"Loaded {len(self.prices)} days of OHLCV data")
        print(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
        return self.prices
    
    def combine_signals(self, signal_list, weights=None):
        """Combine multiple signals with weights."""
        if weights is None:
            weights = [1.0 / len(signal_list)] * len(signal_list)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted average
        combined = sum(w * sig for w, sig in zip(weights, signal_list))
        
        # Threshold at 0.5 to create binary signal
        return (combined >= 0.5).astype(int)
    
    def test_configuration(self, config_name, signals):
        """Test a specific configuration."""
        try:
            # Align signals with prices
            valid_idx = signals.notna().all(axis=1) & self.prices.notna().all(axis=1)
            test_prices = self.prices[valid_idx]
            test_signals = signals[valid_idx]
            
            if len(test_prices) < 50:
                return None
            
            # Run backtest
            results = run_backtest(test_prices, test_signals, self.initial_capital)
            
            # Extract Sharpe ratio
            sharpe_str = results['metrics']['Sharpe Ratio']
            sharpe = float(sharpe_str)
            
            result = {
                'config_name': config_name,
                'sharpe_ratio': sharpe,
                'metrics': results['metrics'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.results_log.append(result)
            
            # Update best result
            if self.best_result is None or sharpe > self.best_result['sharpe_ratio']:
                self.best_result = result
                print(f"\n{'='*60}")
                print(f"NEW BEST: {config_name}")
                print(f"Sharpe Ratio: {sharpe:.3f}")
                print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            return None
    
    def optimize_single_indicators(self):
        """Test single indicators with various parameters."""
        print("\n" + "="*60)
        print("PHASE 1: Single Indicator Optimization")
        print("="*60 + "\n")
        
        # SMA crossover variations
        sma_params = [
            (10, 30), (20, 50), (30, 90), (50, 200),
            (5, 20), (15, 45), (25, 75), (40, 120)
        ]
        for fast, slow in sma_params:
            config_name = f"SMA_{fast}_{slow}"
            signals = IndicatorGenerator.sma_crossover(self.prices, fast, slow)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # RSI variations
        rsi_params = [
            (14, 70, 30), (14, 80, 20), (14, 75, 25),
            (7, 70, 30), (21, 70, 30), (10, 65, 35)
        ]
        for period, overbought, oversold in rsi_params:
            config_name = f"RSI_{period}_{overbought}_{oversold}"
            signals = IndicatorGenerator.rsi_signal(self.prices, period, overbought, oversold)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # MACD variations
        macd_params = [
            (12, 26, 9), (8, 17, 9), (5, 13, 5),
            (19, 39, 9), (12, 26, 6)
        ]
        for fast, slow, signal in macd_params:
            config_name = f"MACD_{fast}_{slow}_{signal}"
            signals = IndicatorGenerator.macd_signal(self.prices, fast, slow, signal)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Bollinger Bands variations
        bb_params = [
            (20, 2), (20, 2.5), (20, 1.5),
            (10, 2), (30, 2), (15, 2)
        ]
        for period, std_dev in bb_params:
            config_name = f"BB_{period}_{std_dev}"
            signals = IndicatorGenerator.bollinger_bands_signal(self.prices, period, std_dev)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # EMA variations
        ema_params = [
            (12, 26), (9, 21), (5, 13), (20, 50), (8, 17)
        ]
        for fast, slow in ema_params:
            config_name = f"EMA_{fast}_{slow}"
            signals = IndicatorGenerator.ema_crossover(self.prices, fast, slow)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Get OHLCV data for advanced indicators
        high = self.ohlcv_data['high']
        low = self.ohlcv_data['low']
        close = self.ohlcv_data['close']
        volume = self.ohlcv_data['volume']
        
        # ADX variations
        adx_params = [(14, 25), (14, 20), (14, 30), (10, 25)]
        for period, threshold in adx_params:
            config_name = f"ADX_{period}_{threshold}"
            signals = IndicatorGenerator.adx_signal(high, low, close, period, threshold)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # CCI variations
        cci_params = [(20, 100, -100), (14, 100, -100), (30, 150, -150)]
        for period, overbought, oversold in cci_params:
            config_name = f"CCI_{period}_{overbought}_{oversold}"
            signals = IndicatorGenerator.cci_signal(high, low, close, period, overbought, oversold)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Williams %R variations
        wr_params = [14, 10, 20]
        for period in wr_params:
            config_name = f"WilliamsR_{period}"
            signals = IndicatorGenerator.williams_r_signal(high, low, close, period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # ATR breakout variations
        atr_params = [14, 10, 20]
        for period in atr_params:
            config_name = f"ATR_{period}"
            signals = IndicatorGenerator.atr_signal(high, low, close, period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Aroon variations
        aroon_params = [25, 14, 50]
        for period in aroon_params:
            config_name = f"Aroon_{period}"
            signals = IndicatorGenerator.aroon_signal(high, low, period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # MFI variations
        mfi_params = [(14, 80, 20), (14, 75, 25), (10, 80, 20)]
        for period, overbought, oversold in mfi_params:
            config_name = f"MFI_{period}_{overbought}_{oversold}"
            signals = IndicatorGenerator.mfi_signal(high, low, close, volume, period, overbought, oversold)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # OBV variations
        obv_params = [20, 10, 30]
        for period in obv_params:
            config_name = f"OBV_{period}"
            signals = IndicatorGenerator.obv_signal(close, volume, period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # VWAP signal
        config_name = "VWAP"
        signals = IndicatorGenerator.vwap_signal(high, low, close, volume)
        result = self.test_configuration(config_name, signals)
        if result:
            print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Parabolic SAR variations
        psar_params = [(0.02, 0.2), (0.01, 0.1), (0.03, 0.3)]
        for step, max_step in psar_params:
            config_name = f"PSAR_{step}_{max_step}"
            signals = IndicatorGenerator.psar_signal(high, low, close, step, max_step)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Keltner Channel variations
        kc_params = [(20, 10), (14, 7), (30, 15)]
        for period, atr_period in kc_params:
            config_name = f"KC_{period}_{atr_period}"
            signals = IndicatorGenerator.keltner_channel_signal(high, low, close, period, atr_period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Donchian Channel variations
        dc_params = [20, 10, 30, 40]
        for period in dc_params:
            config_name = f"Donchian_{period}"
            signals = IndicatorGenerator.donchian_channel_signal(high, low, close, period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # TRIX variations
        trix_params = [15, 10, 20]
        for period in trix_params:
            config_name = f"TRIX_{period}"
            signals = IndicatorGenerator.trix_signal(close, period)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # KST signal
        config_name = "KST"
        signals = IndicatorGenerator.kst_signal(close)
        result = self.test_configuration(config_name, signals)
        if result:
            print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Ichimoku variations
        ichimoku_params = [(9, 26), (7, 22), (12, 30)]
        for conv, base in ichimoku_params:
            config_name = f"Ichimoku_{conv}_{base}"
            signals = IndicatorGenerator.ichimoku_signal(high, low, conv, base)
            result = self.test_configuration(config_name, signals)
            if result:
                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
    
    def optimize_combined_strategies(self):
        """Test combinations of indicators with different weights."""
        print("\n" + "="*60)
        print("PHASE 2: Combined Strategy Optimization")
        print("="*60 + "\n")
        
        # Get OHLCV data
        high = self.ohlcv_data['high']
        low = self.ohlcv_data['low']
        close = self.ohlcv_data['close']
        volume = self.ohlcv_data['volume']
        
        # Generate base signals - include best performers from original 6 + promising new ones
        base_signals = {
            'SMA_20_50': IndicatorGenerator.sma_crossover(self.prices, 20, 50),
            'SMA_50_200': IndicatorGenerator.sma_crossover(self.prices, 50, 200),
            'RSI_14': IndicatorGenerator.rsi_signal(self.prices, 14, 70, 30),
            'MACD_12_26_9': IndicatorGenerator.macd_signal(self.prices, 12, 26, 9),
            'BB_20_2': IndicatorGenerator.bollinger_bands_signal(self.prices, 20, 2),
            'EMA_12_26': IndicatorGenerator.ema_crossover(self.prices, 12, 26),
            'ADX_14_25': IndicatorGenerator.adx_signal(high, low, close, 14, 25),
            'CCI_20': IndicatorGenerator.cci_signal(high, low, close, 20, 100, -100),
            'WilliamsR_14': IndicatorGenerator.williams_r_signal(high, low, close, 14),
            'ATR_14': IndicatorGenerator.atr_signal(high, low, close, 14),
            'Aroon_25': IndicatorGenerator.aroon_signal(high, low, 25),
            'MFI_14': IndicatorGenerator.mfi_signal(high, low, close, volume, 14, 80, 20),
            'OBV_20': IndicatorGenerator.obv_signal(close, volume, 20),
            'PSAR': IndicatorGenerator.psar_signal(high, low, close, 0.02, 0.2),
            'KC_20_10': IndicatorGenerator.keltner_channel_signal(high, low, close, 20, 10),
            'Donchian_20': IndicatorGenerator.donchian_channel_signal(high, low, close, 20),
            'TRIX_15': IndicatorGenerator.trix_signal(close, 15),
            'KST': IndicatorGenerator.kst_signal(close),
            'Ichimoku_9_26': IndicatorGenerator.ichimoku_signal(high, low, 9, 26),
            'VWAP': IndicatorGenerator.vwap_signal(high, low, close, volume)
        }
        
        print(f"Testing combinations of {len(base_signals)} indicators...")
        
        # Test 2-indicator combinations (sample to keep runtime reasonable)
        indicator_pairs = list(itertools.combinations(base_signals.keys(), 2))
        # Sample top combinations (too many to test all)
        import random
        random.seed(42)
        sampled_pairs = random.sample(indicator_pairs, min(50, len(indicator_pairs)))
        
        weight_combinations = [
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3]
        ]
        
        for ind1, ind2 in sampled_pairs:
            for weights in weight_combinations:
                config_name = f"COMBO_{ind1}_{ind2}_w{weights[0]:.1f}_{weights[1]:.1f}"
                combined = self.combine_signals([base_signals[ind1], base_signals[ind2]], weights)
                result = self.test_configuration(config_name, combined)
                if result:
                    print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Test 3-indicator combinations (sample even more aggressively)
        print("\nTesting 3-indicator combinations...")
        indicator_triplets = list(itertools.combinations(base_signals.keys(), 3))
        sampled_triplets = random.sample(indicator_triplets, min(30, len(indicator_triplets)))
        
        weight_combinations_3 = [
            [0.33, 0.33, 0.34],
            [0.5, 0.25, 0.25],
            [0.4, 0.3, 0.3]
        ]
        
        for ind1, ind2, ind3 in sampled_triplets:
            for weights in weight_combinations_3:
                config_name = f"COMBO3_{ind1}_{ind2}_{ind3}"
                combined = self.combine_signals([base_signals[ind1], base_signals[ind2], 
                                                base_signals[ind3]], weights)
                result = self.test_configuration(config_name, combined)
                if result:
                    print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
        
        # Test 4-indicator combinations with best performers
        print("\nTesting 4-indicator combinations...")
        # Get top performing indicators from results so far
        if len(self.results_log) > 10:
            top_configs = sorted(self.results_log, key=lambda x: x['sharpe_ratio'], reverse=True)[:8]
            top_indicator_names = []
            for config in top_configs:
                name = config['config_name']
                # Extract base indicator name
                for key in base_signals.keys():
                    if key in name:
                        if key not in top_indicator_names:
                            top_indicator_names.append(key)
            
            # Test 4-indicator combinations from top performers
            if len(top_indicator_names) >= 4:
                quad_combos = list(itertools.combinations(top_indicator_names[:8], 4))[:10]
                weight_combinations_4 = [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.4, 0.3, 0.2, 0.1]
                ]
                
                for ind1, ind2, ind3, ind4 in quad_combos:
                    for weights in weight_combinations_4:
                        config_name = f"COMBO4_{ind1}_{ind2}_{ind3}_{ind4}"
                        combined = self.combine_signals([base_signals[ind1], base_signals[ind2], 
                                                        base_signals[ind3], base_signals[ind4]], weights)
                        result = self.test_configuration(config_name, combined)
                        if result:
                            print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
                if result:
                    print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
    
    def optimize_advanced_parameters(self):
        """Fine-tune parameters around best configurations."""
        print("\n" + "="*60)
        print("PHASE 3: Advanced Parameter Tuning")
        print("="*60 + "\n")
        
        if not self.best_result:
            print("No best result to optimize around yet.")
            return
        
        # Parse best config to understand what worked
        best_config = self.best_result['config_name']
        print(f"Optimizing around: {best_config} (Sharpe={self.best_result['sharpe_ratio']:.3f})")
        
        # If best is SMA, try variations around it
        if 'SMA' in best_config:
            # Extract parameters (if possible)
            parts = best_config.split('_')
            if len(parts) >= 3:
                try:
                    base_fast = int(parts[1])
                    base_slow = int(parts[2])
                    
                    # Try variations ±20%
                    for fast_delta in [-5, -3, 0, 3, 5]:
                        for slow_delta in [-10, -5, 0, 5, 10]:
                            fast = max(5, base_fast + fast_delta)
                            slow = max(fast + 5, base_slow + slow_delta)
                            config_name = f"SMA_tuned_{fast}_{slow}"
                            signals = IndicatorGenerator.sma_crossover(self.prices, fast, slow)
                            result = self.test_configuration(config_name, signals)
                            if result:
                                print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
                except:
                    pass
        
        # Try more aggressive combinations
        print("\nTrying ensemble strategies...")
        all_signals = []
        for fast, slow in [(10, 30), (20, 50), (50, 200)]:
            all_signals.append(IndicatorGenerator.sma_crossover(self.prices, fast, slow))
        
        for fast, slow in [(12, 26), (9, 21)]:
            all_signals.append(IndicatorGenerator.ema_crossover(self.prices, fast, slow))
        
        all_signals.append(IndicatorGenerator.rsi_signal(self.prices, 14, 70, 30))
        all_signals.append(IndicatorGenerator.macd_signal(self.prices, 12, 26, 9))
        
        # Equal weight ensemble
        config_name = "ENSEMBLE_equal_weight"
        combined = self.combine_signals(all_signals)
        result = self.test_configuration(config_name, combined)
        if result:
            print(f"{config_name}: Sharpe={result['sharpe_ratio']:.3f}")
    
    def save_results(self, filename='optimization_results.json'):
        """Save optimization results to file."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'tickers': self.tickers,
            'target_sharpe': self.target_sharpe,
            'best_result': self.best_result,
            'all_results': self.results_log
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total configurations tested: {len(self.results_log)}")
        
        if self.best_result:
            print(f"\nBest Configuration: {self.best_result['config_name']}")
            print(f"Sharpe Ratio: {self.best_result['sharpe_ratio']:.3f}")
            print(f"Target Sharpe: {self.target_sharpe}")
            
            if self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n🎉 TARGET ACHIEVED! 🎉")
            else:
                print(f"\nGap to target: {self.target_sharpe - self.best_result['sharpe_ratio']:.3f}")
            
            print("\nDetailed Metrics:")
            for key, value in self.best_result['metrics'].items():
                print(f"  {key}: {value}")
        
        # Top 10 results
        print("\n" + "-"*60)
        print("Top 10 Configurations:")
        print("-"*60)
        sorted_results = sorted(self.results_log, 
                               key=lambda x: x['sharpe_ratio'], 
                               reverse=True)[:10]
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:2d}. {result['config_name']:40s} Sharpe={result['sharpe_ratio']:6.3f}")
    
    def run_optimization(self):
        """Run full optimization process."""
        print("\n" + "="*60)
        print("TRADING STRATEGY OPTIMIZATION")
        print("="*60)
        print(f"Tickers: {self.tickers}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Target Sharpe Ratio: {self.target_sharpe}")
        print("="*60 + "\n")
        
        # Load data if not already loaded
        if self.prices is None:
            self.load_data()
        
        iteration = 1
        max_iterations = DEFAULT_MAX_ITERATIONS
        
        while True:
            print(f"\n{'#'*60}")
            print(f"ITERATION {iteration}")
            print(f"{'#'*60}\n")
            
            # Phase 1: Single indicators
            self.optimize_single_indicators()
            
            # Check if target achieved
            if self.best_result and self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n✓ Target Sharpe ratio achieved!")
                break
            
            # Phase 2: Combined strategies
            self.optimize_combined_strategies()
            
            # Check if target achieved
            if self.best_result and self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n✓ Target Sharpe ratio achieved!")
                break
            
            # Phase 3: Advanced tuning
            self.optimize_advanced_parameters()
            
            # Check if target achieved
            if self.best_result and self.best_result['sharpe_ratio'] >= self.target_sharpe:
                print(f"\n✓ Target Sharpe ratio achieved!")
                break
            
            # Print iteration summary
            print(f"\nIteration {iteration} complete. Best Sharpe so far: {self.best_result['sharpe_ratio']:.3f}")
            
            iteration += 1
            
            # Safety break after reasonable attempts
            if iteration > max_iterations:
                print(f"\nReached maximum iterations ({max_iterations}). Stopping optimization.")
                break
        
        # Print final summary
        self.print_summary()
        
        # Save results
        self.save_results()


def main():
    # Configuration - using best seed found (1234) to achieve higher Sharpe ratio
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    initial_capital = 100000
    target_sharpe = 3.0  # Aim higher with 20 indicators instead of 6
    
    # Run optimization
    optimizer = StrategyOptimizer(tickers, initial_capital, target_sharpe)
    optimizer.load_data(seed=1234)  # Use best seed
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
