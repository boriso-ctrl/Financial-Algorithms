"""Debug: Check Phase 3 signals with and without regime filters."""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.financial_algorithms.data.yfinance_loader import load_stock_bars
from src.financial_algorithms.strategies.hybrid_phase3_phase6 import HybridPhase3Phase6


def debug_signals():
    """Load data and check signal generation at each step."""
    
    print("\n" + "="*70)
    print("DEBUG: Phase 3 Base Signals vs Regime Filters")
    print("="*70)
    
    # Load data
    print("\nLoading AAPL, MSFT, AMZN (1 year daily)...")
    df = load_stock_bars(['AAPL', 'MSFT', 'AMZN'], interval='1d', period='1y')
    print(f"[OK] Loaded {len(df):,} bars")
    
    # Test hybrid with LOOSE parameters
    hybrid_loose = HybridPhase3Phase6(
        rsi_threshold_oversold=10,  # Very loose
        rsi_threshold_overbought=90,  # Very loose
        min_volume_ma_ratio=0.3,  # Very loose
    )
    
    # Test hybrid with TIGHT parameters
    hybrid_tight = HybridPhase3Phase6(
        rsi_threshold_oversold=45,  # Very tight
        rsi_threshold_overbought=55,  # Very tight
        min_volume_ma_ratio=0.95,  # Very tight
    )
    
    # Test on one symbol
    df_aapl = df[df['symbol'] == 'AAPL'].reset_index(drop=True)
    print(f"\nAnalyzing AAPL ({len(df_aapl)} bars)...")
    
    # Check Phase 3 base signal
    base_signals = hybrid_loose.generate_phase3_signal(df_aapl)
    num_base_signals = (base_signals != 0).sum()
    print(f"\nPhase 3 Base Signals: {num_base_signals} (trading days)")
    
    # Check regime filters
    regime_loose = hybrid_loose.regime_filter(df_aapl)
    regime_tight = hybrid_tight.regime_filter(df_aapl)
    
    print(f"Regime Filter (Loose 10-90 RSI, 0.3 Vol): {regime_loose.sum()} days allowed")
    print(f"Regime Filter (Tight 45-55 RSI, 0.95 Vol): {regime_tight.sum()} days allowed")
    
    # Check signals after filtering
    filtered_loose, _, _ = hybrid_loose.generate_hybrid_signal(df_aapl)
    filtered_tight, _, _ = hybrid_tight.generate_hybrid_signal(df_aapl)
    
    num_filtered_loose = (filtered_loose != 0).sum()
    num_filtered_tight = (filtered_tight != 0).sum()
    
    print(f"Hybrid Signals (Loose): {num_filtered_loose}")
    print(f"Hybrid Signals (Tight): {num_filtered_tight}")
    
    # Detailed regime analysis
    print(f"\n{'-'*70}")
    print("REGIME FILTER ANALYSIS")
    print(f"{'-'*70}")
    
    # Check RSI distribution
    rsi = hybrid_loose.calculate_rsi(df_aapl['close'])
    print(f"\nRSI Statistics:")
    print(f"  Min: {rsi.min():.1f}, Max: {rsi.max():.1f}")
    print(f"  Mean: {rsi.mean():.1f}, Std: {rsi.std():.1f}")
    print(f"  % Below 30: {(rsi < 30).sum() / len(rsi) * 100:.1f}%")
    print(f"  % Between 30-70: {((rsi >= 30) & (rsi <= 70)).sum() / len(rsi) * 100:.1f}%")
    print(f"  % Above 70: {(rsi > 70).sum() / len(rsi) * 100:.1f}%")
    
    # Check volume distribution
    vol_ma = df_aapl['volume'].rolling(20).mean()
    vol_ratio = df_aapl['volume'] / vol_ma
    print(f"\nVolume Ratio (current / 20-day MA):")
    print(f"  Min: {vol_ratio.min():.2f}, Max: {vol_ratio.max():.2f}")
    print(f"  Mean: {vol_ratio.mean():.2f}, Std: {vol_ratio.std():.2f}")
    print(f"  % Below 0.5: {(vol_ratio < 0.5).sum() / len(vol_ratio) * 100:.1f}%")
    print(f"  % Between 0.5-1.0: {((vol_ratio >= 0.5) & (vol_ratio < 1.0)).sum() / len(vol_ratio) * 100:.1f}%")
    print(f"  % Above 1.0: {(vol_ratio >= 1.0).sum() / len(vol_ratio) * 100:.1f}%")
    
    # Check volatility
    volatility = hybrid_loose.calculate_volatility(df_aapl['high'], df_aapl['low'], df_aapl['close'])
    print(f"\nATR Volatility (% of close):")
    print(f"  Min: {volatility.min()*100:.2f}%, Max: {volatility.max()*100:.2f}%")
    print(f"  Mean: {volatility.mean()*100:.2f}%, Std: {volatility.std()*100:.2f}%")
    print(f"  % < 1%: {(volatility < 0.01).sum() / len(volatility) * 100:.1f}%")
    print(f"  % 1-3%: {((volatility >= 0.01) & (volatility < 0.03)).sum() / len(volatility) * 100:.1f}%")
    print(f"  % > 3%: {(volatility >= 0.03).sum() / len(volatility) * 100:.1f}%")
    
    # Show sample of signals and regimes
    print(f"\n{'-'*70}")
    print("SAMPLE: First 50 trading days")
    print(f"{'-'*70}")
    
    sample = df_aapl.head(50)[['timestamp', 'close', 'volume']].copy()
    sample['Base Signal'] = base_signals.head(50).values
    sample['RSI'] = rsi.head(50).values
    sample['Vol Ratio'] = vol_ratio.head(50).values
    sample['Regime (Loose)'] = regime_loose.head(50).values
    sample['Regime (Tight)'] = regime_tight.head(50).values
    sample['Hybrid Loose'] = filtered_loose.head(50).values
    sample['Hybrid Tight'] = filtered_tight.head(50).values
    
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print(sample[['timestamp', 'close', 'Base Signal', 'RSI', 'Vol Ratio', 
                  'Regime (Loose)', 'Regime (Tight)', 'Hybrid Loose', 'Hybrid Tight']].to_string())


if __name__ == "__main__":
    debug_signals()
