"""Smoke tests for indicator functions using synthetic data.

These tests avoid external data dependencies and only validate shapes/columns.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.price import (
    ma_cross_strategy,
    rsi_strategy,
    bb_rsi_strategy,
    rsi_obv_bb_strategy,
    adx_strategy,
    cci_adx_strategy,
    wr_strategy,
    vwsma_strategy,
    multi_factor_combo,
)
from strategies.volume import (
    on_balance_volume,
    vwma_signal,
    volume_oscillator_signal,
    volume_price_trend_signal,
    acc_dist_signal,
    chaikin_money_flow_signal,
    ease_of_movement_signal,
    force_index_signal,
    negative_volume_index_signal,
)


def _synthetic_ohlcv(n: int = 120):
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    base = pd.Series(np.linspace(100, 120, n), index=idx)
    df = pd.DataFrame({
        'Open': base * (1 + 0.001),
        'High': base * (1 + 0.01),
        'Low': base * (1 - 0.01),
        'Close': base,
        'Volume': np.linspace(1_000_000, 2_000_000, n),
        'Trend': ['Uptrend'] * n,
    }, index=idx)
    return df


def test_ma_cross_signal_column():
    df = _synthetic_ohlcv()
    out = ma_cross_strategy(df, sl=0.0, n1=5, n2=10)
    assert 'MA_signal' in out.columns
    assert len(out) == len(df)
    # Graded conviction should allow non-binary values
    assert (out['MA_signal'].abs() <= 5).all()


def test_rsi_signal_column():
    df = _synthetic_ohlcv()
    out = rsi_strategy(df, sl=0.0)
    assert 'RSI_signal' in out.columns
    assert len(out) == len(df)


def test_bb_rsi_signal_column():
    df = _synthetic_ohlcv()
    out = bb_rsi_strategy(df, sl=0.0)
    assert 'BBRSI_signal' in out.columns


def test_rsi_obv_bb_signal_column():
    df = _synthetic_ohlcv()
    out = rsi_obv_bb_strategy(df, sl=0.0)
    assert 'ROB_signal' in out.columns


def test_adx_signal_column():
    df = _synthetic_ohlcv()
    out = adx_strategy(df, sl=0.0)
    assert 'ADX_signal' in out.columns


def test_cci_adx_signal_column():
    df = _synthetic_ohlcv()
    out = cci_adx_strategy(df, sl=0.0)
    assert 'CDX_signal' in out.columns


def test_wr_signal_column():
    df = _synthetic_ohlcv()
    out = wr_strategy(df, sl=0.0)
    assert 'WR_signal' in out.columns


def test_vwsma_signal_column():
    df = _synthetic_ohlcv()
    out = vwsma_strategy(df, sl=0.0)
    assert 'VWSMA_signal' in out.columns


def test_multi_factor_combo_column():
    df = _synthetic_ohlcv()
    out = multi_factor_combo(df, max_signal_abs=5.0)
    assert 'combo_signal' in out.columns
    assert len(out) == len(df)
    assert (out['combo_signal'].abs() <= 5).all()


def test_volume_signals_shapes():
    df = _synthetic_ohlcv()
    close, volume = df['Close'], df['Volume']
    high, low = df['High'], df['Low']

    assert len(on_balance_volume(close, volume)) == len(df)
    assert len(vwma_signal(close, volume)) == len(df)
    assert len(volume_oscillator_signal(volume)) == len(df)
    assert len(volume_price_trend_signal(close, volume)) == len(df)
    assert len(acc_dist_signal(high, low, close, volume)) == len(df)
    assert len(chaikin_money_flow_signal(high, low, close, volume)) == len(df)
    assert len(ease_of_movement_signal(high, low, close, volume)) == len(df)
    assert len(force_index_signal(close, volume)) == len(df)
    assert len(negative_volume_index_signal(close, volume)) == len(df)
