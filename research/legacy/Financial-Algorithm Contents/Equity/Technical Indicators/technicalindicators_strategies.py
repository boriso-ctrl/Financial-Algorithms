# -*- coding: utf-8 -*-
"""
Compatibility shim for legacy imports.

All strategy functions are now split into individual modules under the
`indicators.price` package. Importing from this file re-exports them.
"""

import os
import sys

# Add repository root to sys.path so `indicators` can be imported when this
# file is executed from its legacy location.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from indicators.price import (
    ma_cross_strategy,
    sar_stoch_strategy,
    stoch_macd_strategy,
    rsi_strategy,
    bb_rsi_strategy,
    rsi_obv_bb_strategy,
    adx_strategy,
    cci_adx_strategy,
    wr_strategy,
    vwsma_strategy,
)

__all__ = [
    'ma_cross_strategy',
    'sar_stoch_strategy',
    'stoch_macd_strategy',
    'rsi_strategy',
    'bb_rsi_strategy',
    'rsi_obv_bb_strategy',
    'adx_strategy',
    'cci_adx_strategy',
    'wr_strategy',
    'vwsma_strategy',
]
