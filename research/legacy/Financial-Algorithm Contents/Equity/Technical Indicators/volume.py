# -*- coding: utf-8 -*-
"""
Compatibility shim for legacy imports.

All volume indicators are now split into individual modules under the
`indicators.volume` package. Importing from this file re-exports them.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from indicators.volume import (
    acc_dist_index,
    on_balance_volume,
    chaikin_money_flow,
    force_index,
    ease_of_movement,
    volume_price_trend,
    volume_oscillator,
    volume_weighted_moving_average,
    negative_volume_index,
    put_call_ratio,
)

__all__ = [
    'acc_dist_index',
    'on_balance_volume',
    'chaikin_money_flow',
    'force_index',
    'ease_of_movement',
    'volume_price_trend',
    'volume_oscillator',
    'volume_weighted_moving_average',
    'negative_volume_index',
    'put_call_ratio',
]
