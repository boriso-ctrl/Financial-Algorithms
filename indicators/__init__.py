"""Compatibility shims for legacy imports; prefer strategies.price/volume.

Imports resolve directly from indicator modules to avoid circular deps with strategies.
"""

from indicators.price.ma_cross import ma_cross_strategy  # noqa: F401
from indicators.price.rsi import rsi_strategy  # noqa: F401
from indicators.price.bb_rsi import bb_rsi_strategy  # noqa: F401
from indicators.price.rsi_obv_bb import rsi_obv_bb_strategy  # noqa: F401
from indicators.price.adx import adx_strategy  # noqa: F401
from indicators.price.cci_adx import cci_adx_strategy  # noqa: F401
from indicators.price.williams_r import wr_strategy  # noqa: F401
from indicators.price.vwsma import vwsma_strategy  # noqa: F401
from strategies.price.multi_factor_combo import multi_factor_combo  # noqa: F401
from indicators.price.macd import macd_signal  # noqa: F401
from indicators.price.atr_trend import atr_trend_signal  # noqa: F401

from indicators.volume.acc_dist_index import acc_dist_index, acc_dist_signal  # noqa: F401
from indicators.volume.on_balance_volume import on_balance_volume  # noqa: F401
from indicators.volume.chaikin_money_flow import chaikin_money_flow, chaikin_money_flow_signal  # noqa: F401
from indicators.volume.force_index import force_index, force_index_signal  # noqa: F401
from indicators.volume.ease_of_movement import ease_of_movement, ease_of_movement_signal  # noqa: F401
from indicators.volume.volume_price_trend import volume_price_trend, volume_price_trend_signal  # noqa: F401
from indicators.volume.volume_oscillator import volume_oscillator, volume_oscillator_signal  # noqa: F401
from indicators.volume.vwma import volume_weighted_moving_average, vwma_signal  # noqa: F401
from indicators.volume.negative_volume_index import negative_volume_index, negative_volume_index_signal  # noqa: F401
from indicators.volume.put_call_ratio import put_call_ratio  # noqa: F401

__all__ = [
    'ma_cross_strategy',
    'rsi_strategy',
    'bb_rsi_strategy',
    'rsi_obv_bb_strategy',
    'adx_strategy',
    'cci_adx_strategy',
    'wr_strategy',
    'vwsma_strategy',
    'multi_factor_combo',
    'macd_signal',
    'atr_trend_signal',
    'acc_dist_index',
    'acc_dist_signal',
    'on_balance_volume',
    'chaikin_money_flow',
    'chaikin_money_flow_signal',
    'force_index',
    'force_index_signal',
    'ease_of_movement',
    'ease_of_movement_signal',
    'volume_price_trend',
    'volume_price_trend_signal',
    'volume_oscillator',
    'volume_oscillator_signal',
    'volume_weighted_moving_average',
    'vwma_signal',
    'negative_volume_index',
    'negative_volume_index_signal',
    'put_call_ratio',
]
