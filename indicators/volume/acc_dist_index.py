"""Accumulation/Distribution Index (ADI)."""

import numpy as np
import pandas as pd


def acc_dist_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fillna: bool = False) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0)
    ad = clv * volume
    ad = ad + ad.shift(1, fill_value=ad.mean())
    if fillna:
        ad = ad.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ad, name='adi')
