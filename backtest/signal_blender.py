"""Utilities to blend multiple indicator signals with weights.

Signals are expected as DataFrames aligned on index/columns. The output is a
weighted sum, clipped to a maximum absolute conviction (default ±5).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def blend_signals(
    signal_map: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    """Blend multiple signal DataFrames using provided weights.

    Args:
        signal_map: name -> signal DataFrame (index=dates, cols=tickers)
        weights: name -> weight multiplier (e.g., 2.0 for core, 0.5 for minor)
        max_signal_abs: clip final signal magnitude to this absolute value

    Returns:
        DataFrame of weighted signals aligned on the common index/columns.
    """
    if not signal_map:
        raise ValueError("signal_map is empty")

    # Determine common index/columns across all signals
    indices = [df.index for df in signal_map.values()]
    cols = [df.columns for df in signal_map.values()]
    common_index = indices[0]
    common_cols = cols[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)
    for c in cols[1:]:
        common_cols = common_cols.intersection(c)

    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError("No common dates or tickers across signals")

    # Reindex all signals to the common axes
    aligned = {}
    for name, df in signal_map.items():
        aligned[name] = df.reindex(index=common_index, columns=common_cols).fillna(0)

    # Build weighted sum
    total = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    for name, df in aligned.items():
        w = weights.get(name, 1.0)
        total += df * w

    # Normalize by sum of absolute weights to keep scale consistent
    denom = sum(abs(w) for w in weights.values()) or 1.0
    blended = total / denom

    # Clip to conviction cap
    blended = blended.clip(lower=-max_signal_abs, upper=max_signal_abs)
    return blended
