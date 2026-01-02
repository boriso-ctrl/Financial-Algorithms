"""Grid/Monte Carlo search over indicator weight blends to maximize Sharpe.

Runs combinations of BBRSI, SAR-Stoch, VWSMA, Williams %R, Volume Oscillator,
OBV, VPT, and a placeholder Put/Call ratio. Signals are blended with provided
weights, then fed to the backtester. Results are ranked by Sharpe (risk-free
included).

Example:
    python scripts/search_combos.py --tickers AAPL MSFT AMZN --years 3 --max-combos 500

Use --max-combos to cap runtime; the grid size is len(weight-grid) ** num_components.
If the grid is larger than max-combos, a random subset is sampled.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from data_loader import load_daily_prices
from backtest.simple_backtest import run_backtest
from backtest.signal_blender import blend_signals
from indicators.price.ma_cross import ma_cross_strategy
from indicators.price.sar_stoch import sar_stoch_strategy
from indicators.price.stoch_macd import stoch_macd_strategy
from indicators.price.rsi import rsi_strategy
from indicators.price.bb_rsi import bb_rsi_strategy
from indicators.price.rsi_obv_bb import rsi_obv_bb_strategy
from indicators.price.adx import adx_strategy
from indicators.price.cci_adx import cci_adx_strategy
from indicators.price.williams_r import wr_strategy
from indicators.price.vwsma import vwsma_strategy
from indicators.price.macd import macd_signal
from indicators.price.atr_trend import atr_trend_signal
from indicators.volume.volume_oscillator import volume_oscillator_signal
from indicators.volume.on_balance_volume import on_balance_volume
from indicators.volume.volume_price_trend import volume_price_trend_signal
from indicators.volume.acc_dist_index import acc_dist_signal
from indicators.volume.chaikin_money_flow import chaikin_money_flow_signal
from indicators.volume.force_index import force_index_signal
from indicators.volume.ease_of_movement import ease_of_movement_signal
from indicators.volume.vwma import vwma_signal
from indicators.volume.negative_volume_index import negative_volume_index_signal
from indicators.volume.put_call_ratio import put_call_ratio

# Placeholder PCR: returns zeros; kept so weight search can disable/enable later when data exists.
def _put_call_ratio_placeholder(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, name="pcr_placeholder")


DEFAULT_COMPONENTS = [
    "bbrsi",
    "sar_stoch",
    "vwsma",
    "williams_r",
    "volume_osc",
    "obv",
    "vpt",
    "put_call_ratio",
]

ALL_COMPONENTS = [
    "ma_cross",
    "sar_stoch",
    "stoch_macd",
    "rsi",
    "bb_rsi",
    "rsi_obv_bb",
    "adx",
    "cci_adx",
    "williams_r",
    "vwsma",
    "macd",
    "atr_trend",
    "volume_osc",
    "obv",
    "vpt",
    "acc_dist",
    "cmf",
    "force_index",
    "eom",
    "vwma",
    "nvi",
    "put_call_ratio",
]


def _ensure_ohlcv(close_series: pd.Series) -> pd.DataFrame:
    """Fabricate minimal OHLCV/Trend from a Close series to satisfy indicators."""
    df = pd.DataFrame(index=close_series.index)
    df["Close"] = close_series
    df["High"] = close_series * 1.001
    df["Low"] = close_series * 0.999
    df["Volume"] = 1_000_000
    df["Trend"] = "Uptrend"
    return df


def _broadcast(series: pd.Series, tickers: List[str]) -> pd.DataFrame:
    df = pd.concat([series] * len(tickers), axis=1)
    df.columns = tickers
    return df


def build_component_signals(prices: pd.DataFrame, components: List[str]) -> Dict[str, pd.DataFrame]:
    """Compute per-component signals for selected components, broadcast to tickers."""
    tickers = list(prices.columns)
    signals: Dict[str, pd.DataFrame] = {}
    failures = set()

    def add_signal(name: str, series: pd.Series, ticker: str):
        if name not in signals:
            signals[name] = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
        signals[name][ticker] = series.reindex(prices.index).fillna(0)

    for ticker in tickers:
        close = prices[ticker].dropna()
        ohlcv = _ensure_ohlcv(close)
        high = ohlcv["High"]
        low = ohlcv["Low"]
        vol = ohlcv["Volume"]

        def safe_compute(name, fn):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - exploratory search
                if name not in failures:
                    print(f"  Skipping {name} for {ticker}: {exc}")
                    failures.add(name)
                return pd.Series(0.0, index=ohlcv.index, name=name)

        available = {
            "ma_cross": lambda: ma_cross_strategy(ohlcv.copy(), sl=0.0)["MA_signal"],
            "sar_stoch": lambda: sar_stoch_strategy(ohlcv.copy())["SS_signal"],
            "stoch_macd": lambda: stoch_macd_strategy(ohlcv.copy(), sl=0.0)["SMACD_signal"],
            "rsi": lambda: rsi_strategy(ohlcv.copy(), sl=0.0)["RSI_signal"],
            "bb_rsi": lambda: bb_rsi_strategy(ohlcv.copy(), sl=0.0)["BBRSI_signal"],
            "rsi_obv_bb": lambda: rsi_obv_bb_strategy(ohlcv.copy(), sl=0.0)["ROB_signal"],
            "adx": lambda: adx_strategy(ohlcv.copy(), sl=0.0)["ADX_signal"],
            "cci_adx": lambda: cci_adx_strategy(ohlcv.copy(), sl=0.0)["CDX_signal"],
            "williams_r": lambda: wr_strategy(ohlcv.copy(), sl=0.0)["WR_signal"],
            "vwsma": lambda: vwsma_strategy(ohlcv.copy(), sl=0.0)["VWSMA_signal"],
            "macd": lambda: macd_signal(close),
            "atr_trend": lambda: atr_trend_signal(close),
            "volume_osc": lambda: volume_oscillator_signal(vol),
            "obv": lambda: on_balance_volume(close, vol),
            "vpt": lambda: volume_price_trend_signal(close, vol),
            "acc_dist": lambda: acc_dist_signal(high, low, close, vol),
            "cmf": lambda: chaikin_money_flow_signal(high, low, close, vol),
            "force_index": lambda: force_index_signal(close, vol),
            "eom": lambda: ease_of_movement_signal(high, low, close, vol),
            "vwma": lambda: vwma_signal(close, vol),
            "nvi": lambda: negative_volume_index_signal(close, vol),
            "put_call_ratio": lambda: _put_call_ratio_placeholder(close.index) if put_call_ratio() is None else put_call_ratio(),
        }

        for name in components:
            if name not in available:
                continue
            series = safe_compute(name, available[name])
            add_signal(name, series, ticker)

    signals = {k: v.fillna(0) for k, v in signals.items()}
    return signals


def _active_count(mapping: Dict[str, float]) -> int:
    return sum(1 for v in mapping.values() if abs(v) > 1e-9)


def _sample_weight_sets(weight_grid: List[float], components: List[str], sample_size: int, method: str, seed: Optional[int], max_active: Optional[int]) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    grid = np.array(weight_grid)
    n = len(grid)
    d = len(components)
    samples = []

    def idx_to_weights(idx_arr: np.ndarray) -> Dict[str, float]:
        return {comp: float(grid[i]) for comp, i in zip(components, idx_arr.tolist())}

    if method == "random":
        indices = rng.integers(low=0, high=n, size=(sample_size, d))
    elif method == "lhs":
        # Latin Hypercube: stratify each dimension and shuffle per column
        base = (rng.random((sample_size, d)) + np.arange(sample_size)[:, None]) / sample_size
        indices = np.minimum((base * n).astype(int), n - 1)
        for j in range(d):
            rng.shuffle(indices[:, j])
    elif method == "sobol":
        try:
            from scipy.stats import qmc  # type: ignore

            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
            pts = sampler.random(sample_size)
            indices = np.minimum((pts * n).astype(int), n - 1)
        except Exception:
            # Fallback to random if scipy not available
            indices = rng.integers(low=0, high=n, size=(sample_size, d))
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    for idx_row in indices:
        mapping = idx_to_weights(idx_row)
        if all(abs(w) < 1e-9 for w in mapping.values()):
            continue
        if max_active is not None and _active_count(mapping) > max_active:
            continue
        samples.append(mapping)
    return samples


def generate_weight_sets(weight_grid: List[float], components: List[str], max_combos: int, max_active: Optional[int], sampling_method: Optional[str], sample_size: Optional[int], seed: Optional[int]) -> List[Dict[str, float]]:
    grid = list(weight_grid)
    all_combos = len(grid) ** len(components)

    if sampling_method:
        size = sample_size if sample_size is not None else max_combos
        return _sample_weight_sets(grid, components, size, sampling_method, seed, max_active)

    combos_iter = itertools.product(grid, repeat=len(components))

    if all_combos <= max_combos:
        combos = list(combos_iter)
    else:
        # Sample indices without building full list
        sample_indices = set(random.sample(range(all_combos), max_combos))
        combos = []
        for idx, weights in enumerate(combos_iter):
            if idx in sample_indices:
                combos.append(weights)
    weight_dicts: List[Dict[str, float]] = []
    for weights in combos:
        mapping = {comp: w for comp, w in zip(components, weights)}
        if all(abs(w) < 1e-9 for w in mapping.values()):
            continue  # skip all-zero
        if max_active is not None and _active_count(mapping) > max_active:
            continue  # enforce sparsity cap
        weight_dicts.append(mapping)
    return weight_dicts


def evaluate_combo(prices: pd.DataFrame, comp_signals: Dict[str, pd.DataFrame], weights: Dict[str, float], max_signal_abs: float, allow_short: bool, cost_bps: float, slippage_bps: float, risk_free_rate: float) -> Dict:
    active_signals = {k: v for k, v in comp_signals.items() if abs(weights.get(k, 0)) > 1e-9}
    active_weights = {k: weights[k] for k in active_signals.keys()}
    blended = blend_signals(active_signals, active_weights, max_signal_abs=max_signal_abs)

    results = run_backtest(
        prices,
        blended,
        initial_capital=100_000.0,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        allow_short=allow_short,
        risk_free_rate=risk_free_rate,
        max_signal_abs=max_signal_abs,
    )
    return {
        "weights": active_weights,
        "metrics": results["metrics"],
    }


def _score(res: Dict, sparsity_penalty: float) -> float:
    try:
        sharpe = float(res["metrics"].get("Sharpe Ratio", float("-inf")))
    except Exception:
        sharpe = float("-inf")
    active = len(res.get("weights", {}))
    return sharpe - sparsity_penalty * max(active - 1, 0)


def _run_stage(prices: pd.DataFrame, comp_signals: Dict[str, pd.DataFrame], weight_sets: List[Dict[str, float]], max_signal_abs: float, allow_short: bool, cost_bps: float, slippage_bps: float, risk_free_rate: float, sparsity_penalty: float) -> List[Dict]:
    results = []
    for idx, wmap in enumerate(weight_sets, 1):
        res = evaluate_combo(prices, comp_signals, wmap, max_signal_abs, allow_short, cost_bps, slippage_bps, risk_free_rate)
        res["_score"] = _score(res, sparsity_penalty)
        results.append(res)
        if idx % 50 == 0:
            print(f"  Tested {idx}/{len(weight_sets)} combos...")
    return results


def _neighbors(weight_grid: List[float], base_weights: Dict[str, float], components: List[str], radius: int = 1) -> List[Dict[str, float]]:
    grid = list(weight_grid)
    idx_map = {v: i for i, v in enumerate(grid)}
    neighbor_lists: List[List[float]] = []
    for comp in components:
        w = base_weights.get(comp, 0.0)
        if w not in idx_map:
            neighbor_lists.append([0.0])
            continue
        idx = idx_map[w]
        candidates = set()
        for delta in range(-radius, radius + 1):
            j = idx + delta
            if 0 <= j < len(grid):
                candidates.add(grid[j])
        neighbor_lists.append(sorted(candidates))

    combos = []
    for weights in itertools.product(*neighbor_lists):
        mapping = {comp: w for comp, w in zip(components, weights)}
        combos.append(mapping)
    return combos


def search(prices: pd.DataFrame, weight_grid: List[float], components: List[str], max_combos: int, allow_short: bool, cost_bps: float, slippage_bps: float, risk_free_rate: float, max_signal_abs: float, top_n: int, max_active: Optional[int], sparsity_penalty: float, refine_top_k: int, refine_max_combos: int, refine_radius: int, sampling_method: Optional[str], sample_size: Optional[int], seed: Optional[int]) -> List[Dict]:
    comp_signals_full = build_component_signals(prices, components)
    comp_signals = {k: v for k, v in comp_signals_full.items() if k in components}

    weight_sets = generate_weight_sets(weight_grid, components, max_combos, max_active, sampling_method, sample_size, seed)

    print(f"Stage 1: evaluating {len(weight_sets)} combos (cap {max_combos}{' sampled' if sampling_method else ''})...")
    results_stage1 = _run_stage(prices, comp_signals, weight_sets, max_signal_abs, allow_short, cost_bps, slippage_bps, risk_free_rate, sparsity_penalty)

    all_results = list(results_stage1)
    evaluated_keys = {tuple(wmap.get(c, 0.0) for c in components) for wmap in weight_sets}

    ranked_stage1 = sorted(results_stage1, key=lambda r: r.get("_score", float("-inf")), reverse=True)

    if refine_top_k > 0 and ranked_stage1:
        seeds = ranked_stage1[:refine_top_k]
        neighbor_maps = []
        for seed in seeds:
            neighbor_maps.extend(_neighbors(weight_grid, seed["weights"], components, radius=refine_radius))

        # Deduplicate and enforce limits/constraints
        uniq_neighbors = []
        for m in neighbor_maps:
            key = tuple(m.get(c, 0.0) for c in components)
            if key in evaluated_keys:
                continue
            if all(abs(v) < 1e-9 for v in m.values()):
                continue
            if max_active is not None and _active_count(m) > max_active:
                continue
            evaluated_keys.add(key)
            uniq_neighbors.append(m)
            if len(uniq_neighbors) >= refine_max_combos:
                break

        if uniq_neighbors:
            print(f"Stage 2 (refine): evaluating {len(uniq_neighbors)} neighbor combos around top-{refine_top_k}...")
            results_stage2 = _run_stage(prices, comp_signals, uniq_neighbors, max_signal_abs, allow_short, cost_bps, slippage_bps, risk_free_rate, sparsity_penalty)
            all_results.extend(results_stage2)

    # Rank by score (Sharpe minus sparsity penalty)
    ranked = sorted(all_results, key=lambda r: r.get("_score", float("-inf")), reverse=True)
    return ranked[:top_n]


def _fmt(v):
    try:
        return float(v)
    except Exception:
        return v


def parse_weight_grid(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Search indicator weight blends for best Sharpe.")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "AMZN"], help="Tickers to backtest")
    parser.add_argument("--years", type=int, default=3, help="Lookback years")
    parser.add_argument("--weight-grid", type=str, default="0,0.5,1,2", help="Comma-separated weights to grid over")
    parser.add_argument("--max-combos", type=int, default=500, help="Cap on combinations to evaluate (samples if exceeded)")
    parser.add_argument("--components", nargs="+", default=None, help="Optional subset of components to include")
    parser.add_argument("--max-active", type=int, default=None, help="Maximum non-zero weights to allow (sparsity cap)")
    parser.add_argument("--sparsity-penalty", type=float, default=0.0, help="Penalty per extra active weight when ranking")
    parser.add_argument("--sampling-method", type=str, default=None, choices=["random", "lhs", "sobol"], help="Optional sampling method instead of grid enumeration")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to draw when sampling-method is set")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--refine-top-k", type=int, default=0, help="If >0, take top-K from stage 1 and evaluate neighbor combos")
    parser.add_argument("--refine-max-combos", type=int, default=500, help="Cap on neighbor combos evaluated in stage 2")
    parser.add_argument("--refine-radius", type=int, default=1, help="Neighborhood radius (grid steps) for refinement")
    parser.add_argument("--top-n", type=int, default=10, help="How many top combos to show")
    parser.add_argument("--allow-short", action="store_true", help="Allow shorting")
    parser.add_argument("--cost-bps", type=float, default=1.0, help="Commission bps")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps")
    parser.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual risk-free rate")
    parser.add_argument("--max-signal-abs", type=float, default=5.0, help="Signal clip")
    parser.add_argument("--report-json", type=str, help="Optional path to write full top-N results as JSON")
    args = parser.parse_args()

    print("Loading prices...")
    prices = load_daily_prices(args.tickers)
    prices = prices.last(f"{args.years}Y") if args.years else prices

    weight_grid = parse_weight_grid(args.weight_grid)
    if args.components == ["all"]:
        components = list(ALL_COMPONENTS)
    else:
        components = args.components if args.components else list(DEFAULT_COMPONENTS)

    unknown = set(components) - set(ALL_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown component names: {sorted(unknown)}")

    print(f"Weight grid: {weight_grid}; components: {len(components)}")
    print(f"Max combos (stage1): {args.max_combos}; refine-top-k: {args.refine_top_k}; refine-max: {args.refine_max_combos}")
    if args.max_active:
        print(f"Sparsity cap: {args.max_active} active weights; penalty: {args.sparsity_penalty}")
    if args.sampling_method:
        print(f"Sampling: {args.sampling_method} with sample-size={args.sample_size or args.max_combos} seed={args.seed}")

    top = search(
        prices,
        weight_grid=weight_grid,
        components=components,
        max_combos=args.max_combos,
        allow_short=args.allow_short,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        risk_free_rate=args.risk_free_rate,
        max_signal_abs=args.max_signal_abs,
        top_n=args.top_n,
        max_active=args.max_active,
        sparsity_penalty=args.sparsity_penalty,
        refine_top_k=args.refine_top_k,
        refine_max_combos=args.refine_max_combos,
        refine_radius=args.refine_radius,
        sampling_method=args.sampling_method,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print("\nTop combos (by score: Sharpe minus sparsity_penalty * active_excess):")
    for i, res in enumerate(top, 1):
        m = res["metrics"]
        sharpe = _fmt(m.get('Sharpe Ratio'))
        sortino = _fmt(m.get('Sortino Ratio'))
        calmar = _fmt(m.get('Calmar Ratio'))
        total_ret = _fmt(m.get('Total Return'))
        max_dd = _fmt(m.get('Max Drawdown'))
        score = _fmt(res.get('_score'))
        print(f"[{i}] Score={score} Sharpe={sharpe} Sortino={sortino} Calmar={calmar} Return={total_ret} MaxDD={max_dd}")
        print(f"    Weights: {res['weights']}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(top, f, indent=2)
        print(f"Saved top-{len(top)} results to {args.report_json}")


if __name__ == "__main__":
    main()
