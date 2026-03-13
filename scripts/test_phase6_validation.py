#!/usr/bin/env python3
"""Quick test: Static vs Adaptive blending on Phase 3 champion.

Compares:
1. Static blend (Phase 3, all weights 1.0)
2. Adaptive blend with exponential scaling (Phase 6.5, all weights 1.0, base 2.3)

This validates that adaptive infrastructure works and quantifies baseline benefit.
"""

import sys
from pathlib import Path
import logging

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from search_combos import build_component_signals

from financial_algorithms.data import load_daily_prices
from financial_algorithms.backtest import run_backtest, blend_signals
from financial_algorithms.backtest.adaptive_blender import blend_signals_adaptive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHASE3_INDICATORS = [
    "ma_cross", "sar_stoch", "stoch_macd", "rsi", "bb_rsi",
    "rsi_obv_bb", "adx", "cci_adx", "williams_r", "vwsma",
    "macd", "atr_trend", "volume_osc", "cmf", "force_index",
]

PHASE3_WEIGHTS = {ind: 1.0 for ind in PHASE3_INDICATORS}


def main():
    logger.info("="*70)
    logger.info("PHASE 6.5 VALIDATION: Static vs Adaptive Blending")
    logger.info("="*70)
    
    # Load data
    tickers = ["AAPL", "MSFT", "AMZN"]
    logger.info(f"Loading prices for {tickers}...")
    prices = load_daily_prices(tickers)
    # Filter to last 3 years
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=3*365)
    prices = prices[prices.index >= cutoff]
    
    # Build signals
    logger.info(f"Building {len(PHASE3_INDICATORS)} component signals...")
    comp_signals = build_component_signals(prices, PHASE3_INDICATORS)
    
    # TEST 1: Static blend (Phase 3 baseline)
    logger.info("\n[1] STATIC BLEND (Phase 3 baseline)")
    logger.info(f"    Weights: all 1.0")
    logger.info(f"    Scaling: None")
    
    active_signals = {k: v for k, v in comp_signals.items() if k in PHASE3_INDICATORS}
    active_weights = PHASE3_WEIGHTS.copy()
    
    blended_static = blend_signals(active_signals, active_weights, max_signal_abs=5.0)
    results_static = run_backtest(
        prices, blended_static,
        initial_capital=100_000.0,
        cost_bps=5.0,
        slippage_bps=5.0,
        allow_short=True,
        risk_free_rate=0.02,
        max_signal_abs=5.0,
    )
    metrics_static = results_static['metrics']
    
    # Parse metrics
    sharpe_static = float(metrics_static['Sharpe Ratio'])
    ret_static = float(metrics_static['Total Return'].strip('%')) / 100
    dd_static = float(metrics_static['Max Drawdown'].strip('%')) / 100
    wr_static = float(metrics_static['Win Rate'].strip('%')) / 100
    
    logger.info(f"    Sharpe Ratio: {sharpe_static:.3f}")
    logger.info(f"    Cumulative Return: {ret_static:.1%}")
    logger.info(f"    Max Drawdown: {dd_static:.1%}")
    logger.info(f"    Win Rate: {wr_static:.1%}")
    
    # TEST 2: Adaptive blend with exponential scaling, same weights
    logger.info("\n[2] ADAPTIVE BLEND (Phase 6.5, no importance changes)")
    logger.info(f"    Weights: all 1.0 (same as Phase 3)")
    logger.info(f"    Scaling: Exponential, base=2.3")
    
    blended_adaptive = blend_signals_adaptive(
        active_signals,
        importance_weights=active_weights,
        global_exponential_base=2.3,
        use_exponential_scaling=True,
        max_signal_abs=5.0,
    )
    results_adaptive = run_backtest(
        prices, blended_adaptive,
        initial_capital=100_000.0,
        cost_bps=5.0,
        slippage_bps=5.0,
        allow_short=True,
        risk_free_rate=0.02,
        max_signal_abs=5.0,
    )
    metrics_adaptive = results_adaptive['metrics']
    
    # Parse metrics
    sharpe_adaptive = float(metrics_adaptive['Sharpe Ratio'])
    ret_adaptive = float(metrics_adaptive['Total Return'].strip('%')) / 100
    dd_adaptive = float(metrics_adaptive['Max Drawdown'].strip('%')) / 100
    wr_adaptive = float(metrics_adaptive['Win Rate'].strip('%')) / 100
    
    logger.info(f"    Sharpe Ratio: {sharpe_adaptive:.3f}")
    logger.info(f"    Cumulative Return: {ret_adaptive:.1%}")
    logger.info(f"    Max Drawdown: {dd_adaptive:.1%}")
    logger.info(f"    Win Rate: {wr_adaptive:.1%}")
    
    # COMPARISON
    logger.info("\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    
    sharpe_delta = sharpe_adaptive - sharpe_static
    sharpe_pct = (sharpe_delta / abs(sharpe_static)) * 100
    
    logger.info(f"Sharpe Delta: {sharpe_delta:+.3f} ({sharpe_pct:+.1f}%)")
    
    ret_delta = ret_adaptive - ret_static
    dd_delta = dd_adaptive - dd_static
    wr_delta = wr_adaptive - wr_static
    
    logger.info(f"Return Delta: {ret_delta:+.1%}")
    logger.info(f"Max Drawdown Delta: {dd_delta:+.1%}")
    logger.info(f"Win Rate Delta: {wr_delta:+.1%}")
    
    # Expected behavior: exponential scaling should preserve linear blending
    # if weights haven't changed; but may improve signal quality slightly
    logger.info("\n" + "="*70)
    logger.info("INTERPRETATION")
    logger.info("="*70)
    
    if abs(sharpe_delta) < 0.05:
        logger.info("✓ Adaptive with same weights ≈ Static (as expected)")
        logger.info("  Exponential scaling is stable and doesn't break linear blending.")
    elif sharpe_delta > 0.05:
        logger.info("✓ Adaptive slightly outperforms Static")
        logger.info("  Exponential scaling to strong signals provides modest benefit.")
    else:
        logger.info("⚠ Adaptive underperforms Static (investigate scaling function)")
    
    logger.info("\n" + "="*70)
    logger.info("Next: Run Phase 6.5 Bayesian search to optimize importance weights.")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
