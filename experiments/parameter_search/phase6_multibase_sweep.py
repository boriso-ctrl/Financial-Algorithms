#!/usr/bin/env python3
"""Phase 6.5 Multi-Base Sweep: Find optimal exponential base for adaptive scaling.

Tests exponential bases {1.5, 1.8, 2.0, 2.3, 2.5, 3.0} with Phase 3 static weights
to determine which scaling curve is most effective.

Expected runtime: ~3-5 minutes for all 6 bases on AAPL/MSFT/AMZN (3y).
"""

import sys
from pathlib import Path
import logging
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from search_combos import build_component_signals

from financial_algorithms.data import load_daily_prices
from financial_algorithms.backtest import run_backtest
from financial_algorithms.backtest.adaptive_blender import blend_signals_adaptive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PHASE3_INDICATORS = [
    "ma_cross", "sar_stoch", "stoch_macd", "rsi", "bb_rsi",
    "rsi_obv_bb", "adx", "cci_adx", "williams_r", "vwsma",
    "macd", "atr_trend", "volume_osc", "cmf", "force_index",
]

PHASE3_WEIGHTS = {ind: 1.0 for ind in PHASE3_INDICATORS}

# Bases to test
BASES_TO_TEST = [1.5, 1.8, 2.0, 2.3, 2.5, 3.0]


def test_base(base, prices, comp_signals):
    """Test a single exponential base with Phase 3 weights."""
    logger.info(f"\nTesting exponential base: {base}")
    
    active_signals = {k: v for k, v in comp_signals.items() if k in PHASE3_INDICATORS}
    active_weights = PHASE3_WEIGHTS.copy()
    
    # Blend with adaptive scaling using this base
    blended = blend_signals_adaptive(
        active_signals,
        importance_weights=active_weights,
        global_exponential_base=base,
        use_exponential_scaling=True,
        max_signal_abs=5.0,
    )
    
    # Backtest
    results = run_backtest(
        prices, blended,
        initial_capital=100_000.0,
        cost_bps=5.0,
        slippage_bps=5.0,
        allow_short=True,
        risk_free_rate=0.02,
        max_signal_abs=5.0,
    )
    
    metrics = results['metrics']
    
    # Parse metrics
    sharpe_str = metrics.get('Sharpe Ratio', '0.0')
    sharpe = float(sharpe_str)
    
    ret_str = metrics.get('Total Return', '0%').strip('%')
    cumulative_return = float(ret_str) / 100 if ret_str else 0.0
    
    dd_str = metrics.get('Max Drawdown', '0%').strip('%')
    max_drawdown = float(dd_str) / 100 if dd_str else 0.0
    
    wr_str = metrics.get('Win Rate', '0%').strip('%')
    win_rate = float(wr_str) / 100 if wr_str else 0.0
    
    logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
    logger.info(f"  Cumulative Return: {cumulative_return:.1%}")
    logger.info(f"  Max Drawdown: {max_drawdown:.1%}")
    logger.info(f"  Win Rate: {win_rate:.1%}")
    
    return {
        'base': base,
        'sharpe': sharpe,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }


def main():
    logger.info("="*70)
    logger.info("PHASE 6.5: Multi-Base Exponential Sweep")
    logger.info("="*70)
    logger.info(f"Testing bases: {BASES_TO_TEST}")
    logger.info(f"Using Phase 3 weights: all 1.0")
    
    # Load data
    tickers = ["AAPL", "MSFT", "AMZN"]
    logger.info(f"\nLoading prices for {tickers} (3 years)...")
    prices = load_daily_prices(tickers)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=3*365)
    prices = prices[prices.index >= cutoff]
    
    # Build signals
    logger.info(f"Building {len(PHASE3_INDICATORS)} component signals...")
    comp_signals = build_component_signals(prices, PHASE3_INDICATORS)
    
    # Test each base
    results = []
    for base in BASES_TO_TEST:
        result = test_base(base, prices, comp_signals)
        results.append(result)
    
    # Find best
    best = max(results, key=lambda x: x['sharpe'])
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    logger.info(f"\n{'Base':<8} {'Sharpe':<10} {'Return':<12} {'Max DD':<12} {'Win Rate':<10}")
    logger.info("-"*70)
    
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        mark = "←BEST" if r['base'] == best['base'] else ""
        logger.info(
            f"{r['base']:<8.1f} {r['sharpe']:<10.3f} "
            f"{r['cumulative_return']:<12.1%} {r['max_drawdown']:<12.1%} "
            f"{r['win_rate']:<10.1%} {mark}"
        )
    
    logger.info("\n" + "="*70)
    logger.info("OPTIMAL BASE")
    logger.info("="*70)
    logger.info(f"Best base: {best['base']}")
    logger.info(f"Best Sharpe: {best['sharpe']:.3f}")
    logger.info(f"vs Phase 3 baseline (1.65): {((best['sharpe'] - 1.65) / 1.65 * 100):+.1f}%")
    
    # Save results
    output_file = Path("data/search_results/phase6_multibase_sweep.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test_type': 'exponential_base_sweep',
                'indicators': PHASE3_INDICATORS,
                'weights': 'all 1.0 (Phase 3 static)',
                'tickers': tickers,
                'bases_tested': BASES_TO_TEST,
            },
            'results': results,
            'best': best,
        }, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)
    
    if best['sharpe'] >= 1.65:
        logger.info(f"✓ Base {best['base']} IMPROVES Phase 3 (1.65 → {best['sharpe']:.3f})")
        logger.info(f"  Use this base in production Phase 6.5 searches.")
    else:
        logger.info(f"✗ Best base ({best['base']}) underperforms Phase 3 baseline")
        logger.info(f"  Phase 3 uniform weights (1.65 Sharpe) remain preferred.")
        logger.info(f"  Consider: Phase 3 as production, Phase 6.5 for future research.")


if __name__ == "__main__":
    main()
