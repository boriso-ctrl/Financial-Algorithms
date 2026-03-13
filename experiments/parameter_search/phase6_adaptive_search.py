"""Phase 6.5 Bayesian Optimization: Find optimal importance weights with adaptive scaling.

Searches over 15 importance weighting configurations using Gaussian Process
Bayesian optimization to maximize Sharpe ratio, with optional exponential
signal scaling to favor strong signals.

Typical runtime: 200-500 iterations with 8-worker backtest parallelization
Expected improvement: 1.65 (Phase 3 static) → 1.8+ (Phase 6.5 adaptive)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Import from scripts module
sys.path.insert(0, str(Path(__file__).parent))
from search_combos import build_component_signals

from financial_algorithms.data import load_daily_prices
from financial_algorithms.backtest import run_backtest
from financial_algorithms.backtest.adaptive_blender import blend_signals_adaptive


# 15 indicators from Phase 3 champion
PHASE3_INDICATORS = [
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
    "cmf",
    "force_index",
]

# Phase 3 static weights (baseline)
PHASE3_WEIGHTS = {
    "ma_cross": 1.0,
    "sar_stoch": 1.0,
    "stoch_macd": 1.0,
    "rsi": 1.0,
    "bb_rsi": 1.0,
    "rsi_obv_bb": 1.0,
    "adx": 1.0,
    "cci_adx": 1.0,
    "williams_r": 1.0,
    "vwsma": 1.0,
    "macd": 1.0,
    "atr_trend": 1.0,
    "volume_osc": 1.0,
    "cmf": 1.0,
    "force_index": 1.0,
}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Phase6AdaptiveSearcher:
    """Bayesian optimization for importance weights and exponential bases."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        comp_signals: Dict[str, pd.DataFrame],
        indicators: List[str] = None,
        use_exponential_scaling: bool = True,
        global_exponential_base: float = 2.3,
        cost_bps: float = 5.0,
        slippage_bps: float = 5.0,
        allow_short: bool = True,
        risk_free_rate: float = 0.02,
        max_signal_abs: float = 5.0,
    ):
        """Initialize searcher.
        
        Args:
            prices: Historical price DataFrame
            comp_signals: Pre-computed component signals
            indicators: List of indicator names to weight (default: PHASE3_INDICATORS)
            use_exponential_scaling: Whether to enable adaptive exponential scaling
            global_exponential_base: Default exponential base for scaling
            cost_bps, slippage_bps: Trading costs
            allow_short: Allow short positions
            risk_free_rate: For Sharpe calculation
            max_signal_abs: Signal clipping bound
        """
        self.prices = prices
        self.comp_signals = comp_signals
        self.indicators = indicators or PHASE3_INDICATORS
        self.use_exponential_scaling = use_exponential_scaling
        self.global_exponential_base = global_exponential_base
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps
        self.allow_short = allow_short
        self.risk_free_rate = risk_free_rate
        self.max_signal_abs = max_signal_abs
        
        # Store results
        self.history: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.best_sharpe: float = float('-inf')
    
    def _evaluate_weights(
        self,
        weight_values: List[float],
    ) -> float:
        """Evaluate a single weight configuration.
        
        Args:
            weight_values: List of 15 weights in [0.1, 3.0]
        
        Returns:
            Negative Sharpe ratio (for minimization)
        """
        try:
            # Build weight dict
            weights = {ind: w for ind, w in zip(self.indicators, weight_values)}
            
            # Filter inactive
            active_weights = {k: v for k, v in weights.items() if abs(v) > 1e-9}
            active_signals = {k: self.comp_signals[k] for k in active_weights.keys()}
            
            if not active_signals:
                return 1e6  # Penalty for all-zero
            
            # Blend with adaptive scaling
            blended = blend_signals_adaptive(
                active_signals,
                importance_weights=active_weights,
                global_exponential_base=self.global_exponential_base,
                use_exponential_scaling=self.use_exponential_scaling,
                max_signal_abs=self.max_signal_abs,
            )
            
            # Backtest
            results = run_backtest(
                self.prices,
                blended,
                initial_capital=100_000.0,
                cost_bps=self.cost_bps,
                slippage_bps=self.slippage_bps,
                allow_short=self.allow_short,
                risk_free_rate=self.risk_free_rate,
                max_signal_abs=self.max_signal_abs,
            )
            
            # Extract Sharpe from metrics dict (formatted as string)
            metrics = results['metrics']
            sharpe_str = metrics.get('Sharpe Ratio', '0.0')
            try:
                sharpe = float(sharpe_str)
            except (ValueError, TypeError):
                sharpe = 0.0
            
            # Parse other metrics
            ret_str = metrics.get('Total Return', '0%').strip('%')
            cumulative_return = float(ret_str) / 100 if ret_str else 0.0
            
            dd_str = metrics.get('Max Drawdown', '0%').strip('%')
            max_drawdown = float(dd_str) / 100 if dd_str else 0.0
            
            wr_str = metrics.get('Win Rate', '0%').strip('%')
            win_rate = float(wr_str) / 100 if wr_str else 0.0
            
            # Store in history
            record = {
                'weights': weights,
                'sharpe': sharpe,
                'cumulative_return': cumulative_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
            }
            self.history.append(record)
            
            # Track best
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_result = record
                logger.info(f"New best Sharpe: {sharpe:.3f}")
            
            return -sharpe  # Negative for minimization
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 1e6  # Penalty
    
    def search(
        self,
        n_calls: int = 200,
        n_initial_points: int = 10,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run Bayesian optimization search.
        
        Args:
            n_calls: Total function evaluations
            n_initial_points: Random initial points before GP kicks in
            seed: Random seed
        
        Returns:
            Best result found
        """
        logger.info(
            f"Starting Phase 6.5 Bayesian search: "
            f"{len(self.indicators)} indicators, "
            f"exponential_scaling={self.use_exponential_scaling}, "
            f"base={self.global_exponential_base}"
        )
        
        # Define search space: 15 importance weights in [0.1, 3.0]
        space = [Real(0.1, 3.0, name=f"w_{ind}") for ind in self.indicators]
        
        # Fit initial points: static weights + random variations
        initial_points = self._generate_initial_points(n_initial_points)
        initial_values = []
        
        logger.info(f"Evaluating {len(initial_points)} initial points...")
        for i, point in enumerate(initial_points):
            val = self._evaluate_weights(point)
            initial_values.append(val)
            logger.info(f"  [{i+1}/{len(initial_points)}] x0 → {-val:.3f}")
        
        # Bayesian optimization
        logger.info("Starting Gaussian Process Bayesian optimization...")
        start_time = time.time()
        
        @use_named_args(space)
        def objective(**params):
            weight_list = [params[f"w_{ind}"] for ind in self.indicators]
            return self._evaluate_weights(weight_list)
        
        result = gp_minimize(
            objective,
            space,
            x0=initial_points,
            y0=initial_values,
            n_calls=n_calls - len(initial_points),
            n_initial_points=0,  # We provided x0, y0 already
            acq_func='EI',
            n_jobs=1,  # Single-threaded; backtest is already parallelized
            random_state=seed,
            verbose=1,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.1f}s")
        
        return self.best_result or self.history[-1]
    
    def _generate_initial_points(self, n_points: int) -> List[List[float]]:
        """Generate initial point set for Bayesian optimization.
        
        Includes: Phase 3 static weights + random variations
        """
        np.random.seed(42)
        points = []
        
        # Phase 3 static (all 1.0)
        points.append([1.0] * len(self.indicators))
        
        # Phase 3 with noise
        for _ in range(n_points // 2):
            noisy = [1.0 + np.random.normal(0, 0.3) for _ in self.indicators]
            noisy = [max(0.1, min(3.0, w)) for w in noisy]
            points.append(noisy)
        
        # Random weights
        for _ in range(max(1, n_points - len(points))):
            random_weights = np.random.uniform(0.1, 3.0, len(self.indicators)).tolist()
            points.append(random_weights)
        
        return points[:n_points]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6.5: Bayesian optimization for adaptive importance weights"
    )
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "AMZN"])
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--n-calls", type=int, default=200)
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-scaling", action="store_true", default=True)
    parser.add_argument("--exponential-base", type=float, default=2.3)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--report-json", type=str)
    
    args = parser.parse_args()
    
    logger.info(f"Phase 6.5 Adaptive Weight Search")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Period: {args.years} years")
    logger.info(f"Exponential base: {args.exponential_base}")
    logger.info(f"Evaluations: {args.n_calls}")
    
    # Load data
    logger.info("Loading prices...")
    prices = load_daily_prices(args.tickers)
    # Filter to specified years
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=args.years*365)
    prices = prices[prices.index >= cutoff]
    
    # Build signals
    logger.info("Building component signals...")
    comp_signals = build_component_signals(prices, PHASE3_INDICATORS)
    
    # Bayesian search
    searcher = Phase6AdaptiveSearcher(
        prices=prices,
        comp_signals=comp_signals,
        indicators=PHASE3_INDICATORS,
        use_exponential_scaling=args.use_scaling,
        global_exponential_base=args.exponential_base,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )
    
    best_result = searcher.search(
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        seed=args.seed,
    )
    
    # Report results
    logger.info("\n" + "="*70)
    logger.info("PHASE 6.5 RESULTS")
    logger.info("="*70)
    logger.info(f"Best Sharpe: {best_result['sharpe']:.3f}")
    logger.info(f"Cumulative Return: {best_result['cumulative_return']:.1%}")
    logger.info(f"Max Drawdown: {best_result['max_drawdown']:.1%}")
    logger.info(f"Win Rate: {best_result['win_rate']:.1%}")
    logger.info("\nBest Importance Weights:")
    for ind, w in best_result['weights'].items():
        logger.info(f"  {ind:20s}: {w:.3f}")
    
    # Comparison to Phase 3
    baseline_sharpe = 1.65
    improvement_pct = ((best_result['sharpe'] - baseline_sharpe) / abs(baseline_sharpe)) * 100
    logger.info(f"\nPhase 3 Baseline Sharpe: {baseline_sharpe:.3f}")
    logger.info(f"Improvement: {improvement_pct:+.1f}%")
    
    # Save results
    if args.report_json:
        report = {
            'metadata': {
                'search_type': 'Bayesian (GP)',
                'tickers': args.tickers,
                'years': args.years,
                'exponential_base': args.exponential_base,
                'use_scaling': args.use_scaling,
                'n_calls': args.n_calls,
            },
            'best': best_result,
            'history': searcher.history,
            'improvement_vs_phase3_pct': improvement_pct,
        }
        
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
