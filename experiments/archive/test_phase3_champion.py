"""Phase 5: Robustness Testing - Test Phase 3 Champion Across Regimes

Validates the Phase 3 champion portfolio (Sharpe 1.65, 15 indicators, AAPL/MSFT/AMZN, 3y)
across different universes and lookback windows.

Usage:
    python scripts/test_phase3_champion.py --universes aapl_msft_amzn top10 --lookbacks 1 2 3 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from financial_algorithms.data import load_daily_prices
from financial_algorithms.backtest import run_backtest, blend_signals

# Import all signal strategies from the working search_combos baseline
sys.path.insert(0, str(ROOT / "scripts"))
from search_combos import build_component_signals, evaluate_combo

# Phase 3 Champion Portfolio (Sharpe 1.65)
PHASE3_WEIGHTS = {
    "ma_cross": 1.0,
    "sar_stoch": 2.0,
    "stoch_macd": 1.5,
    "rsi": 1.0,
    "bb_rsi": 2.0,
    "rsi_obv_bb": 1.0,
    "adx": 1.5,
    "cci_adx": 1.5,
    "williams_r": 1.0,
    "vwsma": 1.5,
    "macd": 1.0,
    "atr_trend": 1.5,
    "volume_osc": 1.0,
    "cmf": 2.0,
    "force_index": 1.0,
}

# Test universes
UNIVERSES = {
    "aapl_msft_amzn": ["AAPL", "MSFT", "AMZN"],
    "top10": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "JNJ", "XOM", "WMT", "HD"],
    "tech": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
}

def test_phase3_on_universe(tickers: List[str], lookback_years: int) -> Dict:
    """Test Phase 3 weights on a specific universe."""
    try:
        print(f"    {tickers[:3]}... ({len(tickers)} tickers, {lookback_years}y)", end=" ... ")
        
        # Load prices
        prices = load_daily_prices(tickers)
        prices = prices.tail(lookback_years * 252)  # Approximate years to trading days
        
        if prices.empty or len(prices) < 100:
            print("❌ Insufficient data")
            return {"status": "FAILED", "reason": "Insufficient data"}
        
        # Use the working evaluate_combo function from search_combos
        result = evaluate_combo(
            prices, 
            build_component_signals(prices, list(PHASE3_WEIGHTS.keys())),
            PHASE3_WEIGHTS,
            max_signal_abs=5.0,
            allow_short=False,
            cost_bps=1.0,
            slippage_bps=5.0,
            risk_free_rate=0.0
        )
        
        metrics = result.get("metrics", {})
        sharpe = float(metrics.get("Sharpe Ratio", 0))
        
        print(f"✅ Sharpe={sharpe:.3f}")
        
        return {
            "status": "OK",
            "sharpe": sharpe,
            "sortino": float(metrics.get("Sortino Ratio", 0)),
            "calmar": float(metrics.get("Calmar Ratio", 0)),
            "return": float(metrics.get("Total Return (%)", 0)),
            "max_dd": float(metrics.get("Max Drawdown (%)", 0)),
        }
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return {"status": "FAILED", "reason": str(e)[:100]}

def main():
    parser = argparse.ArgumentParser(description="Test Phase 3 Champion Across Universes")
    parser.add_argument("--universes", nargs="+", default=["aapl_msft_amzn", "top10"],
                       help="Universes to test")
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[1, 2, 3, 5],
                       help="Lookback years")
    parser.add_argument("--output", type=str, default="data/search_results/phase5_results.json",
                       help="Output JSON path")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 5: Test Champion Portfolio Across Regimes")
    print("=" * 70)
    print(f"Portfolio: Sharpe 1.65 (Phase 3) | 15 indicators")
    print(f"Testing: {len(args.universes)} universes × {len(args.lookbacks)} lookback windows")
    print()
    
    results = {}
    
    for universe in args.universes:
        if universe not in UNIVERSES:
            print(f"⚠️  Unknown universe: {universe}")
            continue
        
        tickers = UNIVERSES[universe]
        print(f"  Universe: {universe} (n={len(tickers)})")
        
        results[universe] = {}
        for lookback in args.lookbacks:
            result = test_phase3_on_universe(tickers, lookback)
            results[universe][f"{lookback}y"] = result
        
        print()
    
    # Aggregate
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    for universe, lookback_results in results.items():
        ok = [r for r in lookback_results.values() if r["status"] == "OK"]
        if ok:
            sharpes = [r["sharpe"] for r in ok]
            print(f"  {universe:20s}: {len(ok)}/{len(lookback_results)} passed | "
                  f"Sharpe {min(sharpes):.3f}–{max(sharpes):.3f}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"✅ Results saved to: {output_path}")

if __name__ == "__main__":
    main()
