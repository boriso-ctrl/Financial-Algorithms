"""
7-Indicator Synergy Analysis

This script systematically tests combinations of up to 7 indicators
to find the optimal synergy that maximizes Sharpe ratio.

Strategy: Build from best performers and add complementary indicators
"""

from optimize_strategy import StrategyOptimizer, IndicatorGenerator
from data_loader_synthetic import generate_synthetic_ohlcv
import json
import itertools

print("="*80)
print("7-INDICATOR SYNERGY ANALYSIS")
print("="*80)
print("\nSystematically testing combinations up to 7 indicators")
print("Goal: Find optimal synergy to maximize Sharpe ratio\n")

# Configuration
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
seed = 1234
initial_capital = 100000

print(f"Configuration:")
print(f"  Tickers: {', '.join(tickers)}")
print(f"  Seed: {seed}")
print(f"  Initial Capital: ${initial_capital:,}")
print(f"  Data: 3 years synthetic OHLCV (seed {seed})")

# Load data
print("\nLoading data...")
ohlcv = generate_synthetic_ohlcv(tickers, days=756, seed=seed)
high = ohlcv['high']
low = ohlcv['low']
close = ohlcv['close']
volume = ohlcv['volume']

optimizer = StrategyOptimizer(tickers, initial_capital=initial_capital, target_sharpe=3.0)
optimizer.ohlcv_data = ohlcv
optimizer.prices = close

# Generate all indicator signals
print("\nGenerating indicator signals...")
all_indicators = {
    'VWAP': IndicatorGenerator.vwap_signal(high, low, close, volume),
    'ATR': IndicatorGenerator.atr_signal(high, low, close, 10),
    'OBV': IndicatorGenerator.obv_signal(close, volume, 10),
    'TRIX': IndicatorGenerator.trix_signal(close, 15),
    'MACD': IndicatorGenerator.macd_signal(close, 19, 39, 9),
    'EMA': IndicatorGenerator.ema_crossover(close, 5, 13),
    'SMA': IndicatorGenerator.sma_crossover(close, 20, 50),
    'Aroon': IndicatorGenerator.aroon_signal(high, low, 25),
    'ADX': IndicatorGenerator.adx_signal(high, low, close, 14, 25),
    'MFI': IndicatorGenerator.mfi_signal(high, low, close, volume, 14, 80, 20),
}

print(f"Generated {len(all_indicators)} indicator signals")

# Track all results
all_results = []

# Phase 1: Verify best 2-indicator combination (baseline)
print("\n" + "="*80)
print("PHASE 1: Baseline - Best 2-Indicator Combination")
print("="*80)

combo_2 = ['VWAP', 'ATR']
signals_2 = [all_indicators[ind] for ind in combo_2]
combined_2 = optimizer.combine_signals(signals_2, [0.5, 0.5])
result_2 = optimizer.test_configuration("_".join(combo_2), combined_2)

if result_2:
    print(f"\n{combo_2}: Sharpe={result_2['sharpe_ratio']:.3f}")
    all_results.append((combo_2, result_2, len(combo_2)))

# Phase 2: Test 3-indicator combinations (strategic)
print("\n" + "="*80)
print("PHASE 2: 3-Indicator Combinations (Strategic Selection)")
print("="*80)
print("Strategy: Build on VWAP+ATR, add complementary indicators\n")

# Start with VWAP+ATR (best 2), add third indicator
base = ['VWAP', 'ATR']
candidates_3 = ['OBV', 'TRIX', 'MACD', 'EMA', 'Aroon', 'MFI']

for third in candidates_3:
    combo = base + [third]
    signals = [all_indicators[ind] for ind in combo]
    combined = optimizer.combine_signals(signals, [0.33, 0.33, 0.34])
    result = optimizer.test_configuration("_".join(combo), combined)
    
    if result:
        print(f"{combo}: Sharpe={result['sharpe_ratio']:.3f}")
        all_results.append((combo, result, len(combo)))

# Find best 3-indicator combo
best_3 = max([r for r in all_results if r[2] == 3], key=lambda x: x[1]['sharpe_ratio'])
print(f"\n✓ Best 3-indicator: {best_3[0]} - Sharpe={best_3[1]['sharpe_ratio']:.3f}")

# Phase 3: Test 4-indicator combinations
print("\n" + "="*80)
print("PHASE 3: 4-Indicator Combinations")
print("="*80)
print(f"Strategy: Build on best 3-indicator ({best_3[0]}), add fourth\n")

base_3 = best_3[0]
remaining = [ind for ind in all_indicators.keys() if ind not in base_3]

for fourth in remaining[:6]:  # Test top 6 candidates
    combo = base_3 + [fourth]
    signals = [all_indicators[ind] for ind in combo]
    combined = optimizer.combine_signals(signals, [0.25, 0.25, 0.25, 0.25])
    result = optimizer.test_configuration("_".join(combo), combined)
    
    if result:
        print(f"{combo}: Sharpe={result['sharpe_ratio']:.3f}")
        all_results.append((combo, result, len(combo)))

# Find best 4-indicator combo
best_4_candidates = [r for r in all_results if r[2] == 4]
if best_4_candidates:
    best_4 = max(best_4_candidates, key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n✓ Best 4-indicator: {best_4[0]} - Sharpe={best_4[1]['sharpe_ratio']:.3f}")
else:
    best_4 = best_3
    print(f"\n✗ No 4-indicator improvement, staying with 3-indicator")

# Phase 4: Test 5-indicator combinations
print("\n" + "="*80)
print("PHASE 4: 5-Indicator Combinations")
print("="*80)

if best_4[2] == 4:
    print(f"Strategy: Build on best 4-indicator ({best_4[0]}), add fifth\n")
    base_4 = best_4[0]
    remaining = [ind for ind in all_indicators.keys() if ind not in base_4]
    
    for fifth in remaining[:5]:  # Test top 5 candidates
        combo = base_4 + [fifth]
        signals = [all_indicators[ind] for ind in combo]
        combined = optimizer.combine_signals(signals, [0.2, 0.2, 0.2, 0.2, 0.2])
        result = optimizer.test_configuration("_".join(combo), combined)
        
        if result:
            print(f"{combo}: Sharpe={result['sharpe_ratio']:.3f}")
            all_results.append((combo, result, len(combo)))

best_5_candidates = [r for r in all_results if r[2] == 5]
if best_5_candidates:
    best_5 = max(best_5_candidates, key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n✓ Best 5-indicator: {best_5[0]} - Sharpe={best_5[1]['sharpe_ratio']:.3f}")
else:
    best_5 = best_4
    print(f"\n✗ No 5-indicator improvement, staying with {best_4[2]}-indicator")

# Phase 5: Test 6-indicator combinations
print("\n" + "="*80)
print("PHASE 5: 6-Indicator Combinations")
print("="*80)

if best_5[2] == 5:
    print(f"Strategy: Build on best 5-indicator ({best_5[0]}), add sixth\n")
    base_5 = best_5[0]
    remaining = [ind for ind in all_indicators.keys() if ind not in base_5]
    
    for sixth in remaining[:4]:  # Test top 4 candidates
        combo = base_5 + [sixth]
        signals = [all_indicators[ind] for ind in combo]
        weights = [1.0/len(combo)] * len(combo)
        combined = optimizer.combine_signals(signals, weights)
        result = optimizer.test_configuration("_".join(combo), combined)
        
        if result:
            print(f"{combo}: Sharpe={result['sharpe_ratio']:.3f}")
            all_results.append((combo, result, len(combo)))

best_6_candidates = [r for r in all_results if r[2] == 6]
if best_6_candidates:
    best_6 = max(best_6_candidates, key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n✓ Best 6-indicator: {best_6[0]} - Sharpe={best_6[1]['sharpe_ratio']:.3f}")
else:
    best_6 = best_5
    print(f"\n✗ No 6-indicator improvement, staying with {best_5[2]}-indicator")

# Phase 6: Test 7-indicator combinations
print("\n" + "="*80)
print("PHASE 6: 7-Indicator Combinations")
print("="*80)

if best_6[2] == 6:
    print(f"Strategy: Build on best 6-indicator ({best_6[0]}), add seventh\n")
    base_6 = best_6[0]
    remaining = [ind for ind in all_indicators.keys() if ind not in base_6]
    
    for seventh in remaining[:3]:  # Test top 3 candidates
        combo = base_6 + [seventh]
        signals = [all_indicators[ind] for ind in combo]
        weights = [1.0/len(combo)] * len(combo)
        combined = optimizer.combine_signals(signals, weights)
        result = optimizer.test_configuration("_".join(combo), combined)
        
        if result:
            print(f"{combo}: Sharpe={result['sharpe_ratio']:.3f}")
            all_results.append((combo, result, len(combo)))

best_7_candidates = [r for r in all_results if r[2] == 7]
if best_7_candidates:
    best_7 = max(best_7_candidates, key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n✓ Best 7-indicator: {best_7[0]} - Sharpe={best_7[1]['sharpe_ratio']:.3f}")
else:
    best_7 = best_6
    print(f"\n✗ No 7-indicator improvement, staying with {best_6[2]}-indicator")

# Find overall best
print("\n" + "="*80)
print("RESULTS SUMMARY - ALL COMBINATIONS")
print("="*80)

# Sort all results by Sharpe ratio
all_results.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)

print("\n┌─────────────────────────────────────────────┬──────────┬──────────┬──────────┐")
print("│ Strategy                                    │  Sharpe  │  Return  │ Drawdown │")
print("├─────────────────────────────────────────────┼──────────┼──────────┼──────────┤")

for combo, result, n_indicators in all_results[:15]:  # Top 15
    name = "_".join(combo)
    name_display = name[:43] + "..." if len(name) > 43 else name
    sharpe = result['metrics']['Sharpe Ratio']
    ret = result['metrics']['Total Return']
    dd = result['metrics']['Max Drawdown']
    
    print(f"│ {name_display:43s} │ {sharpe:>8s} │ {ret:>8s} │ {dd:>8s} │")

print("└─────────────────────────────────────────────┴──────────┴──────────┴──────────┘")

# Best by indicator count
print("\n" + "="*80)
print("BEST STRATEGY BY INDICATOR COUNT")
print("="*80)

for n in range(2, 8):
    candidates = [r for r in all_results if r[2] == n]
    if candidates:
        best = candidates[0]  # Already sorted by Sharpe
        print(f"\n{n} Indicators: {best[0]}")
        print(f"  Sharpe: {best[1]['sharpe_ratio']:.3f}")
        print(f"  Return: {best[1]['metrics']['Total Return']}")
        print(f"  Drawdown: {best[1]['metrics']['Max Drawdown']}")

# Overall winner
print("\n" + "="*80)
print("🏆 OVERALL WINNER - MAXIMUM SHARPE RATIO")
print("="*80)

winner = all_results[0]
print(f"\nStrategy: {winner[0]}")
print(f"Number of Indicators: {winner[2]}")
print(f"\nPerformance Metrics:")
print(f"  Sharpe Ratio:   {winner[1]['metrics']['Sharpe Ratio']}")
print(f"  Total Return:   {winner[1]['metrics']['Total Return']}")
print(f"  CAGR:           {winner[1]['metrics']['CAGR']}")
print(f"  Max Drawdown:   {winner[1]['metrics']['Max Drawdown']}")
print(f"  Win Rate:       {winner[1]['metrics']['Win Rate']}")
print(f"  Final Equity:   {winner[1]['metrics']['Final Equity']}")

print("\n💡 Key Insights:")
print(f"  • Tested {len(all_results)} different combinations")
print(f"  • Best combination uses {winner[2]} indicators")
print(f"  • Achieves Sharpe ratio of {winner[1]['sharpe_ratio']:.3f}")

# Diminishing returns analysis
print("\n📊 Diminishing Returns Analysis:")
for n in range(2, 8):
    candidates = [r for r in all_results if r[2] == n]
    if candidates:
        best_n = max(candidates, key=lambda x: x[1]['sharpe_ratio'])
        print(f"  {n} indicators: Best Sharpe = {best_n[1]['sharpe_ratio']:.3f}")

# Compare to benchmarks
baseline_2 = next((r for r in all_results if r[0] == ['VWAP', 'ATR']), None)
if baseline_2 and winner[0] != ['VWAP', 'ATR']:
    improvement = ((winner[1]['sharpe_ratio'] - baseline_2[1]['sharpe_ratio']) / 
                   baseline_2[1]['sharpe_ratio']) * 100
    print(f"\n📈 Improvement over VWAP+ATR baseline:")
    print(f"  From: {baseline_2[1]['sharpe_ratio']:.3f} (2 indicators)")
    print(f"  To:   {winner[1]['sharpe_ratio']:.3f} ({winner[2]} indicators)")
    print(f"  Gain: {improvement:+.2f}%")

# Save results
output = {
    "timestamp": "2026-01-02",
    "seed": seed,
    "tickers": tickers,
    "total_combinations_tested": len(all_results),
    "winner": {
        "indicators": winner[0],
        "count": winner[2],
        "sharpe": winner[1]['sharpe_ratio'],
        "metrics": winner[1]['metrics']
    },
    "best_by_count": {
        str(n): {
            "indicators": next((r[0] for r in all_results if r[2] == n), None),
            "sharpe": next((r[1]['sharpe_ratio'] for r in all_results if r[2] == n), None)
        }
        for n in range(2, 8)
    },
    "top_10_strategies": [
        {
            "rank": i+1,
            "indicators": combo,
            "count": n_indicators,
            "sharpe": result['sharpe_ratio'],
            "metrics": result['metrics']
        }
        for i, (combo, result, n_indicators) in enumerate(all_results[:10])
    ]
}

with open('7_indicator_synergy_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ Results saved to: 7_indicator_synergy_results.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
