#!/usr/bin/env python3
import json

data = json.load(open('data/search_results/phase6_bayesian_adaptive_fixed.json'))
best = data['best']

print('='*70)
print('PHASE 6.5 BAYESIAN SEARCH - BEST RESULT')
print('='*70)
print(f'Best Sharpe: {best["sharpe"]:.3f}')
print(f'Cumulative Return: {best["cumulative_return"]:.1%}')
print(f'Max Drawdown: {best["max_drawdown"]:.1%}')
print(f'Win Rate: {best["win_rate"]:.1%}')
print()
print('Top 5 Highest Importance Weights:')
sorted_weights = sorted(best['weights'].items(), key=lambda x: x[1], reverse=True)
for name, weight in sorted_weights[:5]:
    print(f'  {name:20s}: {weight:.3f}')

print()
print('Bottom 5 Lowest Importance Weights:')
for name, weight in sorted_weights[-5:]:
    print(f'  {name:20s}: {weight:.3f}')

improvement = ((best['sharpe'] - 1.65) / 1.65) * 100
print()
print(f'Improvement vs Phase 3 (1.65): {improvement:+.1f}%')

print()
print('Evals completed:', len(data['history']))
max_sharpe_evals = max([x['sharpe'] for x in data['history']])
print(f'Max Sharpe seen: {max_sharpe_evals:.3f}')
