"""
Multi-Asset Backtest Runner — V6, 10-Year Horizon
==================================================
Runs AggressiveHybridV6 across a diverse universe of 38+ assets.
Date range: 2015-01-01 -> 2025-12-31  (10 years, ~2500 daily bars)

Sequential execution — yfinance is not thread-safe.
"""

import sys
import os
import json
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Asset universe — 38 assets across stocks, commodities, crypto, int'l
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Broad equity ETFs
    'SPY':   'S&P 500',
    'QQQ':   'NASDAQ 100',
    'IWM':   'Russell 2000',
    'DIA':   'Dow Jones',
    'VTI':   'Total US Market',

    # Sector ETFs
    'XLK':   'Technology',
    'XLE':   'Energy',
    'XLF':   'Financials',
    'XLV':   'Health Care',
    'XBI':   'Biotech',

    # Large-cap stocks
    'AAPL':  'Apple',
    'MSFT':  'Microsoft',
    'NVDA':  'NVIDIA',
    'AMZN':  'Amazon',
    'GOOGL': 'Alphabet',
    'META':  'Meta',
    'TSLA':  'Tesla',
    'JPM':   'JPMorgan',
    'BRK-B': 'Berkshire',
    'UNH':   'UnitedHealth',

    # Commodities (ETF proxies)
    'GLD':   'Gold',
    'SLV':   'Silver',
    'USO':   'Oil (USO)',
    'UNG':   'Natural Gas',
    'CORN':  'Corn',
    'WEAT':  'Wheat',
    'PDBC':  'Diversified Commodities',

    # Fixed income / alternatives
    'TLT':   '20yr Treasury',
    'HYG':   'High Yield Bonds',
    'GDX':   'Gold Miners',

    # Crypto (via ETFs / trusts — limited history pre-2024)
    'IBIT':  'Bitcoin ETF (IBIT)',
    'FBTC':  'Bitcoin ETF (Fidelity)',
    'ETHA':  'Ethereum ETF',
    'GBTC':  'Bitcoin Trust (GBTC)',
    'MSTR':  'MicroStrategy (BTC proxy)',

    # International
    'EEM':   'Emerging Markets',
    'EWJ':   'Japan',
    'EWZ':   'Brazil',
}

START = '2015-01-01'
END   = '2025-12-31'


# ---------------------------------------------------------------------------
def run_one(symbol, label):
    try:
        trader = AggressiveHybridV6(ticker=symbol, start=START, end=END)
        if not trader.fetch_data():
            return {'symbol': symbol, 'label': label, 'error': 'fetch failed'}
        if len(trader.data) < 300:
            return {'symbol': symbol, 'label': label,
                    'error': f'insufficient data ({len(trader.data)} bars)'}
        results = trader.backtest()
        if results is None or 'error' in results:
            return {'symbol': symbol, 'label': label,
                    'error': results.get('error', 'unknown')}
        results['label'] = label
        return results
    except Exception as e:
        return {'symbol': symbol, 'label': label, 'error': str(e)}


def score(r):
    """Composite score: risk-adjusted return + raw CAGR bonus"""
    return r.get('sharpe', 0) + r.get('cagr', 0) / 30


def print_table(rows, title):
    cols = ['symbol', 'label', 'years', 'return_pct', 'cagr', 'sharpe',
            'sortino', 'calmar', 'win_rate', 'trades', 'max_dd', 'profit_factor']
    headers = {
        'symbol': 'Symbol', 'label': 'Asset', 'years': 'Yrs',
        'return_pct': 'Return%', 'cagr': 'CAGR%', 'sharpe': 'Sharpe',
        'sortino': 'Sortino', 'calmar': 'Calmar', 'win_rate': 'WR%',
        'trades': 'Trades', 'max_dd': 'MaxDD%', 'profit_factor': 'PF',
    }
    widths = {
        'symbol': 6, 'label': 22, 'years': 4, 'return_pct': 8, 'cagr': 7,
        'sharpe': 7, 'sortino': 7, 'calmar': 7, 'win_rate': 6,
        'trades': 7, 'max_dd': 7, 'profit_factor': 5,
    }
    sep  = '-' * (sum(widths.values()) + 3 * len(cols))
    hrow = '  '.join(f"{headers[c]:<{widths[c]}}" for c in cols)
    print(f'\n{"=" * len(sep)}')
    print(f'  {title}')
    print('=' * len(sep))
    print(hrow)
    print(sep)
    for r in rows:
        print(
            f"{r.get('symbol',''):<6}  "
            f"{r.get('label',''):<22}  "
            f"{r.get('years', 0):<4}  "
            f"{r.get('return_pct', 0):<8.1f}  "
            f"{r.get('cagr', 0):<7.1f}  "
            f"{r.get('sharpe', 0):<7.2f}  "
            f"{r.get('sortino', 0):<7.2f}  "
            f"{r.get('calmar_ratio', 0):<7.2f}  "
            f"{r.get('win_rate', 0):<6.1f}  "
            f"{r.get('trades', 0):<7}  "
            f"{r.get('max_dd', 0):<7.1f}  "
            f"{r.get('profit_factor', 0):<5.2f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\n{'=' * 72}")
    print(f"  MULTI-ASSET BACKTEST  |  V6 10yr Strategy  |  {START} -> {END}")
    print(f"  Universe: {len(UNIVERSE)} assets  |  Long-only  |  3-regime trend filter")
    print(f"{'=' * 72}\n")

    results_good = []
    results_fail = []
    total = len(UNIVERSE)

    for done, (sym, lbl) in enumerate(UNIVERSE.items(), 1):
        res = run_one(sym, lbl)
        if 'error' in res:
            results_fail.append(res)
            status = f"  [{done:>2}/{total}] FAIL  {sym:<6}  {res['error']}"
        else:
            results_good.append(res)
            status = (f"  [{done:>2}/{total}] OK    {sym:<6}  "
                      f"Sharpe {res['sharpe']:.2f}  CAGR {res['cagr']:.1f}%  "
                      f"WR {res['win_rate']:.0f}%  Trades {res['trades']}")
        print(status, flush=True)

    # -------------------------------------------------------------------------
    results_good.sort(key=score, reverse=True)

    print_table(results_good, 'ALL ASSETS -- ranked by Sharpe + CAGR/30')
    print_table(results_good[:10], 'TOP 10 PERFORMERS')
    if results_good:
        print_table(sorted(results_good, key=score)[:5], 'BOTTOM 5')

    # -------------------------------------------------------------------------
    if results_good:
        sharpes  = [r['sharpe'] for r in results_good]
        cagrs    = [r['cagr']   for r in results_good]
        dds      = [r['max_dd'] for r in results_good]
        profitable = sum(1 for r in results_good if r['cagr'] > 0)
        gt2_sharpe = sum(1 for r in results_good if r['sharpe'] >= 2.0)
        gt15_sharpe= sum(1 for r in results_good if r['sharpe'] >= 1.5)
        gt10_sharpe= sum(1 for r in results_good if r['sharpe'] >= 1.0)

        print(f"\n{'=' * 60}")
        print(f"  UNIVERSE SUMMARY  ({len(results_good)} assets, 10yr, long-only)")
        print(f"{'=' * 60}")
        print(f"  Avg Sharpe:   {np.mean(sharpes):.2f}  |  Median: {np.median(sharpes):.2f}")
        print(f"  Avg CAGR:     {np.mean(cagrs):.1f}%  |  Median: {np.median(cagrs):.1f}%")
        print(f"  Avg Max DD:   {np.mean(dds):.1f}%   |  Worst:  {max(dds):.1f}%")
        print(f"  Sharpe > 2.0: {gt2_sharpe} assets")
        print(f"  Sharpe > 1.5: {gt15_sharpe} assets")
        print(f"  Sharpe > 1.0: {gt10_sharpe} assets")
        print(f"  Profitable:   {profitable} / {len(results_good)}")
        print(f"  Failed:       {len(results_fail)} assets")
        print(f"{'=' * 60}")

    if results_fail:
        print(f"\nFailed assets: {', '.join(r['symbol'] for r in results_fail)}")

    # -------------------------------------------------------------------------
    all_results = results_good + results_fail
    out_path = 'intraday/results/multi_asset_v6_10yr.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved -> {out_path}")
