"""
Multi-Asset Backtest Runner
===========================
Runs AggressiveHybridV5Sharpe across a diverse universe of:
  - Equity indices / ETFs
  - Large-cap stocks
  - Commodities (via ETFs)
  - Fixed income (as regime context)
  - Crypto

Date range: 2020-01-01 → 2025-12-31  (5+ years, 1500+ daily bars)
"""

import sys
import os
import json
import time
import logging
import pandas as pd
import numpy as np

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(__file__))
from aggressive_hybrid_v5_sharpe import AggressiveHybridV5Sharpe

logging.basicConfig(level=logging.WARNING)   # suppress per-asset INFO spam

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
UNIVERSE = {
    # ── Broad equity ETFs ────────────────────────────────────────────────
    'SPY':  'S&P 500',
    'QQQ':  'NASDAQ 100',
    'IWM':  'Russell 2000',
    'DIA':  'Dow Jones',
    'VTI':  'Total US Market',

    # ── Sector ETFs ───────────────────────────────────────────────────────
    'XLK':  'Technology',
    'XLE':  'Energy',
    'XLF':  'Financials',
    'XLV':  'Health Care',
    'XBI':  'Biotech',

    # ── Large-cap stocks ─────────────────────────────────────────────────
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'NVDA': 'NVIDIA',
    'AMZN': 'Amazon',
    'GOOGL':'Alphabet',
    'META': 'Meta',
    'TSLA': 'Tesla',
    'JPM':  'JPMorgan',
    'BRK-B':'Berkshire',
    'UNH':  'UnitedHealth',

    # ── Commodities (ETF proxies) ─────────────────────────────────────────
    'GLD':  'Gold',
    'SLV':  'Silver',
    'USO':  'Oil (USO)',
    'UNG':  'Natural Gas',
    'CORN': 'Corn',
    'WEAT': 'Wheat',
    'PDBC': 'Diversified Commodities',

    # ── Fixed income / alternatives ───────────────────────────────────────
    'TLT':  '20yr Treasury',
    'HYG':  'High Yield Bonds',
    'GDX':  'Gold Miners',

    # ── Crypto (via ETFs / trusts) ────────────────────────────────────────
    'IBIT': 'Bitcoin ETF (IBIT)',
    'FBTC': 'Bitcoin ETF (Fidelity)',
    'ETHA': 'Ethereum ETF',
    'GBTC': 'Bitcoin Trust (GBTC)',
    'MSTR': 'MicroStrategy (BTC proxy)',

    # ── International ─────────────────────────────────────────────────────
    'EEM':  'Emerging Markets',
    'EWJ':  'Japan',
    'EWZ':  'Brazil',
}

START = '2020-01-01'
END   = '2025-12-31'

# ---------------------------------------------------------------------------

def run_one(symbol, label):
    """Run backtest for a single asset; return results dict or error dict."""
    try:
        trader = AggressiveHybridV5Sharpe(ticker=symbol, start=START, end=END)
        if not trader.fetch_data():
            return {'symbol': symbol, 'label': label, 'error': 'fetch failed'}
        if len(trader.data) < 300:
            return {'symbol': symbol, 'label': label, 'error': f'insufficient data ({len(trader.data)} bars)'}

        results = trader.backtest()
        if results is None or 'error' in results:
            return {'symbol': symbol, 'label': label, 'error': results.get('error','no trades')}

        results['label'] = label
        return results
    except Exception as e:
        return {'symbol': symbol, 'label': label, 'error': str(e)}


def score(r):
    """Composite score = Sharpe + CAGR/30 (balances risk-adj return & raw return)"""
    return r.get('sharpe', 0) + r.get('cagr', 0) / 30


def print_table(rows, title):
    cols = ['symbol', 'label', 'years', 'return_pct', 'cagr', 'sharpe', 'sortino',
            'win_rate', 'trades', 'max_dd', 'profit_factor']
    headers = {
        'symbol': 'Symbol',    'label': 'Asset',          'years': 'Yrs',
        'return_pct': 'Return%', 'cagr': 'CAGR%',         'sharpe': 'Sharpe',
        'sortino': 'Sortino',  'win_rate': 'WR%',          'trades': 'Trades',
        'max_dd': 'MaxDD%',    'profit_factor': 'PF',
    }
    widths = {'symbol':6,'label':22,'years':4,'return_pct':8,'cagr':7,
              'sharpe':7,'sortino':7,'win_rate':6,'trades':7,'max_dd':7,'profit_factor':5}

    sep  = '-' * (sum(widths.values()) + 3 * len(cols))
    hrow = '  '.join(f"{headers[c]:<{widths[c]}}" for c in cols)
    print(f'\n{"=" * len(sep)}')
    print(f'  {title}')
    print('=' * len(sep))
    print(hrow)
    print(sep)
    for r in rows:
        row = (
            f"{r.get('symbol',''):<6}  "
            f"{r.get('label',''):<22}  "
            f"{r.get('years',0):<4}  "
            f"{r.get('return_pct',0):<8.1f}  "
            f"{r.get('cagr',0):<7.1f}  "
            f"{r.get('sharpe',0):<7.2f}  "
            f"{r.get('sortino',0):<7.2f}  "
            f"{r.get('win_rate',0):<6.1f}  "
            f"{r.get('trades',0):<7}  "
            f"{r.get('max_dd',0):<7.1f}  "
            f"{r.get('profit_factor',0):<5.2f}"
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\n{'=' * 70}")
    print(f"  MULTI-ASSET BACKTEST  |  V5 Sharpe Strategy  |  {START} → {END}")
    print(f"  Universe: {len(UNIVERSE)} assets (equities, commodities, crypto, int'l)")
    print(f"{'=' * 70}\n")

    results_good = []
    results_fail = []
    total = len(UNIVERSE)

    # Run sequentially — yfinance is not thread-safe; concurrent calls batch
    # together and return cross-contaminated multi-ticker DataFrames
    for done, (sym, lbl) in enumerate(UNIVERSE.items(), 1):
        res = run_one(sym, lbl)
        if 'error' in res:
            results_fail.append(res)
            status = f"  [{done:>2}/{total}] ❌  {sym:<6}  ERROR: {res['error']}"
        else:
            results_good.append(res)
            status = (f"  [{done:>2}/{total}] ✅  {sym:<6}  "
                      f"Sharpe {res['sharpe']:.2f}  CAGR {res['cagr']:.1f}%  "
                      f"WR {res['win_rate']:.0f}%  Trades {res['trades']}")
        print(status, flush=True)

    # ---------------------------------------------------------------------
    # Sort and display
    # ---------------------------------------------------------------------
    if not results_good:
        print("\n⚠️  No successful backtests — check internet connectivity.")
        sys.exit(1)

    # Sort by composite score (Sharpe + CAGR/30)
    results_good.sort(key=score, reverse=True)

    print_table(results_good, 'ALL ASSETS — ranked by Sharpe + CAGR/30')

    # Sub-tables
    top10 = results_good[:10]
    print_table(top10, 'TOP 10 PERFORMERS')

    # Worst performers
    worst = sorted(results_good, key=score)[:5]
    print_table(worst, 'BOTTOM 5 (lowest Sharpe+CAGR score)')

    # Stats summary
    sharpes = [r['sharpe'] for r in results_good]
    cagrs   = [r['cagr']   for r in results_good]
    dds     = [r['max_dd'] for r in results_good]
    n = len(results_good)

    print(f"\n{'=' * 60}")
    print(f"  UNIVERSE SUMMARY  ({n} assets backfilled)")
    print(f"{'=' * 60}")
    print(f"  Avg Sharpe:     {np.mean(sharpes):.2f}  |  Median: {np.median(sharpes):.2f}")
    print(f"  Avg CAGR:       {np.mean(cagrs):.1f}%  |  Median: {np.median(cagrs):.1f}%")
    print(f"  Avg Max DD:     {np.mean(dds):.1f}%  |  Worst:  {max(dds):.1f}%")
    print(f"  Sharpe > 2.0:   {sum(1 for s in sharpes if s >= 2.0)} assets")
    print(f"  Sharpe > 1.5:   {sum(1 for s in sharpes if s >= 1.5)} assets")
    print(f"  Profitable:     {sum(1 for r in results_good if r['return_pct'] > 0)} / {n}")
    print(f"  Failed/skipped: {len(results_fail)} assets")
    print(f"{'=' * 60}")

    if results_fail:
        print("\nFailed assets:", ', '.join(r['symbol'] for r in results_fail))

    # Save to JSON
    out = {
        'config': {'start': START, 'end': END, 'strategy': 'AggressiveHybridV5Sharpe'},
        'summary': {
            'total_assets': total,
            'successful': n,
            'failed': len(results_fail),
            'avg_sharpe': round(np.mean(sharpes), 2),
            'avg_cagr': round(np.mean(cagrs), 2),
        },
        'results': results_good,
        'failed': results_fail,
    }
    out_path = 'intraday/results/multi_asset_results_v5.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nFull results saved → {out_path}")
