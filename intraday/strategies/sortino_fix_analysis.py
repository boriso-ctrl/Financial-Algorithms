"""
Sortino Fix Analysis — Honest post-hoc approach.

Problem: Combining GBTC + BTC-USD weekend (Config G) drops Sortino from ~1.16 to ~0.6.
Root cause: Both strategies are BTC-exposed → correlated downside.

Three scenarios tested:
  A) GBTC alone (baseline)
  B) BTC-USD Config G alone
  C) Additive combined (GBTC PnL + BTC PnL on shared $100k starting base)
  D) Direction-gated combined: BTC-USD skips entry when GBTC already has open longs
     (BTC re-run from scratch with gate applied)

Method: Each strategy runs its own faithful standalone backtest. Combined portfolio
is computed by summing daily PnL from each leg (additive). For the direction gate,
BTC-USD is re-run using the strategy's own generate_buy/sell_signals, gating entry
on dates when GBTC had open long positions.

This avoids joint-sim reimplementation bugs (max_pos mismatch, shared DD halt
over-suppression, vol scaling differences).
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from intraday.strategies.aggressive_hybrid_v6_10yr import AggressiveHybridV6

START = '2015-01-01'
END   = '2025-01-01'

GBTC_PARAMS = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=145, adx_thresh=32, min_strength_up=0.30,
    trail_cushion=0.5, post_partial_mult=2.5, macd_fast=8, macd_slow=38,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=60,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
    signal_ticker='BTC-USD', signal_ema_period=180,
)

BTC_PARAMS = dict(
    trail_atr=4.0, vol_target=0.60, tp_mult=3.0, partial_tp_mult=1.0,
    rsi_period=9, rsi_oversold=33, atr_period=14,
    ema_trend=145, adx_thresh=32, min_strength_up=0.25,
    trail_cushion=0.5, post_partial_mult=2.0, macd_fast=8, macd_slow=38,
    max_hold_trend=90, max_hold_mr=25,
    enable_bb_signal=True, partial_qty_pct=0.33, vol_regime_scale=1.1,
    allow_shorts=True, max_hold_short=60,
    use_onchain=True, mvrv_long_thresh=2.0, mvrv_short_thresh=3.5,
    fg_fear_thresh=25, fg_greed_thresh=75,
    entry_days={4, 5},
)


def compute_metrics(equity_series: pd.Series, label: str) -> dict:
    """Compute standard metrics from a daily equity series."""
    rets  = equity_series.pct_change().dropna()
    years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    cagr  = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1
    sh    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    # RMS semideviation (target=0): matches main strategy's _metrics() formula.
    # Using std(negative_rets) understates downside by excluding zero-return days
    # and applying Bessel correction — avoid that inconsistency.
    downside_sq = np.minimum(rets.values, 0) ** 2
    ds          = np.sqrt(downside_sq.mean()) * np.sqrt(252)
    so          = (rets.mean() * 252) / ds if ds > 0 else 0.0
    roll  = equity_series.expanding().max()
    mdd   = ((equity_series - roll) / roll).min() * 100
    final = float(equity_series.iloc[-1])
    n_neg = (rets < 0).sum()
    n_pos = (rets > 0).sum()
    print(f"  {label:<48}  Sharpe={sh:>6.3f}  Sortino={so:>6.3f}  "
          f"CAGR={cagr*100:>6.2f}%  MaxDD={float(mdd):>6.1f}%  "
          f"Final=${final:>10,.0f}  ({n_pos}up/{n_neg}dn days)")
    return dict(cagr=cagr*100, sharpe=sh, sortino=so, max_dd=float(mdd), final=final)


def equity_to_daily_change(strategy_equity_curve: list, strategy_index) -> pd.Series:
    """Convert a strategy's equity_curve list to a daily-PnL series on the given index."""
    eq  = pd.Series(strategy_equity_curve, index=strategy_index[:len(strategy_equity_curve)])
    return eq.diff().fillna(0)


def run_btc_gated(btc_s: AggressiveHybridV6, gbtc_long_dates: set,
                  skip_when_gbtc_long: bool) -> pd.Series:
    """
    Re-run BTC-USD entry loop on already-prepared strategy instance.
    Uses generate_buy/sell_signals for signal generation.
    Skips entry generation on dates in gbtc_long_dates if skip_when_gbtc_long=True.

    Returns daily equity change series indexed over BTC-USD data dates.
    """
    npa = lambda col: btc_s.data[col].to_numpy(dtype=float)
    _open  = npa('Open');  _high  = npa('High');  _low = npa('Low')
    _close = npa('Close'); _atr   = npa('ATR');   _vix = npa('VIX')
    _rvol  = npa('Realized_Vol')
    _dates = btc_s.data.index

    equity        = 100_000.0
    peak_equity   = 100_000.0
    positions     = []
    closed_trades = []
    pending       = []
    daily_pnl     = []

    def close_pos(pos, ep, date):
        nonlocal equity
        pnl = ((ep - pos['entry']) * pos['qty'] if pos['side'] == 1
               else (pos['entry'] - ep) * pos['qty'])
        equity += pnl
        closed_trades.append({'entry_date': pos['date'], 'exit_date': date,
                               'pnl': pnl, 'side': pos['side']})
        positions.remove(pos)

    for idx in range(200, len(btc_s.data)):
        date   = _dates[idx]
        open_  = _open[idx];  high  = _high[idx]; low = _low[idx]
        close  = _close[idx]; atr   = _atr[idx];  vix = _vix[idx]
        rv     = _rvol[idx]
        eq_before = equity

        # Fill pending (yesterday's signals) at today's open
        new_pending = []
        for order in pending:
            sl_dist = abs(open_ - (open_ - order['trail_atr'] * atr
                                   if order['side'] == 1
                                   else open_ + order['trail_atr'] * atr))
            sl_dist = max(sl_dist, 0.01)
            # BTC-USD: fractional quantity (crypto supports fractional shares)
            qty = order['risk'] / sl_dist
            if qty > 0:
                sl  = open_ - order['trail_atr'] * atr if order['side'] == 1 else open_ + order['trail_atr'] * atr
                ptp = open_ + btc_s.partial_tp_mult * atr if order['side'] == 1 else open_ - btc_s.partial_tp_mult * atr
                tp  = open_ + btc_s.tp_mult * atr if order['side'] == 1 else open_ - btc_s.tp_mult * atr
                positions.append({
                    'date': date, 'entry': open_, 'qty': qty,
                    'sl': sl, 'tp': tp, 'partial_tp': ptp,
                    'partial_taken': False, 'trail_atr': order['trail_atr'],
                    'trail_active': False, 'side': order['side'],
                    'sig_type': order['sig_type'],
                })
        pending = []  # reset: only 1 bar lookahead, same as standalone

        # Manage exits
        for pos in list(positions):
            ep = None; er = ''
            _short = pos['side'] == -1
            max_hold = (btc_s.max_hold_short if _short else
                        (btc_s.max_hold_trend if pos.get('sig_type') == 'trend' else btc_s.max_hold_mr))
            if pos['side'] == 1:
                if not pos['trail_active'] and high >= pos['entry'] + btc_s.trail_cushion * atr:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    nt = high - pos['trail_atr'] * atr
                    if nt > pos['sl']: pos['sl'] = nt
                if low <= pos['sl']:  ep, er = pos['sl'], 'Trail'
                elif high >= pos['tp']: ep, er = pos['tp'], 'TP'
                elif not pos['partial_taken'] and high >= pos['partial_tp']:
                    pq  = pos['qty'] * btc_s.partial_qty_pct
                    ppnl = (pos['partial_tp'] - pos['entry']) * pq
                    equity += ppnl
                    pos['qty'] -= pq; pos['partial_taken'] = True
                    pos['tp'] = pos['entry'] + pos['trail_atr'] * atr * btc_s.post_partial_mult
                    pos['sl'] = pos['entry']
            else:  # short
                if not pos['trail_active'] and low <= pos['entry'] - btc_s.trail_cushion * atr:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    nt = low + pos['trail_atr'] * atr
                    if nt < pos['sl']: pos['sl'] = nt
                if high >= pos['sl']:  ep, er = pos['sl'], 'Trail'
                elif low <= pos['tp']: ep, er = pos['tp'], 'TP'
            if ep is None and (date - pos['date']).days > max_hold:
                ep, er = close, 'Max_Hold'
            if ep is not None:
                close_pos(pos, ep, date)

        daily_pnl.append(equity - eq_before)
        peak_equity = max(peak_equity, equity)

        # Signal generation — Fri/Sat only
        if date.dayofweek not in {4, 5}:
            continue

        # Direction gate: skip entry if GBTC had open longs on this date
        if skip_when_gbtc_long and date in gbtc_long_dates:
            continue

        if idx < 200:
            continue

        lt   = btc_s.data['Lead_LT'].iloc[idx]
        e50  = btc_s.data['Lead_EMA50'].iloc[idx]
        e200 = btc_s.data['Lead_EMA200'].iloc[idx]
        lt_regime = ('up' if lt >= 0 and e50 > e200
                     else 'down' if lt < 0 and e50 < e200 else 'transition')

        peak_equity_ = max(peak_equity, equity)
        cur_dd       = (peak_equity_ - equity) / peak_equity_
        dd_halt      = getattr(btc_s, 'dd_halt', 0.20)
        if cur_dd >= dd_halt:
            continue

        vt = btc_s.vol_target
        vs = float(np.clip(vt / max(rv, 0.05), 0.4, 1.2))
        n  = len(positions)

        if lt_regime == 'up' and vix <= 45 and n < 2:
            sigs, str_, stype = btc_s.generate_buy_signals(idx)
            if sigs and str_ >= btc_s.min_strength_up:
                risk = equity * 0.010 * vs
                pending.append({'side': 1, 'risk': risk, 'sig_type': stype,
                                'trail_atr': btc_s.trail_atr})
        if lt_regime == 'down' and vix <= 60 and n < 2:
            sigs, str_, stype = btc_s.generate_sell_signals(idx)
            if sigs and str_ >= 0.35:
                risk = equity * 0.010 * vs
                pending.append({'side': -1, 'risk': risk, 'sig_type': stype,
                                'trail_atr': btc_s.trail_atr})

    pnl_series = pd.Series(daily_pnl, index=_dates[200:])
    n_trades   = len(closed_trades)
    final_eq   = 100_000.0 + float(sum(daily_pnl))
    print(f"    BTC-USD {'gated' if skip_when_gbtc_long else 'standalone'}: "
          f"{n_trades} trades, final equity=${final_eq:,.0f}")
    return pnl_series


# ──────────────────────────────────────────────────────────────────────
print("=" * 90)
print("SORTINO FIX ANALYSIS — Additive PnL + Direction Gate")
print("=" * 90)

# ── Step 1: GBTC standalone ───────────────────────────────────────────
print("\n[1] Running GBTC standalone backtest...")
g = AggressiveHybridV6('GBTC', start=START, end=END, **GBTC_PARAMS)
g.fetch_data()
gr = g.backtest()

gbtc_eq = pd.Series(g.equity_curve, index=g.data.index[:len(g.equity_curve)])
# Reindex to a daily calendar (forward-fill weekdays → weekend/holiday gaps stay flat)
gbtc_daily = gbtc_eq.reindex(
    pd.bdate_range(START, END), method='ffill'
).dropna()

print(f"    GBTC: {gr['trades']} trades, final equity=${gbtc_eq.iloc[-1]:,.0f}")

# Build set of dates on which GBTC had at least one open LONG position
gbtc_long_dates = set()
for t in g.closed_trades:
    if t.get('side', '').upper() == 'LONG':
        d_range = pd.date_range(t['entry_date'], t['exit_date'], freq='D')
        gbtc_long_dates.update(d_range)

print(f"    GBTC long-position calendar: {len(gbtc_long_dates)} dates covered")

# ── Step 2: BTC-USD Config G standalone ──────────────────────────────
print("\n[2] Running BTC-USD Config G standalone backtest...")
b = AggressiveHybridV6('BTC-USD', start=START, end=END, **BTC_PARAMS)
b.fetch_data()
b.prepare_indicators()
# Run the standalone for baseline metrics
b_standalone = AggressiveHybridV6('BTC-USD', start=START, end=END, **BTC_PARAMS)
b_standalone.data  = b.data.copy()
b_standalone.vix   = b.vix
b_standalone._mvrv = b._mvrv
b_standalone._fg   = b._fg
b_standalone.prepare_indicators()
br = b_standalone.backtest()
print(f"    BTC-USD standalone: {br['trades']} trades")

# BTC daily PnL from standalone
btc_eq_standalone = pd.Series(
    b_standalone.equity_curve,
    index=b_standalone.data.index[:len(b_standalone.equity_curve)]
)

# ── Step 3: Re-run BTC-USD with gating using our own loop ─────────────
print("\n[3] BTC-USD standalone via re-run loop (verifying loop accuracy)...")
btc_pnl_ungated = run_btc_gated(b, gbtc_long_dates, skip_when_gbtc_long=False)

print("\n[4] BTC-USD with direction gate (skip when GBTC long)...")
# Need a fresh strategy instance for the gated run
b2 = AggressiveHybridV6('BTC-USD', start=START, end=END, **BTC_PARAMS)
b2.data  = b.data.copy()
b2.vix   = b.vix
b2._mvrv = b._mvrv
b2._fg   = b._fg
b2.prepare_indicators()
btc_pnl_gated = run_btc_gated(b2, gbtc_long_dates, skip_when_gbtc_long=True)

# ── Step 4: Build combined equity series ─────────────────────────────
print("\n[5] Building combined equity series...\n")

# GBTC daily PnL
gbtc_pnl = gbtc_eq.diff().fillna(0)
gbtc_pnl.index = gbtc_pnl.index.normalize()  # strip time if any

# Align all PnL series to a single daily calendar (GBTC weekdays + BTC all days)
all_dates = gbtc_pnl.index.union(btc_pnl_ungated.index).sort_values()

gbtc_pnl_aligned   = gbtc_pnl.reindex(all_dates, fill_value=0)
btc_pnl_u_aligned  = btc_pnl_ungated.reindex(all_dates, fill_value=0)
btc_pnl_g_aligned  = btc_pnl_gated.reindex(all_dates, fill_value=0)

# Build equity series for each scenario
combined_pnl_naive  = gbtc_pnl_aligned + btc_pnl_u_aligned
combined_pnl_gated  = gbtc_pnl_aligned + btc_pnl_g_aligned

# GBTC-only standalone (re-indexed to all_dates for consistent comparison)
gbtc_alone_eq = pd.Series(
    100_000.0 + gbtc_pnl_aligned.cumsum().values,
    index=all_dates
)
btc_alone_eq = pd.Series(
    100_000.0 + btc_pnl_u_aligned.cumsum().values,
    index=all_dates
)
combined_naive_eq = pd.Series(
    100_000.0 + combined_pnl_naive.cumsum().values,
    index=all_dates
)
combined_gated_eq = pd.Series(
    100_000.0 + combined_pnl_gated.cumsum().values,
    index=all_dates
)

# ── Results ───────────────────────────────────────────────────────────
print(f"  {'Scenario':<48}  {'Sharpe':>7}  {'Sortino':>8}  {'CAGR':>7}  "
      f"{'MaxDD':>7}  {'Final $':>12}  Days")
print("  " + "-" * 110)

compute_metrics(gbtc_alone_eq,         "A) GBTC alone (Config G regime)")
compute_metrics(btc_alone_eq,          "B) BTC-USD Config G alone (additive)")
compute_metrics(combined_naive_eq,     "C) Combined additive (GBTC + BTC, no gate)")
compute_metrics(combined_gated_eq,     "D) Combined + direction gate (BTC skips when GBTC long)")

# ── Overlap analysis ──────────────────────────────────────────────────
print("\n[6] Overlap analysis:")
# How many BTC-USD loss days occurred when GBTC also had open longs?
btc_loss_days = btc_pnl_u_aligned.index[btc_pnl_u_aligned < 0]
overlap        = btc_loss_days[btc_loss_days.isin(gbtc_long_dates)]
print(f"    BTC loss days: {len(btc_loss_days)}")
print(f"    GBTC-covered dates: {len(gbtc_long_dates)}")
print(f"    BTC loss days overlapping GBTC long: {len(overlap)} ({100*len(overlap)/max(len(btc_loss_days),1):.1f}%)")

total_btc_loss    = btc_pnl_u_aligned[btc_pnl_u_aligned < 0].sum()
overlap_btc_loss  = btc_pnl_u_aligned.loc[overlap].sum() if len(overlap) > 0 else 0
print(f"    Total BTC loss over period: ${total_btc_loss:,.0f}")
print(f"    Overlap-period BTC loss: ${overlap_btc_loss:,.0f} ({100*overlap_btc_loss/min(total_btc_loss,-1):.1f}% of total)")

# Correlation of daily PnL
both_active = (gbtc_pnl_aligned != 0) & (btc_pnl_u_aligned != 0)
corr = gbtc_pnl_aligned[both_active].corr(btc_pnl_u_aligned[both_active])
print(f"    Daily PnL correlation (days both active): {corr:.3f}")

corr_down = gbtc_pnl_aligned[both_active & (gbtc_pnl_aligned < 0)].corr(
    btc_pnl_u_aligned[both_active & (gbtc_pnl_aligned < 0)])
print(f"    Downside correlation (GBTC losing days):  {corr_down:.3f}")
