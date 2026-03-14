"""
Joint portfolio backtest — GBTC + BTC-USD weekend on shared equity pool.

Fixes tested:
  Fix 1: Lower vol_target on BTC-USD (reduces concurrent position sizes)
  Fix 2: Entry gate — BTC-USD skips entries when GBTC has open longs in same direction
  Fix 3: Shared DD governor — both legs read the same peak equity; BTC-USD halts when
          combined portfolio DD exceeds threshold
  Fix 4: All three combined

The key insight: in a real portfolio both strategies share the same broker account.
We simulate this by running them in lockstep on a single equity variable and a
unified position list, with GBTC and BTC-USD positions tagged by their source strategy.

Daily loop:
  1. Exit all open positions (GBTC positions exit on weekdays using GBTC bars,
     BTC-USD positions exit daily using BTC bars)
  2. Check combined equity DD — apply shared halt/reduce if breached
  3. GBTC signal generation (weekdays only)
  4. BTC-USD signal generation (Fri+Sat only, with optional GBTC-overlap gate)
  5. Fill next day opens
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


def portfolio_metrics(equity_curve: list, index) -> dict:
    eq   = pd.Series(equity_curve, index=index[:len(equity_curve)])
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (eq.iloc[-1] / 100_000) ** (1 / years) - 1
    sh    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    down  = rets[rets < 0]
    ds    = down.std() * np.sqrt(252) if len(down) > 0 else 1e-9
    so    = (rets.mean() * 252) / ds
    roll  = eq.expanding().max()
    mdd   = ((eq - roll) / roll).min() * 100
    return dict(cagr=cagr*100, sharpe=sh, sortino=so, max_dd=float(mdd),
                final=float(eq.iloc[-1]))


def joint_backtest(btc_vol_target=0.60, use_direction_gate=False,
                   shared_dd_halt=0.40, shared_dd_reduce=0.25,
                   label=''):
    """
    Run GBTC + BTC-USD on a shared $100k equity pool.
    
    btc_vol_target:    vol_target for BTC-USD leg (Fix 1: lower = smaller positions)
    use_direction_gate: skip BTC-USD entries if GBTC already has open longs (Fix 2)
    shared_dd_halt:    combined portfolio DD at which BTC-USD entries halt (Fix 3)
                       Default 0.40 (40%) — rarely triggers even in crypto bear markets
    shared_dd_reduce:  combined portfolio DD at which BTC-USD scales to 50%
    """
    # Prepare both strategies (data + indicators only, no backtest)
    gbtc_s = AggressiveHybridV6('GBTC', start=START, end=END, **GBTC_PARAMS)
    gbtc_s.fetch_data()
    gbtc_s.prepare_indicators()

    btc_p = dict(**BTC_PARAMS)
    btc_p['vol_target'] = btc_vol_target
    btc_s = AggressiveHybridV6('BTC-USD', start=START, end=END, **btc_p)
    btc_s.data  = _btc_ref.data.copy()
    btc_s.vix   = _btc_ref.vix
    btc_s._mvrv = _btc_ref._mvrv
    btc_s._fg   = _btc_ref._fg
    btc_s.prepare_indicators()

    # Shared portfolio state
    equity       = 100_000.0
    peak_equity  = 100_000.0
    equity_curve = [100_000.0]
    positions    = []      # each pos tagged with 'source': 'GBTC' or 'BTC'
    closed_trades = []
    gbtc_pending = []
    btc_pending  = []

    # Build a combined daily date index (union of both)
    all_dates = gbtc_s.data.index.union(btc_s.data.index)

    # Pre-extract numpy arrays
    def npa(df, col):
        return df[col].to_numpy(dtype=float) if col in df.columns else np.zeros(len(df))

    # GBTC arrays
    g_dates  = gbtc_s.data.index
    g_open   = npa(gbtc_s.data, 'Open');  g_high = npa(gbtc_s.data, 'High')
    g_low    = npa(gbtc_s.data, 'Low');   g_close= npa(gbtc_s.data, 'Close')
    g_atr    = npa(gbtc_s.data, 'ATR');   g_vix  = npa(gbtc_s.data, 'VIX')

    # BTC arrays
    b_dates  = btc_s.data.index
    b_open   = npa(btc_s.data, 'Open');  b_high = npa(btc_s.data, 'High')
    b_low    = npa(btc_s.data, 'Low');   b_close= npa(btc_s.data, 'Close')
    b_atr    = npa(btc_s.data, 'ATR');   b_vix  = npa(btc_s.data, 'VIX')

    g_idx_map = {d: i for i, d in enumerate(g_dates)}
    b_idx_map = {d: i for i, d in enumerate(b_dates)}

    def exit_position(pos, exit_price, reason, date):
        nonlocal equity
        pnl = ((exit_price - pos['entry']) * pos['qty'] if pos['side'] == 1
               else (pos['entry'] - exit_price) * pos['qty'])
        equity += pnl
        closed_trades.append({'exit_date': date, 'pnl': pnl,
                               'reason': reason, 'source': pos['source']})
        positions.remove(pos)

    def process_exits(date, open_, high, low, close, atr, source_filter):
        nonlocal equity
        for pos in list(positions):
            if pos['source'] != source_filter:
                continue
            ep = None; er = ''
            _short = pos['side'] == -1
            mhold = (pos.get('max_hold_override') or
                     (gbtc_s.max_hold_short if source_filter == 'GBTC' else btc_s.max_hold_short)
                     if _short else
                     (gbtc_s.max_hold_trend if pos.get('sig_type') == 'trend' else gbtc_s.max_hold_mr)
                     if source_filter == 'GBTC' else
                     (btc_s.max_hold_trend if pos.get('sig_type') == 'trend' else btc_s.max_hold_mr))
            if pos['side'] == 1:
                if not pos['trail_active'] and high >= pos['entry'] + gbtc_s.trail_cushion * atr:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    nt = high - pos['trail_atr'] * atr
                    if nt > pos['sl']: pos['sl'] = nt
                if low <= pos['sl']:
                    ep, er = pos['sl'], 'TrailSL'
                elif high >= pos['tp']:
                    ep, er = pos['tp'], 'TP'
                elif not pos['partial_taken'] and high >= pos['partial_tp']:
                    pq = max(1, int(pos['qty'] * gbtc_s.partial_qty_pct))
                    ppnl = (pos['partial_tp'] - pos['entry']) * pq
                    equity += ppnl
                    pos['qty'] -= pq; pos['partial_taken'] = True
                    pos['tp'] = pos['entry'] + pos['trail_atr'] * atr * gbtc_s.post_partial_mult
                    pos['sl'] = pos['entry']
            else:
                if not pos['trail_active'] and low <= pos['entry'] - gbtc_s.trail_cushion * atr:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    nt = low + pos['trail_atr'] * atr
                    if nt < pos['sl']: pos['sl'] = nt
                if high >= pos['sl']:
                    ep, er = pos['sl'], 'TrailSL'
                elif low <= pos['tp']:
                    ep, er = pos['tp'], 'TP'
            if ep is None and (date - pos['date']).days > mhold:
                ep, er = close, 'Max_Hold'
            if ep is not None:
                exit_position(pos, ep, er, date)

    def fill_pending(pending, atr, date):
        filled = []
        for order in pending:
            side   = order['side']
            entry  = order['open_price']
            ta     = order['trail_atr_val']
            sl     = entry - ta * atr if side == 1 else entry + ta * atr
            ptp    = entry + order['ptm'] * atr if side == 1 else entry - order['ptm'] * atr
            tp     = entry + order['tm']  * atr if side == 1 else entry - order['tm']  * atr
            sl_dist = max(abs(entry - sl), 0.01)
            # BTC-USD supports fractional shares; GBTC requires whole shares
            if order['source'] == 'BTC':
                qty = order['risk'] / sl_dist   # float — fractional crypto ok
            else:
                qty = int(order['risk'] / sl_dist)  # GBTC: whole shares only
            if qty > 0:
                positions.append({
                    'date': date, 'entry': entry, 'qty': qty,
                    'sl': sl, 'tp': tp, 'partial_tp': ptp,
                    'partial_taken': False, 'trail_atr': ta,
                    'trail_active': False, 'side': side,
                    'sig_type': order['sig_type'], 'source': order['source'],
                })
                filled.append(order)
        for o in filled:
            pending.remove(o)

    for date in all_dates:
        gi = g_idx_map.get(date)
        bi = b_idx_map.get(date)

        # Exit GBTC positions if today has GBTC data
        if gi is not None and gi >= 200:
            process_exits(date, g_open[gi], g_high[gi], g_low[gi],
                          g_close[gi], g_atr[gi], 'GBTC')

        # Exit BTC positions if today has BTC data
        if bi is not None and bi >= 200:
            process_exits(date, b_open[bi], b_high[bi], b_low[bi],
                          b_close[bi], b_atr[bi], 'BTC')

        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        current_dd  = (peak_equity - equity) / peak_equity

        # Fill pending entries at today's open
        if gi is not None:
            fill_pending(gbtc_pending, g_atr[gi], date)
        if bi is not None:
            fill_pending(btc_pending, b_atr[bi], date)

        # ── GBTC signal generation (weekdays, standard logic via strategy instance) ──
        # GBTC has its own internal DD governor; don't apply shared halt (it over-suppresses)
        if gi is not None and gi >= 200 and date.dayofweek < 5:
            gbtc_s.equity    = equity
            gbtc_s.positions = [p for p in positions if p['source'] == 'GBTC']
            dd_scale = 0.5 if current_dd >= shared_dd_reduce else 1.0
            rv = float(gbtc_s.data['Realized_Vol'].iloc[gi])
            vt = gbtc_s.vol_target * (gbtc_s.vol_regime_scale
                 if gbtc_s.data['Lead_LT'].iloc[gi] >= 0 else 1.0)
            vs = float(np.clip(vt / max(rv, 0.05), 0.4, 1.2))
            lt_regime = ('up' if gbtc_s.data['Lead_LT'].iloc[gi] >= 0
                         and gbtc_s.data['Lead_EMA50'].iloc[gi] > gbtc_s.data['Lead_EMA200'].iloc[gi]
                         else 'down' if gbtc_s.data['Lead_LT'].iloc[gi] < 0
                         and gbtc_s.data['Lead_EMA50'].iloc[gi] < gbtc_s.data['Lead_EMA200'].iloc[gi]
                         else 'transition')
            vix_now = g_vix[gi]
            n_gbtc = sum(1 for p in positions if p['source'] == 'GBTC')
            max_pos = 4 if lt_regime == 'up' else 2

            if lt_regime == 'up' and vix_now <= 45 and n_gbtc < max_pos:
                sigs, str_, stype = gbtc_s.generate_buy_signals(gi)
                if sigs and str_ >= gbtc_s.min_strength_up:
                    risk = equity * 0.010 * vs * dd_scale
                    gbtc_pending.append({
                        'side': 1, 'risk': risk, 'sig_type': stype,
                        'source': 'GBTC', 'open_price': g_open[min(gi+1, len(g_open)-1)],
                        'trail_atr_val': gbtc_s.trail_atr,
                        'ptm': gbtc_s.partial_tp_mult, 'tm': gbtc_s.tp_mult,
                    })
            if lt_regime == 'down' and vix_now <= 60 and n_gbtc < max_pos:
                sigs, str_, stype = gbtc_s.generate_sell_signals(gi)
                if sigs and str_ >= 0.35:
                    risk = equity * 0.010 * vs * dd_scale
                    gbtc_pending.append({
                        'side': -1, 'risk': risk, 'sig_type': stype,
                        'source': 'GBTC', 'open_price': g_open[min(gi+1, len(g_open)-1)],
                        'trail_atr_val': gbtc_s.trail_atr,
                        'ptm': gbtc_s.partial_tp_mult, 'tm': gbtc_s.tp_mult,
                    })

        # ── BTC-USD signal generation (Fri+Sat only) ──────────────────────────────
        if bi is not None and bi >= 200 and date.dayofweek in {4, 5}:
            # Fix 3: shared DD halt
            if current_dd >= shared_dd_halt:
                pass  # skip BTC entries
            else:
                # Fix 2: direction gate — skip if GBTC has open longs (same direction risk)
                gbtc_long_open = sum(1 for p in positions if p['source'] == 'GBTC' and p['side'] == 1)
                if use_direction_gate and gbtc_long_open > 0:
                    pass  # skip — GBTC already long BTC
                else:
                    dd_scale = 0.5 if current_dd >= shared_dd_reduce else 1.0
                    rv = float(btc_s.data['Realized_Vol'].iloc[bi])
                    vt = btc_vol_target
                    vs = float(np.clip(vt / max(rv, 0.05), 0.4, 1.2))
                    n_btc = sum(1 for p in positions if p['source'] == 'BTC')
                    lt = btc_s.data['Lead_LT'].iloc[bi]
                    lt_e50 = btc_s.data['Lead_EMA50'].iloc[bi]
                    lt_e200 = btc_s.data['Lead_EMA200'].iloc[bi]
                    lt_regime = ('up' if lt >= 0 and lt_e50 > lt_e200
                                 else 'down' if lt < 0 and lt_e50 < lt_e200 else 'transition')
                    vix_now = b_vix[bi]

                    if lt_regime == 'up' and vix_now <= 45 and n_btc < 2:
                        sigs, str_, stype = btc_s.generate_buy_signals(bi)
                        if sigs and str_ >= btc_s.min_strength_up:
                            risk = equity * 0.010 * vs * dd_scale
                            btc_pending.append({
                                'side': 1, 'risk': risk, 'sig_type': stype,
                                'source': 'BTC', 'open_price': b_open[min(bi+1, len(b_open)-1)],
                                'trail_atr_val': btc_s.trail_atr,
                                'ptm': btc_s.partial_tp_mult, 'tm': btc_s.tp_mult,
                            })
                    if lt_regime == 'down' and vix_now <= 60 and n_btc < 2:
                        sigs, str_, stype = btc_s.generate_sell_signals(bi)
                        if sigs and str_ >= 0.35:
                            risk = equity * 0.010 * vs * dd_scale
                            btc_pending.append({
                                'side': -1, 'risk': risk, 'sig_type': stype,
                                'source': 'BTC', 'open_price': b_open[min(bi+1, len(b_open)-1)],
                                'trail_atr_val': btc_s.trail_atr,
                                'ptm': btc_s.partial_tp_mult, 'tm': btc_s.tp_mult,
                            })

    # Build equity index for metrics — equity_curve has one entry per date + 1 initial
    idx = all_dates[:len(equity_curve)]
    if len(equity_curve) > len(all_dates):
        equity_curve = equity_curve[:len(all_dates)]
    m = portfolio_metrics(equity_curve, idx)
    gbtc_trades = sum(1 for t in closed_trades if t['source'] == 'GBTC')
    btc_trades  = sum(1 for t in closed_trades if t['source'] == 'BTC')
    print(f"  {label:<42}  Sharpe={m['sharpe']:>6.3f}  Sortino={m['sortino']:>6.3f}  "
          f"CAGR={m['cagr']:>6.2f}%  MaxDD={m['max_dd']:>6.1f}%  "
          f"Final=${m['final']:>10,.0f}  (G:{gbtc_trades} B:{btc_trades})")
    return m


# ── Pre-fetch BTC-USD data once ───────────────────────────────────────────────
print("Fetching BTC-USD reference data...")
_btc_ref = AggressiveHybridV6('BTC-USD', start=START, end=END, **BTC_PARAMS)
_btc_ref.fetch_data()
print(f"BTC-USD: {len(_btc_ref.data)} bars\n")

print("Running joint portfolio backtests...\n")
print(f"  {'Config':<42}  {'Sharpe':>7}  {'Sortino':>8}  {'CAGR':>7}  "
      f"{'MaxDD':>7}  {'Final $':>12}  Trades")
print("  " + "-" * 100)

# Baseline: standalone GBTC
g = AggressiveHybridV6('GBTC', start=START, end=END, **GBTC_PARAMS)
g.fetch_data()
gr = g.backtest()
geq = pd.Series(g.equity_curve, index=g.data.index[:len(g.equity_curve)])
grets = geq.pct_change().dropna()
gdown = grets[grets < 0]; gds = gdown.std() * np.sqrt(252)
gso = (grets.mean() * 252) / gds
print(f"  {'GBTC alone':<42}  Sharpe={gr['sharpe']:>6.3f}  Sortino={gso:>6.3f}  "
      f"CAGR={gr['cagr']:>6.2f}%  MaxDD={gr['max_dd']:>6.1f}%  "
      f"Final=${100_000*(1+gr['cagr']/100)**10:>10,.0f}  ({gr['trades']} trades)")
print()

# Joint portfolio variants
joint_backtest(btc_vol_target=0.60, use_direction_gate=False,
               label='Naive additive (Config G baseline)')
joint_backtest(btc_vol_target=0.30, use_direction_gate=False,
               label='Fix 1: BTC vol=0.30 (half sizing)')
joint_backtest(btc_vol_target=0.20, use_direction_gate=False,
               label='Fix 1: BTC vol=0.20 (1/3 sizing)')
joint_backtest(btc_vol_target=0.60, use_direction_gate=True,
               label='Fix 2: Direction gate (no BTC if GBTC long)')
joint_backtest(btc_vol_target=0.30, use_direction_gate=True,
               label='Fix 1+2: vol=0.30 + direction gate')
joint_backtest(btc_vol_target=0.20, use_direction_gate=True,
               label='Fix 1+2: vol=0.20 + direction gate')
