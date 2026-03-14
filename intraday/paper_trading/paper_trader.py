"""
Alpaca Paper Trading Runner — V7/V8 Strategy
=============================================

Run this script once per day, ideally 15-30 minutes after market close.
It will:
  1. Pull the last 260 days of daily OHLCV data for each asset
  2. Run V6/V7/V8 indicators + signal generation on the latest bar
  3. Manage open positions: update trailing stops, take partial profits
  4. Queue new bracket orders (market entry + TP limit + SL stop) for next open
  5. Print a full status report

Usage:
  python paper_trader.py              # live run (submits real paper trades)
  python paper_trader.py --dry-run    # simulate only, no orders placed

Credentials:
  Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the .env file at project root.
  Paper trading endpoint is used by default (paper=True).

State persistence:
  intraday/paper_trading/state.json   — open position metadata (SL levels, etc.)
  intraday/paper_trading/trade_log.json — all completed trade records
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    GetOrdersRequest,
    ReplaceOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[2]
STATE_FILE = Path(__file__).parent / 'state.json'
LOG_FILE   = Path(__file__).parent / 'trade_log.json'

sys.path.insert(0, str(REPO_ROOT / 'intraday' / 'strategies'))
from aggressive_hybrid_v6_10yr import AggressiveHybridV6

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-asset configs (V7 signal-optimized; swap in V8 when ready)
# Update this dict with v8_execution_configs.json values once V8 completes.
# ---------------------------------------------------------------------------
ASSET_CONFIGS: dict[str, dict] = {
    'QQQ': {
        # V6 numeric
        'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 3.0, 'partial_tp_mult': 1.5,
        # V7 signal
        'rsi_period': 9, 'rsi_oversold': 38, 'atr_period': 14,
        'ema_trend': 30, 'adx_thresh': 18, 'min_strength_up': 0.30,
        # V8 execution
        'trail_cushion': 1.5, 'post_partial_mult': 1.5,
        'macd_fast': 8, 'macd_slow': 38,
        'max_hold_trend': 60, 'max_hold_mr': 25,
    },
    'GBTC': {
        'trail_atr': 4.0, 'vol_target': 0.60, 'tp_mult': 3.0, 'partial_tp_mult': 1.0,
        'rsi_period': 9, 'rsi_oversold': 33, 'atr_period': 14,
        'ema_trend': 145, 'adx_thresh': 32, 'min_strength_up': 0.30,
        'trail_cushion': 0.5, 'post_partial_mult': 2.5,
        'macd_fast': 8, 'macd_slow': 38,
        'max_hold_trend': 90, 'max_hold_mr': 25,
        # V10: on-chain signals + BTC-USD 24/7 regime leading (ema=180)
        # BTC-USD crossover states drive regime gate; GBTC native EMAs for entries
        # Result: Sharpe 1.660, CAGR 17.94%, 924 trades, MaxDD 6.9% (vs 1.47/14.19% V10)
        'enable_bb_signal': True, 'partial_qty_pct': 0.33, 'vol_regime_scale': 1.1,
        'allow_shorts': True, 'max_hold_short': 60,
        'use_onchain': True, 'mvrv_long_thresh': 2.0, 'mvrv_short_thresh': 3.5,
        'fg_fear_thresh': 25, 'fg_greed_thresh': 75,
        'signal_ticker': 'BTC-USD', 'signal_ema_period': 180,
    },
    'XLK': {
        'trail_atr': 3.0, 'vol_target': 0.22, 'tp_mult': 4.5, 'partial_tp_mult': 1.5,
        'rsi_period': 14, 'rsi_oversold': 38, 'atr_period': 14,
        'ema_trend': 80, 'adx_thresh': 22, 'min_strength_up': 0.20,
        'trail_cushion': 1.0, 'post_partial_mult': 2.0,
        'macd_fast': 8, 'macd_slow': 26,
        'max_hold_trend': 80, 'max_hold_mr': 25,
        # V9 additions (CAGR 20.0%→20.29%)
        'vol_regime_scale': 1.1,
    },
    'NVDA': {
        'trail_atr': 3.5, 'vol_target': 0.22, 'tp_mult': 3.0, 'partial_tp_mult': 1.5,
        'rsi_period': 9, 'rsi_oversold': 38, 'atr_period': 20,
        'ema_trend': 50, 'adx_thresh': 27, 'min_strength_up': 0.25,
        'trail_cushion': 2.0, 'post_partial_mult': 2.0,
        'macd_fast': 12, 'macd_slow': 26,
        'max_hold_trend': 60, 'max_hold_mr': 25,
        # V9 additions (Sharpe 2.09→2.16, CAGR 12.4%→12.71%)
        'enable_bb_signal': True, 'partial_qty_pct': 0.67, 'vol_regime_scale': 1.1,
    },
    # ── BTC-USD weekend strategy (Config G, hold=90) ────────────────────────
    # Backtest: Combined portfolio CAGR 25.38%, Sharpe 1.632, MaxDD -12.4%, $848k
    # Entry restricted to Fri+Sat bars only (entry_days={4,5}).
    # BTC-USD trades 24/7; Alpaca symbol is 'BTC/USD' (slash), yfinance is 'BTC-USD'.
    'BTC/USD': {
        'yf_ticker': 'BTC-USD',   # Yahoo Finance ticker for data fetch
        'is_crypto': True,        # enables fractional qty + GTC TIF for entries
        'entry_days': {4, 5},     # Friday=4, Saturday=5 only
        'trail_atr': 4.0, 'vol_target': 0.60, 'tp_mult': 3.0, 'partial_tp_mult': 1.0,
        'rsi_period': 9, 'rsi_oversold': 33, 'atr_period': 14,
        'ema_trend': 145, 'adx_thresh': 32, 'min_strength_up': 0.25,
        'trail_cushion': 0.5, 'post_partial_mult': 2.0,
        'macd_fast': 8, 'macd_slow': 38,
        'max_hold_trend': 90, 'max_hold_mr': 25,
        'enable_bb_signal': True, 'partial_qty_pct': 0.33, 'vol_regime_scale': 1.1,
        'allow_shorts': True, 'max_hold_short': 60,
        'use_onchain': True, 'mvrv_long_thresh': 2.0, 'mvrv_short_thresh': 3.5,
        'fg_fear_thresh': 25, 'fg_greed_thresh': 75,
    },
}

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def _load_env():
    """Load .env file from repo root into os.environ."""
    env_path = REPO_ROOT / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_credentials() -> tuple[str, str]:
    _load_env()
    key    = os.getenv('ALPACA_API_KEY', '')
    secret = os.getenv('ALPACA_SECRET_KEY', '')
    if not key or not secret:
        raise ValueError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY.\n"
            "Add them to the .env file at the project root."
        )
    return key, secret


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def append_trade_log(entry: dict):
    log = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)
    log.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Market data + indicators
# ---------------------------------------------------------------------------
def fetch_and_prepare(ticker: str, cfg: dict) -> AggressiveHybridV6 | None:
    """Download 260 days of data and run V6 indicators. Returns trader object."""
    end   = datetime.now()
    start = end - timedelta(days=390)   # extra buffer for indicator warmup

    # Crypto: Alpaca uses 'BTC/USD' but yfinance requires 'BTC-USD'
    yf_ticker = cfg.get('yf_ticker', ticker)

    # Strip non-strategy keys before passing to AggressiveHybridV6
    strategy_cfg = {k: v for k, v in cfg.items()
                    if k not in ('yf_ticker', 'is_crypto', 'entry_days')}

    trader = AggressiveHybridV6(
        ticker=yf_ticker,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        **strategy_cfg,
    )
    if not trader.fetch_data():
        logger.error(f"{ticker}: data fetch failed")
        return None
    trader.prepare_indicators()
    return trader


def latest_bar_signals(trader: AggressiveHybridV6) -> dict:
    """
    Evaluate the most recent completed bar (yesterday's close after hours,
    or today's close if running post-close).
    Returns a dict with all values needed for entry/exit decisions.
    """
    df  = trader.data
    idx = len(df) - 1   # latest bar

    row   = df.iloc[idx]
    prev  = df.iloc[idx - 1]

    ema50  = float(row['EMA50'])
    ema200 = float(row['EMA200'])
    lt     = float(row['LT_Trend'])

    if lt >= 0 and ema50 > ema200:
        trend_regime = 'up'
    elif lt < 0 and ema50 < ema200:
        trend_regime = 'down'
    else:
        trend_regime = 'transition'

    buy_sigs, buy_str, buy_type = trader.generate_buy_signals(idx)

    ema50_slope = float(row['EMA50_Slope'])
    vix         = float(row['VIX'])
    realized_vol = float(row['Realized_Vol'])
    atr         = float(row['ATR'])
    adx         = float(row['ADX'])
    close       = float(row['Close'])

    return {
        'date':          str(df.index[idx].date()),
        'close':         close,
        'atr':           atr,
        'adx':           adx,
        'vix':           vix,
        'realized_vol':  realized_vol,
        'ema50_slope':   ema50_slope,
        'trend_regime':  trend_regime,
        'buy_signals':   buy_sigs,
        'buy_strength':  buy_str,
        'buy_type':      buy_type,
    }


# ---------------------------------------------------------------------------
# Position sizing (matches backtest logic)
# ---------------------------------------------------------------------------
def calc_risk_amount(equity: float, vix: float, adx: float,
                     realized_vol: float, current_dd: float,
                     vol_target: float, adx_thresh: float,
                     dd_reduce: float) -> float:
    max_vol_scale = 1.2 if (adx > adx_thresh + 3 and vix < 20) else 1.0
    vol_scale = float(np.clip(vol_target / max(realized_vol, 0.05), 0.4, max_vol_scale))

    base_risk = equity * (0.011 if (adx > adx_thresh + 3 and vix < 20) else 0.010)
    if vix < 15:
        base_risk *= 1.10
    elif vix > 25:
        base_risk *= 0.80
    risk_amount = base_risk * vol_scale
    if current_dd >= dd_reduce:
        risk_amount *= 0.50
    return risk_amount


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------
def round_price(price: float, tick: float = 0.01) -> float:
    return round(round(price / tick) * tick, 2)


def submit_bracket_entry(
    client: TradingClient,
    ticker: str,
    qty: float,
    tp_price: float,
    sl_price: float,
    dry_run: bool,
    is_crypto: bool = False,
) -> str | None:
    """
    Submit a market buy with take-profit limit and stop-loss stop as separate
    orders (managed manually — Alpaca doesn't support full bracket for all assets).
    Crypto orders use GTC time-in-force and support fractional qty.
    Returns the market order ID.
    """
    qty_str = f"{qty:.6f}" if is_crypto else str(int(qty))
    logger.info(f"  ENTRY  {ticker}  qty={qty_str}  TP={tp_price:.2f}  SL={sl_price:.2f}")

    if dry_run:
        logger.info("  [DRY RUN] Order NOT submitted")
        return 'DRY_RUN'

    tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

    try:
        # Market buy
        market_req = MarketOrderRequest(
            symbol=ticker,
            qty=round(qty, 8) if is_crypto else int(qty),
            side=OrderSide.BUY,
            time_in_force=tif,
        )
        order = client.submit_order(market_req)
        order_id = str(order.id)
        logger.info(f"  Market order submitted: {order_id}")

        fill_qty = round(qty, 8) if is_crypto else int(qty)

        # Stop-loss order (GTC)
        sl_req = StopOrderRequest(
            symbol=ticker,
            qty=fill_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round_price(sl_price),
        )
        sl_order = client.submit_order(sl_req)
        logger.info(f"  SL order submitted: {sl_order.id}  @ {sl_price:.2f}")

        # Take-profit limit order (GTC)
        tp_req = LimitOrderRequest(
            symbol=ticker,
            qty=fill_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=round_price(tp_price),
        )
        tp_order = client.submit_order(tp_req)
        logger.info(f"  TP order submitted: {tp_order.id}  @ {tp_price:.2f}")

        return order_id

    except Exception as e:
        logger.error(f"  Order submission failed for {ticker}: {e}")
        return None


def cancel_stop_and_resubmit(
    client: TradingClient,
    sl_order_id: str,
    ticker: str,
    qty: int,
    new_sl: float,
    dry_run: bool,
) -> str | None:
    """Cancel old SL order and place the new trailing stop level."""
    logger.info(f"  TRAIL UPDATE {ticker}  new_SL={new_sl:.2f}")
    if dry_run:
        logger.info("  [DRY RUN] SL update NOT submitted")
        return 'DRY_RUN'
    try:
        client.cancel_order_by_id(sl_order_id)
        sl_req = StopOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round_price(new_sl),
        )
        new_order = client.submit_order(sl_req)
        return str(new_order.id)
    except Exception as e:
        logger.error(f"  SL update failed for {ticker}: {e}")
        return sl_order_id  # keep old ID on failure


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(dry_run: bool = False):
    print(f"\n{'=' * 68}")
    print(f"  ALPACA PAPER TRADER  |  V7 Strategy  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Mode: {'DRY RUN (no orders)' if dry_run else 'LIVE PAPER TRADING'}")
    print(f"{'=' * 68}\n")

    # ── Connect to Alpaca ──────────────────────────────────────────────
    api_key, secret_key = get_credentials()
    client = TradingClient(api_key, secret_key, paper=True)

    account = client.get_account()
    equity   = float(account.equity)
    cash     = float(account.cash)
    peak_eq  = equity   # simplified; track properly in state

    state = load_state()
    peak_eq = state.get('peak_equity', equity)
    peak_eq = max(peak_eq, equity)
    current_dd = (peak_eq - equity) / peak_eq

    print(f"  Account equity : ${equity:,.2f}")
    print(f"  Cash available : ${cash:,.2f}")
    print(f"  Current DD     : {current_dd*100:.1f}%")
    print()

    # ── Get current Alpaca positions ───────────────────────────────────
    alpaca_positions = {p.symbol: p for p in client.get_all_positions()}
    if alpaca_positions:
        print(f"  Open positions: {list(alpaca_positions.keys())}")
    else:
        print("  No open positions on Alpaca")
    print()

    new_state = {'peak_equity': peak_eq}

    for ticker, cfg in ASSET_CONFIGS.items():
        print(f"  {'-' * 64}")
        print(f"  Analysing {ticker}")

        # ── Fetch data + indicators ────────────────────────────────────
        trader = fetch_and_prepare(ticker, cfg)
        if trader is None:
            continue

        bar = latest_bar_signals(trader)
        atr   = bar['atr']
        close = bar['close']

        print(f"  Date={bar['date']}  Close={close:.2f}  ATR={atr:.2f}  "
              f"ADX={bar['adx']:.1f}  VIX={bar['vix']:.1f}  "
              f"Regime={bar['trend_regime']}")

        # ── Position management for existing holdings ──────────────────
        pos_state = state.get(ticker, {})
        alpaca_pos = alpaca_positions.get(ticker)

        if alpaca_pos and pos_state:
            entry_price   = pos_state.get('entry_price', float(alpaca_pos.avg_entry_price))
            trail_active  = pos_state.get('trail_active', False)
            current_sl    = pos_state.get('current_sl', entry_price - cfg['trail_atr'] * atr)
            sl_order_id   = pos_state.get('sl_order_id')
            partial_taken = pos_state.get('partial_taken', False)
            qty           = int(float(alpaca_pos.qty))
            cushion       = cfg.get('trail_cushion', 1.0)

            # Check trail activation
            if not trail_active and close >= entry_price + cushion * atr:
                trail_active = True
                logger.info(f"  Trail ACTIVATED for {ticker}  entry={entry_price:.2f}  "
                            f"threshold={entry_price + cushion * atr:.2f}")

            # Update trailing stop
            if trail_active:
                new_trail = close - cfg['trail_atr'] * atr
                if new_trail > current_sl:
                    logger.info(f"  Trail moved: {current_sl:.2f} -> {new_trail:.2f}")
                    if sl_order_id:
                        sl_order_id = cancel_stop_and_resubmit(
                            client, sl_order_id, ticker, qty, new_trail, dry_run
                        )
                    current_sl = new_trail

            # Check partial TP (close >= entry + partial_tp_mult × ATR)
            partial_tp_price = entry_price + cfg['partial_tp_mult'] * atr
            if not partial_taken and close >= partial_tp_price:
                partial_qty = qty // 2
                logger.info(f"  PARTIAL TP hit for {ticker}  qty={partial_qty}  "
                            f"price~{close:.2f}")
                if not dry_run and partial_qty > 0:
                    try:
                        client.submit_order(MarketOrderRequest(
                            symbol=ticker, qty=partial_qty,
                            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        ))
                        # Move SL to break-even after partial
                        current_sl  = entry_price
                        partial_taken = True
                        if sl_order_id:
                            sl_order_id = cancel_stop_and_resubmit(
                                client, sl_order_id, ticker, qty - partial_qty,
                                entry_price, dry_run
                            )
                    except Exception as e:
                        logger.error(f"  Partial TP order failed: {e}")
                else:
                    logger.info("  [DRY RUN] Partial TP NOT submitted")
                    partial_taken = True  # mark for dry run state tracking

            new_state[ticker] = {
                'entry_price':   entry_price,
                'trail_active':  trail_active,
                'current_sl':    current_sl,
                'sl_order_id':   sl_order_id,
                'partial_taken': partial_taken,
            }
            print(f"  Position: {qty} shares  Entry={entry_price:.2f}  "
                  f"SL={current_sl:.2f}  Trail={'ON' if trail_active else 'OFF'}")

        # ── New entry signal check ─────────────────────────────────────
        elif alpaca_pos is None:
            regime   = bar['trend_regime']
            vix      = bar['vix']
            adx      = bar['adx']
            ema50_sl = bar['ema50_slope']
            sigs     = bar['buy_signals']
            strength = bar['buy_strength']
            sig_type = bar['buy_type']
            is_crypto = cfg.get('is_crypto', False)

            # entry_days gate: e.g. BTC-USD only enters on Fri/Sat bars
            bar_date     = pd.Timestamp(bar['date'])
            entry_days   = cfg.get('entry_days')
            day_ok       = (entry_days is None) or (bar_date.dayofweek in entry_days)

            long_ok = (
                regime == 'up'
                and vix <= 45
                and ema50_sl > -0.01
                and len(sigs) >= 1
                and strength >= cfg.get('min_strength_up', 0.25)
                and current_dd < cfg.get('dd_halt', 0.20)
                and day_ok
            )

            if not day_ok:
                day_name = bar_date.strftime('%A')
                print(f"  {ticker}: no entry — entry_days gate ({day_name} not in allowed days)")

            if long_ok:
                risk_amount = calc_risk_amount(
                    equity=equity,
                    vix=vix, adx=adx,
                    realized_vol=bar['realized_vol'],
                    current_dd=current_dd,
                    vol_target=cfg['vol_target'],
                    adx_thresh=cfg.get('adx_thresh', 25),
                    dd_reduce=cfg.get('dd_reduce', 0.12),
                )
                sl_price  = close - cfg['trail_atr'] * atr
                tp_price  = close + cfg['tp_mult'] * atr
                sl_dist   = max(close - sl_price, 0.01)
                raw_qty   = risk_amount / sl_dist

                if is_crypto:
                    # Fractional crypto: round to 6 decimal places, cap by cash
                    qty = round(raw_qty, 6)
                    max_qty = (cash * 0.95) / close
                    qty = round(min(qty, max_qty), 6)
                else:
                    qty = max(1, int(raw_qty))
                    max_qty = int((cash * 0.95) / close)
                    qty = min(qty, max_qty)

                if qty > 0:
                    qty_str = f"{qty:.6f}" if is_crypto else str(qty)
                    print(f"  SIGNAL  {ticker}  sigs={sigs}  str={strength:.2f}")
                    print(f"          qty={qty_str}  SL={sl_price:.2f}  TP={tp_price:.2f}  "
                          f"risk=${risk_amount:.0f}")

                    order_id = submit_bracket_entry(
                        client, ticker, qty, tp_price, sl_price, dry_run, is_crypto
                    )
                    if order_id:
                        new_state[ticker] = {
                            'entry_price':   close,  # approximate; actual fill at open
                            'trail_active':  False,
                            'current_sl':    sl_price,
                            'sl_order_id':   None,   # updated after fill confirmed
                            'partial_taken': False,
                            'sig_type':      sig_type,
                            'order_id':      order_id,
                        }
                else:
                    print(f"  {ticker}: signal fired but qty=0 (insufficient cash)")
            elif not long_ok and day_ok:
                reason = []
                if regime != 'up':    reason.append(f"regime={regime}")
                if vix > 45:          reason.append(f"VIX={vix:.0f}>45")
                if ema50_sl <= -0.01: reason.append("EMA50 falling")
                if not sigs:          reason.append("no signals")
                elif strength < cfg.get('min_strength_up', 0.25):
                    reason.append(f"strength={strength:.2f} too low")
                print(f"  {ticker}: no entry  ({', '.join(reason) if reason else 'regime not up'})")
        else:
            # Position exists but no state (first run after manual open)
            print(f"  {ticker}: position found but no state — initialising")
            entry_p = float(alpaca_pos.avg_entry_price)
            new_state[ticker] = {
                'entry_price':   entry_p,
                'trail_active':  False,
                'current_sl':    entry_p - cfg['trail_atr'] * atr,
                'sl_order_id':   None,
                'partial_taken': False,
            }

    # ── Save state ─────────────────────────────────────────────────────
    save_state(new_state)
    print(f"\n  State saved -> {STATE_FILE}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"  Run complete  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}  |  DD: {current_dd*100:.1f}%")
    if not dry_run:
        print("  Check Alpaca dashboard: https://app.alpaca.markets")
    print(f"{'=' * 68}\n")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alpaca Paper Trader — V7 Strategy')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate signals but do NOT submit orders to Alpaca')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
