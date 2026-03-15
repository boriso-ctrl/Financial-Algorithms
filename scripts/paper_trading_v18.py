"""
Paper-Trading Harness — Leveraged Alpha v18 (Balanced 3× Tier)
================================================================
Strategy:  L(1%)×3.0 + VT6 + HDDC(1.5/4.0) + E2(85%ZP + 15%CH)
OOS ref:   Sharpe 2.01 | CAGR 46.30% | MaxDD −7.59%

Run daily ~16:30 ET (after market close) via cron / Task Scheduler.

  python scripts/paper_trading_v18.py                  # signals + log (no orders)
  python scripts/paper_trading_v18.py --dry-run        # signals only, no state update
  python scripts/paper_trading_v18.py --execute        # signals + ALPACA ORDERS
  python scripts/paper_trading_v18.py --status         # print current state

Alpaca setup:
  1. Create a free paper-trading account at https://app.alpaca.markets
  2. Copy your API key + secret from the dashboard
  3. Add to .env at the repo root:
       ALPACA_API_KEY=PK...
       ALPACA_SECRET_KEY=...

Outputs:
  paper_trading_v18/signals.csv      — daily signal log
  paper_trading_v18/trade_journal.csv — filled trades
  paper_trading_v18/state.json       — persisted state (positions, equity, DDC)
  paper_trading_v18/config.json      — frozen config snapshot

Kill-switch: pauses new entries if rolling 60-day Sharpe < 1.0.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "paper_trading_v18"
OUT_DIR.mkdir(exist_ok=True)

STATE_FILE   = OUT_DIR / "state.json"
SIGNALS_CSV  = OUT_DIR / "signals.csv"
JOURNAL_CSV  = OUT_DIR / "trade_journal.csv"
CONFIG_FILE  = OUT_DIR / "config.json"

# ── universe ───────────────────────────────────────────────────────────────
SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB", "XLRE"]
BROAD   = ["SPY", "QQQ", "IWM", "EFA"]
SAFE    = ["TLT", "IEF", "GLD", "SHY"]
ALL_TICKERS = SECTORS + BROAD + SAFE

# ── frozen strategy parameters ─────────────────────────────────────────────
# Pair configs: top-5 ShF from IS scan  (all XLP/XLU)
PAIR_CONFIGS = [
    # (leg_a, leg_b, window, entry_z, exit_z)
    ("XLP", "XLU", 63,  2.25, 0.50),
    ("XLP", "XLU", 63,  2.25, 0.75),
    ("XLP", "XLU", 126, 2.25, 0.75),
    ("XLP", "XLU", 63,  2.00, 0.50),
    ("XLP", "XLU", 126, 1.75, 0.50),
]
PAIR_NOTIONAL = 0.06          # per pair, matching OOS_ZP_ShF5_n0.06
N_PAIRS       = len(PAIR_CONFIGS)

# Ensemble blend
ZP_WEIGHT = 0.85
CH_WEIGHT = 0.15

# CrashHedge base leverage
CH_BASE_LEV = 1.0

# Overlays
VT_TARGET  = 0.06            # 6% annualized target vol
VT_LOOKBACK = 63
HDDC_TH1   = -0.015
HDDC_TH2   = -0.04
HDDC_RECOVERY = 0.015

# Leverage
LEV_MULT     = 3.0
LEV_COST_ANN = 0.01          # 1% annual financing cost

# Costs (for accounting only — real fills come from broker)
TX_BPS       = 5
SPREAD_BPS = {
    "SPY": 0.3, "QQQ": 0.5, "IWM": 1.0, "EFA": 1.5,
    "XLK": 1.5, "XLV": 2.0, "XLF": 1.5, "XLE": 2.0,
    "XLI": 2.5, "XLC": 3.0, "XLP": 2.0, "XLU": 2.5,
    "XLB": 3.0, "XLRE": 3.5,
    "TLT": 1.0, "IEF": 1.5, "GLD": 2.0, "SHY": 1.0,
}

# Kill-switch
KILL_LOOKBACK = 60            # trading days
KILL_SHARPE   = 1.0           # pause if rolling Sharpe < this

# Data warmup (longest lookback: 126d pair window + 63d overlay = ~200;
#  use 300 for safety)
WARMUP_DAYS = 400             # calendar days to download before today

# Minimum weight delta to trigger a trade (avoid dust rebalances)
MIN_TRADE_DELTA = 0.005       # 0.5% of portfolio


# ═══════════════════════════════════════════════════════════════════════════
#  CORE INDICATORS  (replicated exactly from v18.py — no drift allowed)
# ═══════════════════════════════════════════════════════════════════════════

def rvol(s: pd.Series, n: int = 21) -> pd.Series:
    return s.pct_change().rolling(n, min_periods=max(10, n // 2)).std() * np.sqrt(252)


def zscore(s: pd.Series, window: int = 63) -> pd.Series:
    m = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().clip(lower=1e-8)
    return (s - m) / sd


# ── pair state machine ────────────────────────────────────────────────────
def pair_position_today(z_series: pd.Series, entry_z: float, exit_z: float
                        ) -> pd.Series:
    """Return position series: +1 long spread, -1 short spread, 0 flat.
    Uses the *lagged* z-score (signal at t, position at t+1)."""
    z = z_series.values
    n = len(z)
    pos = np.zeros(n)
    for i in range(1, n):
        prev = pos[i - 1]
        zi = z[i]
        if np.isnan(zi):
            pos[i] = 0
            continue
        if prev == 0:
            if zi > entry_z:
                pos[i] = -1
            elif zi < -entry_z:
                pos[i] = 1
            else:
                pos[i] = 0
        elif prev > 0:
            pos[i] = 0 if zi > -exit_z else 1
        else:
            pos[i] = 0 if zi < exit_z else -1
    return pd.Series(pos, index=z_series.index)


def compute_pair_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate weight matrix for all 5 ZP pairs."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for leg_a, leg_b, window, ez_in, ez_out in PAIR_CONFIGS:
        if leg_a not in prices.columns or leg_b not in prices.columns:
            continue
        spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
        z = zscore(spread, window)
        pos = pair_position_today(z, ez_in, ez_out)
        notional = PAIR_NOTIONAL
        w[leg_a] += pos * notional
        w[leg_b] += -pos * notional
    return w


# ── crash-hedge regime weights ────────────────────────────────────────────
def compute_crash_hedge_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Replicate strat_crash_hedged() from v18 exactly."""
    qqq = prices["QQQ"]
    v20 = rvol(qqq, 20)
    va = v20.rolling(120, min_periods=30).mean()

    normal    = v20 < va * 1.2
    elevated  = (v20 >= va * 1.2) & (v20 < va * 1.8)
    crisis    = v20 >= va * 1.8
    recovery  = elevated & (v20 < v20.shift(5)) & (qqq > qqq.rolling(10).min())

    bl = CH_BASE_LEV
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # QQQ
    if "QQQ" in w.columns:
        w.loc[normal,   "QQQ"] = bl * 0.7
        w.loc[elevated, "QQQ"] = bl * 0.3
        w.loc[crisis,   "QQQ"] = 0.0
        w.loc[recovery, "QQQ"] = bl * 0.8
    # SPY
    if "SPY" in w.columns:
        w.loc[normal,   "SPY"] = bl * 0.3
        w.loc[elevated, "SPY"] = bl * 0.1
        w.loc[crisis,   "SPY"] = -0.3
        w.loc[recovery, "SPY"] = bl * 0.4
    # IWM
    if "IWM" in w.columns:
        w.loc[elevated, "IWM"] = -0.2
        w.loc[crisis,   "IWM"] = 0.0
        w.loc[recovery, "IWM"] = 0.0
    # GLD
    if "GLD" in w.columns:
        w.loc[elevated, "GLD"] = 0.15
        w.loc[crisis,   "GLD"] = 0.3
    # TLT
    if "TLT" in w.columns:
        w.loc[elevated, "TLT"] = 0.0
        w.loc[crisis,   "TLT"] = 0.2

    return w


# ── overlays (return-stream level) ───────────────────────────────────────

def vol_target_scale(returns: pd.Series) -> pd.Series:
    """Compute the VT multiplier (not applied yet — we need it for position sizing)."""
    realized = returns.rolling(VT_LOOKBACK, min_periods=20).std() * np.sqrt(252)
    realized = realized.clip(lower=0.005)
    scale = (VT_TARGET / realized).clip(lower=0.2, upper=5.0)
    return scale.shift(1).fillna(1.0)


def hddc_scale(returns: pd.Series) -> pd.Series:
    """Compute hierarchical DDC scale factor series (lagged)."""
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    scale = pd.Series(1.0, index=returns.index)
    for i in range(2, len(scale)):
        ddi = dd.iloc[i - 1]
        if ddi < HDDC_TH2:
            scale.iloc[i] = 0.15
        elif ddi < HDDC_TH1:
            t = (ddi - HDDC_TH1) / (HDDC_TH2 - HDDC_TH1)
            scale.iloc[i] = max(0.15, 1.0 - 0.85 * t)
        elif scale.iloc[i - 1] < 1.0:
            scale.iloc[i] = min(1.0, scale.iloc[i - 1] + HDDC_RECOVERY)
        else:
            scale.iloc[i] = 1.0
    return scale


# ═══════════════════════════════════════════════════════════════════════════
#  FULL SIGNAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(prices: pd.DataFrame) -> dict:
    """
    Run the complete signal pipeline on *historical* prices up to today.

    Returns dict with:
      target_weights : dict[str, float]   — per-ticker target weight (today)
      raw_weights    : dict[str, float]   — pre-overlay ensemble weights
      vt_scale       : float              — vol-target multiplier
      hddc_scale     : float              — DDC multiplier
      lev_mult       : float              — leverage
      regime         : str                — CrashHedge regime label
      kill_switch    : bool               — True → pause trading
      rolling_sharpe : float | None
      pair_states    : dict               — per-pair position (-1/0/+1)
      equity_curve   : pd.Series          — simulated equity (for DDC)
    """
    # ── 1. Pair weights ──
    zp_w = compute_pair_weights(prices)

    # ── 2. CrashHedge weights ──
    ch_w = compute_crash_hedge_weights(prices)

    # ── 3. Ensemble raw weights (weight-space blend) ──
    #    v18 actually blends *return streams*, not weight matrices.
    #    But to generate per-ticker target weights we need to do both.
    #    We'll compute return-stream for overlay calibration, then
    #    scale the weight-space blend by the overlay factors.

    # Return stream for overlays  (matches v18 backtest logic)
    ret = prices.pct_change().fillna(0)

    # Pair return stream  (unshifted weights × returns, then shift for lag)
    zp_shifted = zp_w.shift(1).fillna(0)
    zp_ret = (zp_shifted * ret).sum(axis=1)

    # CrashHedge return stream
    ch_shifted = ch_w.shift(1).fillna(0)
    ch_ret = (ch_shifted * ret).sum(axis=1)

    # Ensemble return stream
    ens_ret = ZP_WEIGHT * zp_ret + CH_WEIGHT * ch_ret

    # ── 4. Vol-target overlay ──
    vt = vol_target_scale(ens_ret)

    # ── 5. Apply VT to get post-VT returns, then compute HDDC ──
    post_vt_ret = ens_ret * vt
    hddc = hddc_scale(post_vt_ret)

    # Combined overlay factor for today
    today_vt   = float(vt.iloc[-1])
    today_hddc = float(hddc.iloc[-1])
    overlay    = today_vt * today_hddc

    # ── 6. Leverage ──
    total_scale = overlay * LEV_MULT

    # ── 7. Compute today's target weights ──
    # Ensemble weight = ZP_WT * zp_w + CH_WT * ch_w  (today's row)
    today_ens_w = ZP_WEIGHT * zp_w.iloc[-1] + CH_WEIGHT * ch_w.iloc[-1]
    today_target = today_ens_w * total_scale

    # ── 8. Regime label ──
    qqq = prices["QQQ"]
    v20 = rvol(qqq, 20)
    va  = v20.rolling(120, min_periods=30).mean()
    ratio = v20.iloc[-1] / va.iloc[-1] if va.iloc[-1] > 0 else 1.0
    if ratio < 1.2:
        regime = "Normal"
    elif ratio < 1.8:
        if (v20.iloc[-1] < v20.iloc[-6] if len(v20) > 5 else False) and \
           (qqq.iloc[-1] > qqq.rolling(10).min().iloc[-1]):
            regime = "Recovery"
        else:
            regime = "Elevated"
    else:
        regime = "Crisis"

    # ── 9. Pair states ──
    pair_states = {}
    for leg_a, leg_b, window, ez_in, ez_out in PAIR_CONFIGS:
        spread = np.log(prices[leg_a]) - np.log(prices[leg_b])
        z = zscore(spread, window)
        pos = pair_position_today(z, ez_in, ez_out)
        label = f"{leg_a}/{leg_b}_w{window}_e{ez_in}/x{ez_out}"
        pair_states[label] = int(pos.iloc[-1])

    # ── 10. Kill-switch check ──
    # Use the fully overlaid + leveraged return stream
    lev_ret_stream = post_vt_ret * hddc * LEV_MULT - (LEV_MULT - 1) * LEV_COST_ANN / 252
    rolling_sh = None
    kill = False
    if len(lev_ret_stream) >= KILL_LOOKBACK:
        recent = lev_ret_stream.iloc[-KILL_LOOKBACK:]
        if recent.std() > 1e-10:
            rolling_sh = float(recent.mean() / recent.std() * np.sqrt(252))
            if rolling_sh < KILL_SHARPE:
                kill = True

    # ── 11. Simulated equity (for dashboard / logging) ──
    sim_eq = 100_000 * (1 + lev_ret_stream).cumprod()

    return {
        "target_weights": {t: round(float(today_target[t]), 6) for t in ALL_TICKERS
                           if abs(today_target.get(t, 0)) > 1e-8},
        "raw_weights":    {t: round(float(today_ens_w[t]), 6) for t in ALL_TICKERS
                           if abs(today_ens_w.get(t, 0)) > 1e-8},
        "vt_scale":       round(today_vt, 4),
        "hddc_scale":     round(today_hddc, 4),
        "lev_mult":       LEV_MULT,
        "overlay":        round(overlay, 4),
        "total_scale":    round(total_scale, 4),
        "regime":         regime,
        "kill_switch":    kill,
        "rolling_sharpe": round(rolling_sh, 4) if rolling_sh is not None else None,
        "pair_states":    pair_states,
        "equity_curve":   sim_eq,
        "lev_ret_stream": lev_ret_stream,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "equity": 100_000, "history": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_signal_log(row: dict):
    df = pd.DataFrame([row])
    header = not SIGNALS_CSV.exists()
    df.to_csv(SIGNALS_CSV, mode="a", header=header, index=False)


def append_journal(row: dict):
    df = pd.DataFrame([row])
    header = not JOURNAL_CSV.exists()
    df.to_csv(JOURNAL_CSV, mode="a", header=header, index=False)


def save_config_snapshot():
    """Freeze today's config so we can always prove no parameter was changed."""
    cfg = {
        "pair_configs": [list(c) for c in PAIR_CONFIGS],
        "pair_notional": PAIR_NOTIONAL,
        "zp_weight": ZP_WEIGHT,
        "ch_weight": CH_WEIGHT,
        "vt_target": VT_TARGET,
        "vt_lookback": VT_LOOKBACK,
        "hddc_th1": HDDC_TH1,
        "hddc_th2": HDDC_TH2,
        "hddc_recovery": HDDC_RECOVERY,
        "lev_mult": LEV_MULT,
        "lev_cost_ann": LEV_COST_ANN,
        "kill_lookback": KILL_LOOKBACK,
        "kill_sharpe": KILL_SHARPE,
        "frozen_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Config frozen -> {CONFIG_FILE}")
    else:
        print(f"  Config already frozen (not overwritten).")


# ═══════════════════════════════════════════════════════════════════════════
#  ALPACA INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def _load_env():
    """Load .env file from repo root into os.environ."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_alpaca_client():
    """Return an authenticated Alpaca TradingClient (paper mode)."""
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        print("\n  ERROR: alpaca-py not installed.")
        print("  Run:  pip install alpaca-py")
        sys.exit(1)

    _load_env()
    key    = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret or key.startswith("replace"):
        print("\n  ERROR: Missing ALPACA_API_KEY / ALPACA_SECRET_KEY.")
        print("  Add them to .env at the repo root:")
        print("    ALPACA_API_KEY=PK...")
        print("    ALPACA_SECRET_KEY=...")
        sys.exit(1)

    client = TradingClient(key, secret, paper=True)
    # Verify connection immediately
    try:
        client.get_account()
    except Exception as e:
        print(f"\n  ERROR: Alpaca connection failed: {e}")
        print("  Check your API key/secret and try again.")
        sys.exit(1)

    return client


def alpaca_get_account_info(client) -> dict:
    """Fetch account equity, cash, buying power."""
    acct = client.get_account()
    return {
        "equity":       float(acct.equity),
        "cash":         float(acct.cash),
        "buying_power":  float(acct.buying_power),
        "status":       str(acct.status),
    }


def alpaca_get_positions(client) -> dict[str, dict]:
    """Return {ticker: {qty, market_value, avg_entry, unrealized_pl}}."""
    positions = {}
    for p in client.get_all_positions():
        positions[p.symbol] = {
            "qty":            int(float(p.qty)),
            "market_value":   float(p.market_value),
            "avg_entry":      float(p.avg_entry_price),
            "unrealized_pl":  float(p.unrealized_pl),
            "side":           str(p.side),
        }
    return positions


def alpaca_execute_rebalance(client, target_weights: dict[str, float],
                             account_equity: float) -> list[dict]:
    """
    Rebalance the Alpaca paper account to match target_weights.

    For each ticker:
      target_value = target_weight × account_equity
      delta_value  = target_value − current_market_value
      if |delta_value| > MIN_TRADE_DELTA × equity → submit order

    Returns list of order dicts for logging.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    current_positions = alpaca_get_positions(client)
    orders_placed = []
    all_tickers = set(list(target_weights.keys()) + list(current_positions.keys()))

    for ticker in sorted(all_tickers):
        target_w  = target_weights.get(ticker, 0.0)
        target_val = target_w * account_equity

        cur = current_positions.get(ticker)
        current_val = cur["market_value"] if cur else 0.0
        # Alpaca reports market_value as positive for longs, negative for shorts
        # but qty is always positive with a side field
        if cur and cur["side"] == "short":
            current_val = -abs(current_val)

        delta_val = target_val - current_val

        # Skip dust rebalances
        if abs(delta_val) < MIN_TRADE_DELTA * account_equity:
            continue

        # Get approximate share price from the position or fetch last close
        if cur:
            approx_price = abs(cur["market_value"]) / max(cur["qty"], 1)
        else:
            # Fetch latest price via yfinance (single ticker, fast)
            try:
                tick_data = yf.download(ticker, period="2d", progress=False)
                approx_price = float(tick_data["Close"].iloc[-1])
            except Exception:
                logger.warning(f"  Cannot get price for {ticker}, skipping")
                continue

        if approx_price < 0.01:
            continue

        qty = int(abs(delta_val) / approx_price)
        if qty < 1:
            continue

        side = OrderSide.BUY if delta_val > 0 else OrderSide.SELL

        logger.info(f"  ORDER  {side.name:4s}  {ticker:6s}  qty={qty:5d}  "
                    f"target_val=${target_val:+,.0f}  delta=${delta_val:+,.0f}")

        try:
            order = client.submit_order(
                MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            )
            orders_placed.append({
                "ticker": ticker,
                "side": side.name,
                "qty": qty,
                "order_id": str(order.id),
                "status": str(order.status),
                "target_val": round(target_val, 2),
                "delta_val": round(delta_val, 2),
            })
            logger.info(f"  → Order {order.id} submitted ({order.status})")
        except Exception as e:
            logger.error(f"  → Order FAILED for {ticker}: {e}")
            orders_placed.append({
                "ticker": ticker,
                "side": side.name,
                "qty": qty,
                "order_id": None,
                "status": f"FAILED: {e}",
                "target_val": round(target_val, 2),
                "delta_val": round(delta_val, 2),
            })

    return orders_placed


def print_alpaca_status(acct_info: dict, positions: dict):
    """Print Alpaca account summary."""
    print(f"\n  {THIN}")
    print(f"  ALPACA PAPER ACCOUNT:")
    print(f"    Status:       {acct_info['status']}")
    print(f"    Equity:       ${acct_info['equity']:,.2f}")
    print(f"    Cash:         ${acct_info['cash']:,.2f}")
    print(f"    Buying power: ${acct_info['buying_power']:,.2f}")
    if positions:
        print(f"    Positions:")
        for t, p in sorted(positions.items()):
            print(f"      {t:6s}  qty={p['qty']:5d}  val=${p['market_value']:>10,.2f}  "
                  f"P&L=${p['unrealized_pl']:>+8,.2f}")
    else:
        print(f"    Positions:    (none)")


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

SEP  = "=" * 78
THIN = "-" * 78

def print_report(today: str, sig: dict, state: dict):
    print(f"\n{SEP}")
    print(f"  PAPER TRADING REPORT -- {today}")
    print(f"  Strategy: L(1%)x3 + VT6 + HDDC(1.5/4.0) + E2(85%ZP+15%CH)")
    print(SEP)

    # Regime
    print(f"\n  Regime:          {sig['regime']}")
    print(f"  VT scale:        {sig['vt_scale']:.4f}")
    print(f"  HDDC scale:      {sig['hddc_scale']:.4f}")
    print(f"  Combined overlay: {sig['overlay']:.4f}")
    print(f"  Total scale (x{LEV_MULT:.0f} lev): {sig['total_scale']:.4f}")

    # Kill-switch
    if sig["rolling_sharpe"] is not None:
        status = "** KILL — PAUSED **" if sig["kill_switch"] else "OK"
        print(f"  Rolling {KILL_LOOKBACK}d Sharpe: {sig['rolling_sharpe']:.4f}  [{status}]")
    else:
        print(f"  Rolling Sharpe:  N/A (insufficient history)")

    # Pair states
    print(f"\n  {THIN}")
    print(f"  PAIR STATES:")
    for label, pos in sig["pair_states"].items():
        tag = {1: "LONG spread", -1: "SHORT spread", 0: "FLAT"}[pos]
        print(f"    {label:40s}  {tag}")

    # Target weights
    print(f"\n  {THIN}")
    print(f"  TARGET WEIGHTS (post-overlay, post-leverage):")
    tw = sig["target_weights"]
    if not tw:
        print(f"    (all flat)")
    else:
        gross = 0.0
        for t in sorted(tw, key=lambda x: abs(tw[x]), reverse=True):
            side = "LONG " if tw[t] > 0 else "SHORT"
            print(f"    {t:6s}  {side}  {tw[t]:+8.4f}  ({tw[t]*100:+6.2f}%)")
            gross += abs(tw[t])
        print(f"    {'':6s}  GROSS  {gross:8.4f}  ({gross*100:6.2f}%)")

    # Deltas vs previous
    prev_pos = state.get("positions", {})
    print(f"\n  {THIN}")
    print(f"  TRADE DELTAS (vs previous close):")
    all_tickers_set = set(list(tw.keys()) + list(prev_pos.keys()))
    any_delta = False
    for t in sorted(all_tickers_set):
        old = prev_pos.get(t, 0.0)
        new = tw.get(t, 0.0)
        delta = new - old
        if abs(delta) > 1e-6:
            any_delta = True
            cost_bps = SPREAD_BPS.get(t, 3.0) / 2 + TX_BPS
            est_cost = abs(delta) * cost_bps / 10_000
            print(f"    {t:6s}  {old:+8.4f} -> {new:+8.4f}  d={delta:+8.4f}  "
                  f"~{est_cost*10000:.1f}bps")
    if not any_delta:
        print(f"    (no changes)")

    # Simulated equity
    eq = sig["equity_curve"]
    print(f"\n  {THIN}")
    print(f"  SIMULATED EQUITY (from data start): ${eq.iloc[-1]:,.0f}")
    if len(eq) > 1:
        peak = eq.cummax().iloc[-1]
        dd = (eq.iloc[-1] - peak) / peak
        print(f"  Current drawdown: {dd:.2%}")

    print(f"\n{SEP}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v18 Paper Trading Harness")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute signals only, don't update state")
    parser.add_argument("--execute", action="store_true",
                        help="Submit rebalance orders to Alpaca paper account")
    parser.add_argument("--status", action="store_true",
                        help="Print current state and exit")
    args = parser.parse_args()

    # ── Status mode ──
    if args.status:
        state = load_state()
        print(json.dumps(state, indent=2, default=str))
        return

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'=' * 78}")
    print(f"  v18 Paper Trading Harness -- {today}")
    print(f"{'=' * 78}")

    # ── 0. Pre-flight: validate Alpaca credentials if --execute ──
    alpaca_client = None
    acct_info = None
    if args.execute:
        print(f"\n  Connecting to Alpaca (paper) ...")
        alpaca_client = get_alpaca_client()
        acct_info = alpaca_get_account_info(alpaca_client)
        alpaca_positions = alpaca_get_positions(alpaca_client)
        print_alpaca_status(acct_info, alpaca_positions)

    # ── 1. Download prices ──
    start_date = (datetime.now() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    end_date   = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n  Downloading {len(ALL_TICKERS)} tickers  [{start_date} -> {end_date}] ...")
    raw = yf.download(ALL_TICKERS, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all").ffill()
    print(f"  Got {len(prices)} trading days, {len(prices.columns)} tickers")

    if len(prices) < 200:
        print("  ERROR: Insufficient history. Need >=200 days.")
        sys.exit(1)

    # ── 2. Run pipeline ──
    print(f"\n  Running signal pipeline ...")
    sig = run_pipeline(prices)

    # ── 3. Load state ──
    state = load_state()
    save_config_snapshot()

    # ── 4. Print report ──
    print_report(today, sig, state)

    if args.dry_run:
        print("  [DRY RUN] -- state not updated.\n")
        return

    # ── 6. Log signal ──
    log_row = {
        "date": today,
        "regime": sig["regime"],
        "vt_scale": sig["vt_scale"],
        "hddc_scale": sig["hddc_scale"],
        "overlay": sig["overlay"],
        "total_scale": sig["total_scale"],
        "rolling_sharpe": sig["rolling_sharpe"],
        "kill_switch": sig["kill_switch"],
        "gross_exposure": sum(abs(v) for v in sig["target_weights"].values()),
    }
    # Add per-ticker weights
    for t in ALL_TICKERS:
        log_row[f"w_{t}"] = sig["target_weights"].get(t, 0.0)
    # Add pair states
    for label, pos in sig["pair_states"].items():
        log_row[f"pair_{label}"] = pos
    append_signal_log(log_row)

    # ── 6. Compute and log trade deltas ──
    prev_pos = state.get("positions", {})
    new_pos  = sig["target_weights"]
    all_t = set(list(new_pos.keys()) + list(prev_pos.keys()))
    for t in sorted(all_t):
        old_w = prev_pos.get(t, 0.0)
        new_w = new_pos.get(t, 0.0)
        delta = new_w - old_w
        if abs(delta) > 1e-6:
            append_journal({
                "date": today,
                "ticker": t,
                "old_weight": round(old_w, 6),
                "new_weight": round(new_w, 6),
                "delta": round(delta, 6),
                "side": "BUY" if delta > 0 else "SELL",
                "regime": sig["regime"],
                "vt_scale": sig["vt_scale"],
                "hddc_scale": sig["hddc_scale"],
            })

    # ── 7. Execute on Alpaca ──
    if args.execute and alpaca_client and acct_info:
        print(f"\n  {THIN}")
        print(f"  ALPACA ORDER EXECUTION:")

        if sig["kill_switch"]:
            # Kill-switch: flatten all positions
            print(f"  *** KILL SWITCH ACTIVE -- closing all positions ***")
            orders = alpaca_execute_rebalance(
                alpaca_client, {}, acct_info["equity"]
            )
        else:
            orders = alpaca_execute_rebalance(
                alpaca_client, new_pos, acct_info["equity"]
            )

        if orders:
            for o in orders:
                status_tag = "OK" if o["order_id"] else "FAIL"
                print(f"    [{status_tag}] {o['side']:4s} {o['ticker']:6s} "
                      f"qty={o['qty']}  ${o['delta_val']:+,.0f}")
                # Also log each Alpaca order to the journal
                append_journal({
                    "date": today,
                    "ticker": o["ticker"],
                    "old_weight": round(prev_pos.get(o["ticker"], 0.0), 6),
                    "new_weight": round(new_pos.get(o["ticker"], 0.0), 6),
                    "delta": round(
                        new_pos.get(o["ticker"], 0.0) -
                        prev_pos.get(o["ticker"], 0.0), 6),
                    "side": o["side"],
                    "qty": o["qty"],
                    "order_id": o["order_id"],
                    "order_status": o["status"],
                    "regime": sig["regime"],
                    "vt_scale": sig["vt_scale"],
                    "hddc_scale": sig["hddc_scale"],
                    "source": "alpaca",
                })
        else:
            print(f"    (no orders needed -- positions already aligned)")

        # Refresh account after trades
        acct_after = alpaca_get_account_info(alpaca_client)
        pos_after  = alpaca_get_positions(alpaca_client)
        print(f"\n  Post-trade equity: ${acct_after['equity']:,.2f}")
        state["alpaca_equity"] = acct_after["equity"]
        state["alpaca_positions"] = {
            t: p["market_value"] for t, p in pos_after.items()
        }

    # ── 8. Update state ──
    state["positions"] = new_pos
    state["last_run"]  = today
    state["regime"]    = sig["regime"]
    state["rolling_sharpe"] = sig["rolling_sharpe"]
    state["kill_switch"]    = sig["kill_switch"]
    # Append daily return for equity tracking
    if "daily_returns" not in state:
        state["daily_returns"] = []
    state["daily_returns"].append({
        "date": today,
        "sim_equity": round(float(sig["equity_curve"].iloc[-1]), 2),
    })
    save_state(state)
    print(f"  State saved -> {STATE_FILE}")
    print(f"  Signal log  -> {SIGNALS_CSV}")
    print(f"  Trade log   -> {JOURNAL_CSV}")

    # ── 9. Kill-switch warning ──
    if sig["kill_switch"]:
        print(f"\n  *** KILL SWITCH ACTIVE ***")
        print(f"  Rolling {KILL_LOOKBACK}d Sharpe = {sig['rolling_sharpe']:.4f} < {KILL_SHARPE}")
        print(f"  DO NOT place new entries until Sharpe recovers.\n")

    print("  Done.\n")


if __name__ == "__main__":
    main()
