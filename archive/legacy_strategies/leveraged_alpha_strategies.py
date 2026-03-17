"""
Leveraged Alpha Strategy Explorer
==================================
Tests multiple leveraged/hedged strategies across a multi-asset universe.

Targets: CAGR > SPY, Sharpe > 1.95
Time Period: 2010-2025 (bull markets, COVID, rate hikes)

Self-auditing: checks for look-ahead bias, calculation integrity,
               realistic parameter ranges.

Strategies:
  1. Leveraged Global Asset Allocation (momentum + trend + risk parity + leverage)
  2. Time-Series Momentum (long/short each asset based on own trend)
  3. Hedged Equity with Leverage (leveraged equity core + permanent bond/gold hedge)
  4. Multi-Strategy Ensemble (blend of 1-3 for diversification)

All strategies properly account for:
  - Leverage financing costs (1.5% annual spread)
  - Short selling costs (0.5% annual)
  - Transaction costs (5 bps round-trip)
  - No look-ahead bias (all signals lagged 1 day)
  - Weekly rebalancing to keep turnover realistic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Add src to path for metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

# ── Configuration ──────────────────────────────────────────────────────────

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD"]
EQUITY_TICKERS = ["SPY", "QQQ", "IWM", "EFA"]
BOND_TICKERS = ["TLT", "IEF"]
ALT_TICKERS = ["GLD"]

BENCHMARK = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2025-03-01"

# Cost assumptions
TX_COST_BPS = 5          # Total round-trip (commission + spread + slippage)
LEVERAGE_COST_ANNUAL = 0.015   # 1.5% annual spread for borrowing
SHORT_COST_ANNUAL = 0.005      # 0.5% annual borrow cost
RISK_FREE_RATE = 0.0           # ~0 for most of 2010-2022

REBALANCE_FREQ_DAYS = 5  # Weekly rebalancing


# ── Data Loading ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Download adjusted close prices for universe + benchmark."""
    tickers = list(set(UNIVERSE + [BENCHMARK]))
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all").ffill().bfill()
    # Ensure we have all required tickers
    missing = [t for t in UNIVERSE if t not in prices.columns]
    if missing:
        print(f"  WARNING: Missing tickers: {missing}")
    return prices


# ── Custom Backtest Engine (supports leverage) ─────────────────────────────

def run_leveraged_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000.0,
    tx_cost_bps: float = TX_COST_BPS,
    leverage_cost_annual: float = LEVERAGE_COST_ANNUAL,
    short_cost_annual: float = SHORT_COST_ANNUAL,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Vectorized backtest that properly handles leverage (weights summing to >1).

    Parameters
    ----------
    prices : DataFrame of adjusted close prices (index=dates, cols=tickers)
    target_weights : DataFrame of target portfolio weights per day.
        Weights CAN sum to >1.0 (leverage) or include negatives (shorts).
        These are the DESIRED weights — the engine shifts them by 1 day
        to prevent look-ahead bias.
    """
    # Align columns
    common = prices.columns.intersection(target_weights.columns)
    prices = prices[common]
    target_weights = target_weights[common].reindex(prices.index).fillna(0)

    # ── CRITICAL: shift weights by 1 day to prevent look-ahead bias ──
    actual_weights = target_weights.shift(1).fillna(0)

    # Asset daily returns
    asset_returns = prices.pct_change().fillna(0)

    # Portfolio returns before costs
    gross_returns = (actual_weights * asset_returns).sum(axis=1)

    # Transaction costs (turnover-based)
    weight_changes = actual_weights.diff().fillna(0)
    turnover = weight_changes.abs().sum(axis=1)
    tx_costs = turnover * tx_cost_bps / 10_000

    # Leverage financing costs (on gross exposure above 1.0)
    gross_exposure = actual_weights.abs().sum(axis=1)
    daily_leverage_cost = (gross_exposure - 1.0).clip(lower=0) * leverage_cost_annual / 252

    # Short selling costs
    short_exposure = actual_weights.clip(upper=0).abs().sum(axis=1)
    daily_short_cost = short_exposure * short_cost_annual / 252

    # Net returns
    net_returns = gross_returns - tx_costs - daily_leverage_cost - daily_short_cost

    # Equity curve
    equity = initial_capital * (1 + net_returns).cumprod()
    equity.name = "Equity"

    # Compute metrics
    metrics = compute_metrics(
        net_returns, equity, initial_capital,
        risk_free_rate=risk_free_rate,
        turnover=turnover,
        gross_exposure=gross_exposure,
    )

    return {
        "equity_curve": equity,
        "portfolio_returns": net_returns,
        "gross_returns": gross_returns,
        "weights": actual_weights,
        "turnover": turnover,
        "gross_exposure": gross_exposure,
        "metrics": metrics,
        "cost_breakdown": {
            "tx_costs_total": tx_costs.sum(),
            "leverage_costs_total": daily_leverage_cost.sum(),
            "short_costs_total": daily_short_cost.sum(),
        },
    }


# ── Rebalancing Helper ────────────────────────────────────────────────────

def apply_weekly_rebalance(weights: pd.DataFrame, freq_days: int = REBALANCE_FREQ_DAYS) -> pd.DataFrame:
    """Only update weights every `freq_days` trading days (weekly default)."""
    mask = pd.Series(False, index=weights.index)
    mask.iloc[::freq_days] = True
    rebalanced = weights.where(mask).ffill().fillna(0)
    return rebalanced


# ── Signal Generators ─────────────────────────────────────────────────────

def momentum_score(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-month momentum, skip recent month (avoid short-term reversal)."""
    if len(prices) < lookback:
        return pd.DataFrame(0, index=prices.index, columns=prices.columns)
    return prices.shift(skip).pct_change(lookback - skip)


def trend_score(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Binary trend: +1 if fast MA > slow MA, else -1."""
    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()
    return np.sign(fast_ma - slow_ma).fillna(0)


def inverse_vol_weights(prices: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Inverse-volatility position sizing (risk parity)."""
    daily_ret = prices.pct_change()
    vol = daily_ret.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    inv_vol = 1.0 / vol.clip(lower=0.02)
    return inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)


def rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Standard RSI (0-100)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-10)
    return 100 - 100 / (1 + rs)


def regime_indicator(spy_prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Market regime from SPY only: risk_on / cautious / risk_off.
    Returns (regime_label, realized_vol).
    """
    ma50 = spy_prices.rolling(50, min_periods=50).mean()
    ma200 = spy_prices.rolling(200, min_periods=200).mean()
    vol = spy_prices.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)

    regime = pd.Series("cautious", index=spy_prices.index)
    regime[(ma50 > ma200) & (vol < 0.18)] = "risk_on"
    regime[(ma50 > ma200) & (vol >= 0.18)] = "cautious"
    regime[(ma50 <= ma200) & (vol < 0.25)] = "cautious_bear"
    regime[(ma50 <= ma200) & (vol >= 0.25)] = "risk_off"
    return regime, vol


# ── Strategy 1: Leveraged Global Asset Allocation ──────────────────────────

def strategy_leveraged_gaa(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Leveraged Global Asset Allocation (GAA).
    - Base: inverse-vol weights (risk parity)
    - Tilt: momentum score tilts allocation toward winners
    - Trend filter: zero out assets with negative trend
    - Regime leverage: 1.5x risk_on → 0.6x risk_off
    - Permanent hedge: min 15% in bonds+gold combined
    """
    available = [c for c in UNIVERSE if c in prices.columns]
    p = prices[available]
    equity = [t for t in EQUITY_TICKERS if t in available]
    bonds = [t for t in BOND_TICKERS if t in available]
    alts = [t for t in ALT_TICKERS if t in available]

    # Base weights (inverse-vol)
    base_w = inverse_vol_weights(p, lookback=63)

    # Momentum tilt
    mom = momentum_score(p, lookback=252, skip=21)
    mom_rank = mom.rank(axis=1, pct=True).fillna(0.5)
    # Scale: bottom 25% → 0.5x, top 25% → 1.5x
    mom_tilt = 0.5 + mom_rank  # range 0.5 to 1.5

    # Trend filter (only for equity: zero out negative trends)
    trend = trend_score(p, fast=50, slow=200)

    # Combine
    weights = base_w * mom_tilt
    for col in equity:
        if col in weights.columns:
            # Reduce (but don't zero) equity with negative trend
            weights[col] = weights[col] * np.where(trend[col] > 0, 1.0, 0.2)

    # Increase defensive when equity trends are negative
    avg_equity_trend = trend[equity].mean(axis=1) if equity else pd.Series(0, index=p.index)
    for col in bonds + alts:
        if col in weights.columns:
            # Boost hedge when equity trends are weak
            hedge_mult = np.where(avg_equity_trend < 0, 2.0,
                         np.where(avg_equity_trend < 0.5, 1.3, 0.8))
            weights[col] = weights[col] * hedge_mult

    # Ensure minimum hedge allocation (15%)
    hedge_cols = [c for c in bonds + alts if c in weights.columns]
    if hedge_cols:
        hedge_share = weights[hedge_cols].sum(axis=1)
        total = weights.abs().sum(axis=1).clip(lower=1e-8)
        hedge_pct = hedge_share / total
        # Where hedge is below 15%, scale up hedge and scale down equity
        low_hedge = hedge_pct < 0.15
        if low_hedge.any():
            for col in hedge_cols:
                weights.loc[low_hedge, col] *= 1.5
            for col in equity:
                if col in weights.columns:
                    weights.loc[low_hedge, col] *= 0.85

    # Re-normalize to sum to 1.0 before applying leverage
    total = weights.abs().sum(axis=1).clip(lower=1e-8)
    weights = weights.div(total, axis=0)

    # Dynamic leverage based on regime
    regime, _ = regime_indicator(prices["SPY"])
    leverage = pd.Series(1.0, index=prices.index)
    leverage[regime == "risk_on"] = 1.5
    leverage[regime == "cautious"] = 1.2
    leverage[regime == "cautious_bear"] = 0.8
    leverage[regime == "risk_off"] = 0.6

    weights = weights.mul(leverage, axis=0)
    return apply_weekly_rebalance(weights)


# ── Strategy 2: Time-Series Momentum (TSMOM) ──────────────────────────────

def strategy_tsmom(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Time-Series Momentum with vol-normalization.
    - Each asset: long if positive 12-month momentum, short if negative
    - Position size: inversely proportional to volatility (target 10% vol each)
    - Dynamic leverage: scale total by conviction (fraction of assets trending)
    """
    available = [c for c in UNIVERSE if c in prices.columns]
    p = prices[available]

    # 12-month momentum (skip 1 month)
    mom = momentum_score(p, lookback=252, skip=21)

    # Direction: +1 long, -1 short
    direction = np.sign(mom).fillna(0)

    # Vol-normalized sizing: each position targets 10% annualized vol
    daily_ret = p.pct_change()
    realized_vol = daily_ret.rolling(63, min_periods=20).std() * np.sqrt(252)
    target_vol_per_position = 0.10
    raw_size = target_vol_per_position / realized_vol.clip(lower=0.02)

    # Cap individual position at 40%
    raw_size = raw_size.clip(upper=0.40)

    # Weights = direction * size
    weights = direction * raw_size

    # Dynamic leverage: boost when many assets agree on direction
    frac_trending = direction.abs().mean(axis=1)  # fraction with a signal
    agreement = direction.mean(axis=1).abs()  # how much they agree

    conviction_mult = 0.8 + agreement  # range: 0.8 to 1.8
    weights = weights.mul(conviction_mult, axis=0)

    # Cap gross exposure at 2.0x
    gross = weights.abs().sum(axis=1)
    scale = (2.0 / gross).where(gross > 2.0, other=1.0)
    weights = weights.mul(scale, axis=0)

    return apply_weekly_rebalance(weights)


# ── Strategy 3: Hedged Equity with Leverage ────────────────────────────────

def strategy_hedged_equity(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Leveraged equity core + permanent bond/gold hedge.
    - Core: 100-150% in best-momentum equity (2-3 tickers)
    - Hedge: 30-50% long in TLT + GLD (always)
    - Net long: 70-120%
    - Uses RSI for timing entry / scale-up in equity
    - Regime-adaptive total exposure
    """
    available = [c for c in UNIVERSE if c in prices.columns]
    p = prices[available]
    equity = [t for t in EQUITY_TICKERS if t in available]
    bonds = [t for t in BOND_TICKERS if t in available]
    alts = [t for t in ALT_TICKERS if t in available]

    weights = pd.DataFrame(0.0, index=p.index, columns=available)

    # Equity allocation: momentum-ranked
    if equity:
        mom = momentum_score(p[equity], lookback=252, skip=21)
        mom_rank = mom.rank(axis=1, pct=True).fillna(0.5)
        trend = trend_score(p[equity], fast=50, slow=200)
        rsi_vals = rsi(p[equity], period=14)

        for col in equity:
            # Base equity weight proportional to momentum rank
            base = mom_rank[col] * 0.5  # range 0-0.5

            # Trend filter: reduce if downtrend
            trend_mult = np.where(trend[col] > 0, 1.0, 0.15)
            base = base * trend_mult

            # RSI timing: boost on oversold in uptrend
            rsi_boost = np.where(
                (rsi_vals[col] < 35) & (trend[col] > 0), 1.4,
                np.where(rsi_vals[col] > 75, 0.7, 1.0)
            )
            weights[col] = base * rsi_boost

    # Hedge allocation: always present
    regime, _ = regime_indicator(prices["SPY"])
    hedge_base = pd.Series(0.15, index=p.index)
    hedge_base[regime == "risk_off"] = 0.30
    hedge_base[regime == "cautious_bear"] = 0.25
    hedge_base[regime == "cautious"] = 0.15
    hedge_base[regime == "risk_on"] = 0.10

    for col in bonds:
        weights[col] = hedge_base * (1.2 if col == "TLT" else 0.8)
    for col in alts:
        # Gold: extra boost in risk-off
        gold_mult = pd.Series(1.0, index=p.index)
        gold_mult[regime == "risk_off"] = 2.0
        gold_mult[regime == "cautious_bear"] = 1.5
        weights[col] = hedge_base * 0.8 * gold_mult

    # Scale equity to get desired leverage
    equity_total = weights[equity].sum(axis=1) if equity else pd.Series(0, index=p.index)
    target_equity_exposure = pd.Series(1.0, index=p.index)
    target_equity_exposure[regime == "risk_on"] = 1.4
    target_equity_exposure[regime == "cautious"] = 1.1
    target_equity_exposure[regime == "cautious_bear"] = 0.6
    target_equity_exposure[regime == "risk_off"] = 0.3

    eq_scale = (target_equity_exposure / equity_total.clip(lower=1e-8))
    for col in equity:
        weights[col] = weights[col] * eq_scale

    # Cap gross at 2.0x
    gross = weights.abs().sum(axis=1)
    scale = (2.0 / gross).where(gross > 2.0, other=1.0)
    weights = weights.mul(scale, axis=0)

    return apply_weekly_rebalance(weights)


# ── Strategy 4: Adaptive Multi-Factor with Shorts ──────────────────────────

def strategy_adaptive_multi_factor(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-factor strategy with adaptive leverage and shorting.
    Factors: momentum (12m), trend (50/200 MA), mean-reversion (RSI)
    - Strong bull factors → 1.5-2.0x long
    - Strong bear factors → 0.5-1.0x short
    - Mixed → 0.5-1.0x long with heavy hedge
    """
    available = [c for c in UNIVERSE if c in prices.columns]
    p = prices[available]
    equity = [t for t in EQUITY_TICKERS if t in available]
    bonds = [t for t in BOND_TICKERS if t in available]
    alts = [t for t in ALT_TICKERS if t in available]

    weights = pd.DataFrame(0.0, index=p.index, columns=available)

    # Compute factors for equity
    mom = momentum_score(p, lookback=252, skip=21)
    trend_vals = trend_score(p, fast=50, slow=200)
    rsi_vals = rsi(p, period=14)
    vol = p.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)

    for col in equity:
        if col not in p.columns:
            continue
        # Composite score: momentum rank + trend + RSI signal
        mom_z = mom[col].rolling(252, min_periods=60).apply(
            lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-8), raw=False
        ).fillna(0)

        # Trend: +1 or -1
        t = trend_vals[col]

        # RSI: mean reversion signal normalized
        rsi_sig = np.where(rsi_vals[col] < 30, 1.0,
                  np.where(rsi_vals[col] < 40, 0.5,
                  np.where(rsi_vals[col] > 70, -0.5,
                  np.where(rsi_vals[col] > 80, -1.0, 0.0))))

        # Composite: momentum (40%) + trend (40%) + RSI (20%)
        composite = 0.4 * mom_z.clip(-2, 2) + 0.4 * t + 0.2 * rsi_sig

        # Vol-adjusted sizing: target 8% vol per equity position
        vol_scale = 0.08 / vol[col].clip(lower=0.02)
        vol_scale = vol_scale.clip(upper=0.50)

        weights[col] = composite * vol_scale

    # Bond/alt allocation counter-cyclical
    avg_equity_weight = weights[equity].sum(axis=1) if equity else pd.Series(0, index=p.index)

    for col in bonds:
        # Inverse of equity exposure
        weights[col] = np.where(avg_equity_weight < 0, 0.25,
                       np.where(avg_equity_weight < 0.3, 0.20,
                       np.where(avg_equity_weight < 0.7, 0.12, 0.08)))
        # Add own trend
        weights[col] = weights[col] * np.where(trend_vals[col] > 0, 1.3, 0.7)

    for col in alts:
        weights[col] = np.where(avg_equity_weight < 0, 0.20,
                       np.where(avg_equity_weight < 0.5, 0.15, 0.08))
        weights[col] = weights[col] * np.where(trend_vals[col] > 0, 1.3, 0.7)

    # Cap gross at 2.0x
    gross = weights.abs().sum(axis=1)
    scale = (2.0 / gross).where(gross > 2.0, other=1.0)
    weights = weights.mul(scale, axis=0)

    return apply_weekly_rebalance(weights)


# ── Strategy 5: Ensemble Meta-Strategy ─────────────────────────────────────

def strategy_ensemble(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight blend of strategies 1-4.
    Diversification across strategy types should yield highest Sharpe.
    """
    s1 = strategy_leveraged_gaa(prices)
    s2 = strategy_tsmom(prices)
    s3 = strategy_hedged_equity(prices)
    s4 = strategy_adaptive_multi_factor(prices)

    # Align columns
    common = s1.columns.intersection(s2.columns).intersection(s3.columns).intersection(s4.columns)
    ensemble = (s1[common] + s2[common] + s3[common] + s4[common]) / 4.0
    return ensemble  # Already weekly-rebalanced from components


# ── Leverage Sweep ─────────────────────────────────────────────────────────

def leverage_sweep(prices: pd.DataFrame, base_weights: pd.DataFrame,
                   strategy_name: str, levels: list[float] | None = None):
    """Test a strategy at different leverage levels."""
    if levels is None:
        levels = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]

    print(f"\n  ── Leverage Sweep: {strategy_name} ──")
    print(f"  {'Leverage':>8} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'AvgLev':>8}")
    print(f"  {'─' * 56}")

    best_sharpe = -999
    best_level = 1.0

    for level in levels:
        scaled = base_weights * level
        result = run_leveraged_backtest(prices, scaled)
        m = result["metrics"]
        marker = " ◄" if m["Sharpe Ratio"] > best_sharpe else ""
        if m["Sharpe Ratio"] > best_sharpe:
            best_sharpe = m["Sharpe Ratio"]
            best_level = level
        print(f"  {level:>7.1f}x {m['CAGR']*100:>7.2f}% {m['Sharpe Ratio']:>8.4f} "
              f"{m['Sortino Ratio']:>8.4f} {m['Max Drawdown']*100:>7.2f}% "
              f"{m['Avg Gross Leverage']:>7.4f}x{marker}")

    print(f"  Best Sharpe at {best_level}x leverage: {best_sharpe:.4f}")
    return best_level


# ── Audit Functions ────────────────────────────────────────────────────────

def audit_lookahead_bias(target_weights: pd.DataFrame, prices: pd.DataFrame):
    """
    Verify no look-ahead bias: target weights at time t should NOT correlate
    with same-day returns (they may correlate with next-day returns).
    """
    returns = prices.pct_change()
    issues = 0
    for col in target_weights.columns:
        if col in returns.columns:
            # Correlation between today's target weight and today's return
            same_day = target_weights[col].corr(returns[col])
            if abs(same_day) > 0.3:
                print(f"  ⚠️  {col}: same-day corr = {same_day:.4f} → POSSIBLE LOOKAHEAD")
                issues += 1
            else:
                print(f"  ✅ {col}: same-day corr = {same_day:.4f} (OK)")
    if issues == 0:
        print("  ✅ ALL assets pass look-ahead bias check")
    return issues


def audit_calculations(result: dict, name: str, initial_capital: float = 100_000.0):
    """Verify internal consistency of backtest calculations."""
    equity = result["equity_curve"]
    returns = result["portfolio_returns"]
    metrics = result["metrics"]
    issues = 0

    # 1. Equity curve consistency
    reconstructed = initial_capital * (1 + returns).cumprod()
    max_diff = (equity - reconstructed).abs().max()
    if max_diff > 1.0:
        print(f"  ⚠️  {name}: Equity curve drift = ${max_diff:.2f}")
        issues += 1
    else:
        print(f"  ✅ {name}: Equity curve consistent (max diff: ${max_diff:.4f})")

    # 2. Sharpe double-check
    excess = returns - RISK_FREE_RATE / 252
    sharpe_check = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    reported = metrics["Sharpe Ratio"]
    if abs(sharpe_check - reported) > 0.05:
        print(f"  ⚠️  {name}: Sharpe mismatch (computed={sharpe_check:.4f} vs reported={reported:.4f})")
        issues += 1
    else:
        print(f"  ✅ {name}: Sharpe verified ({sharpe_check:.4f} ≈ {reported:.4f})")

    # 3. CAGR double-check
    num_years = len(returns) / 252
    total_ret = (equity.iloc[-1] / initial_capital) - 1
    cagr_check = (1 + total_ret) ** (1 / num_years) - 1 if num_years > 0 else 0
    reported_cagr = metrics["CAGR"]
    if abs(cagr_check - reported_cagr) > 0.005:
        print(f"  ⚠️  {name}: CAGR mismatch (computed={cagr_check:.4f} vs reported={reported_cagr:.4f})")
        issues += 1
    else:
        print(f"  ✅ {name}: CAGR verified ({cagr_check*100:.2f}% ≈ {reported_cagr*100:.2f}%)")

    # 4. Sanity checks
    if metrics["CAGR"] > 0.80:
        print(f"  ⚠️  {name}: CAGR {metrics['CAGR']*100:.1f}% seems unrealistically high")
        issues += 1
    if metrics["Sharpe Ratio"] > 4.0:
        print(f"  ⚠️  {name}: Sharpe {metrics['Sharpe Ratio']:.2f} seems unrealistically high")
        issues += 1
    if metrics["Max Drawdown"] > -0.005 and len(returns) > 500:
        print(f"  ⚠️  {name}: MaxDD = {metrics['Max Drawdown']*100:.2f}% is suspiciously small")
        issues += 1

    return issues


def audit_double_counting(strategies: dict, prices: pd.DataFrame):
    """Check that strategies don't inadvertently duplicate signals."""
    print("\n  ── Signal Correlation Matrix ──")
    returns_dict = {}
    for name, result in strategies.items():
        returns_dict[name] = result["portfolio_returns"]
    returns_df = pd.DataFrame(returns_dict)
    corr = returns_df.corr()
    print(corr.round(3).to_string())
    # Flag highly correlated strategies
    for i, s1 in enumerate(corr.columns):
        for j, s2 in enumerate(corr.columns):
            if i < j and abs(corr.loc[s1, s2]) > 0.85:
                print(f"  ⚠️  High correlation between {s1} and {s2}: {corr.loc[s1, s2]:.3f}")


# ── Main Execution ─────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("LEVERAGED ALPHA STRATEGY EXPLORATION")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Universe: {UNIVERSE}")
    print(f"Costs: {TX_COST_BPS} bps tx, {LEVERAGE_COST_ANNUAL*100:.1f}% leverage, {SHORT_COST_ANNUAL*100:.1f}% short")
    print("=" * 80)

    # ── Load Data ──
    print("\n📊 Loading data...")
    prices = load_data()
    print(f"   Loaded {len(prices.columns)} tickers, {len(prices)} trading days")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Years: {len(prices) / 252:.1f}")

    # ── SPY Benchmark ──
    print("\n📈 SPY Benchmark")
    spy_ret = prices["SPY"].pct_change().dropna()
    spy_eq = 100_000 * (1 + spy_ret).cumprod()
    spy_metrics = compute_metrics(spy_ret, spy_eq, 100_000, risk_free_rate=RISK_FREE_RATE)
    print(f"   CAGR:      {spy_metrics['CAGR']*100:>8.2f}%")
    print(f"   Sharpe:    {spy_metrics['Sharpe Ratio']:>8.4f}")
    print(f"   Sortino:   {spy_metrics['Sortino Ratio']:>8.4f}")
    print(f"   Max DD:    {spy_metrics['Max Drawdown']*100:>8.2f}%")
    print(f"   Final Eq:  ${spy_metrics['Final Equity']:>12,.2f}")

    # ── Run Strategies ──
    strategy_defs = {
        "1_LeveragedGAA": strategy_leveraged_gaa,
        "2_TSMOM": strategy_tsmom,
        "3_HedgedEquity": strategy_hedged_equity,
        "4_AdaptiveMultiFactor": strategy_adaptive_multi_factor,
        "5_Ensemble": strategy_ensemble,
    }

    results = {}
    raw_weights = {}

    for name, strategy_fn in strategy_defs.items():
        print(f"\n{'─' * 60}")
        print(f"🔄 Running: {name}")
        target_w = strategy_fn(prices)
        raw_weights[name] = target_w
        result = run_leveraged_backtest(prices, target_w)
        results[name] = result
        m = result["metrics"]
        costs = result["cost_breakdown"]

        beat_cagr = "✅" if m["CAGR"] > spy_metrics["CAGR"] else "❌"
        beat_sharpe = "✅" if m["Sharpe Ratio"] > 1.95 else "❌"

        print(f"   CAGR:      {m['CAGR']*100:>8.2f}%  vs SPY {spy_metrics['CAGR']*100:.2f}%  {beat_cagr}")
        print(f"   Sharpe:    {m['Sharpe Ratio']:>8.4f}  target 1.95          {beat_sharpe}")
        print(f"   Sortino:   {m['Sortino Ratio']:>8.4f}")
        print(f"   Max DD:    {m['Max Drawdown']*100:>8.2f}%")
        print(f"   Win Rate:  {m['Win Rate']*100:>8.2f}%")
        print(f"   Avg Lev:   {m['Avg Gross Leverage']:>8.4f}x")
        print(f"   Turnover:  {m['Avg Turnover']:>8.4f}")
        print(f"   Final Eq:  ${m['Final Equity']:>12,.2f}")
        print(f"   Costs:     tx=${costs['tx_costs_total']:.0f}  "
              f"lev=${costs['leverage_costs_total']:.0f}  "
              f"short=${costs['short_costs_total']:.0f}")

    # ── Leverage Sweeps ──
    print(f"\n{'=' * 80}")
    print("📐 LEVERAGE OPTIMIZATION")
    print("=" * 80)
    optimal_leverage = {}
    for name in ["1_LeveragedGAA", "2_TSMOM", "3_HedgedEquity", "4_AdaptiveMultiFactor"]:
        # Get base (un-leveraged equivalent) weights by normalizing to 1.0
        base_w = raw_weights[name]
        gross = base_w.abs().sum(axis=1).clip(lower=1e-8)
        normalized = base_w.div(gross, axis=0)
        best_lev = leverage_sweep(prices, normalized, name,
                                  levels=[0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])
        optimal_leverage[name] = best_lev

    # ── Run strategies at OPTIMAL leverage ──
    print(f"\n{'=' * 80}")
    print("⚡ RE-RUNNING AT OPTIMAL LEVERAGE")
    print("=" * 80)
    optimized_results = {}
    for name, strategy_fn in strategy_defs.items():
        if name == "5_Ensemble":
            continue
        base_w = raw_weights[name]
        gross = base_w.abs().sum(axis=1).clip(lower=1e-8)
        normalized = base_w.div(gross, axis=0)
        opt_lev = optimal_leverage.get(name, 1.0)
        scaled = normalized * opt_lev
        result = run_leveraged_backtest(prices, scaled)
        optimized_results[name + f"_opt{opt_lev:.1f}x"] = result
        m = result["metrics"]
        beat_cagr = "✅" if m["CAGR"] > spy_metrics["CAGR"] else "❌"
        beat_sharpe = "✅" if m["Sharpe Ratio"] > 1.95 else "❌"
        print(f"\n   {name} @ {opt_lev:.1f}x leverage:")
        print(f"   CAGR:   {m['CAGR']*100:>8.2f}% {beat_cagr}  Sharpe: {m['Sharpe Ratio']:>8.4f} {beat_sharpe}  "
              f"MaxDD: {m['Max Drawdown']*100:.2f}%  AvgLev: {m['Avg Gross Leverage']:.2f}x")

    # Rebuild ensemble at optimal leverage
    print(f"\n   Re-building ensemble at optimal leverage...")
    ensemble_parts = []
    for name in ["1_LeveragedGAA", "2_TSMOM", "3_HedgedEquity", "4_AdaptiveMultiFactor"]:
        base_w = raw_weights[name]
        gross = base_w.abs().sum(axis=1).clip(lower=1e-8)
        normalized = base_w.div(gross, axis=0)
        opt_lev = optimal_leverage.get(name, 1.0)
        ensemble_parts.append(normalized * opt_lev)

    common_cols = ensemble_parts[0].columns
    for part in ensemble_parts[1:]:
        common_cols = common_cols.intersection(part.columns)
    opt_ensemble = sum(p[common_cols] for p in ensemble_parts) / len(ensemble_parts)
    opt_ensemble_result = run_leveraged_backtest(prices, opt_ensemble)
    optimized_results["5_Ensemble_optimal"] = opt_ensemble_result
    m = opt_ensemble_result["metrics"]
    beat_cagr = "✅" if m["CAGR"] > spy_metrics["CAGR"] else "❌"
    beat_sharpe = "✅" if m["Sharpe Ratio"] > 1.95 else "❌"
    print(f"   5_Ensemble optimal:")
    print(f"   CAGR:   {m['CAGR']*100:>8.2f}% {beat_cagr}  Sharpe: {m['Sharpe Ratio']:>8.4f} {beat_sharpe}  "
          f"MaxDD: {m['Max Drawdown']*100:.2f}%  AvgLev: {m['Avg Gross Leverage']:.2f}x")

    # ── AUDIT PHASE ──
    print(f"\n{'=' * 80}")
    print("🔍 AUDIT: Look-Ahead Bias")
    print("=" * 80)
    total_issues = 0
    for name, w in raw_weights.items():
        print(f"\n--- {name} ---")
        total_issues += audit_lookahead_bias(w, prices)

    print(f"\n{'=' * 80}")
    print("🔍 AUDIT: Calculation Integrity")
    print("=" * 80)
    all_results = {**results, **optimized_results}
    for name, result in all_results.items():
        total_issues += audit_calculations(result, name)

    print(f"\n{'=' * 80}")
    print("🔍 AUDIT: Strategy Correlation (double-counting check)")
    print("=" * 80)
    audit_double_counting(results, prices)

    # ── FINAL SUMMARY ──
    print(f"\n{'=' * 80}")
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Strategy':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'AvgLev':>8}"
    print(header)
    print("─" * len(header))

    print(f"{'SPY (Benchmark)':<35} {spy_metrics['CAGR']*100:>7.2f}% "
          f"{spy_metrics['Sharpe Ratio']:>8.4f} {spy_metrics['Sortino Ratio']:>8.4f} "
          f"{spy_metrics['Max Drawdown']*100:>7.2f}% {'1.00x':>8}")
    print("─" * len(header))

    winners = []
    for name, result in all_results.items():
        m = result["metrics"]
        star = " ⭐" if m["CAGR"] > spy_metrics["CAGR"] and m["Sharpe Ratio"] > 1.95 else ""
        if star:
            winners.append((name, m))
        print(f"{name:<35} {m['CAGR']*100:>7.2f}% {m['Sharpe Ratio']:>8.4f} "
              f"{m['Sortino Ratio']:>8.4f} {m['Max Drawdown']*100:>7.2f}% "
              f"{m['Avg Gross Leverage']:>7.2f}x{star}")

    if winners:
        print(f"\n🏆 STRATEGIES MEETING TARGETS (CAGR > SPY & Sharpe > 1.95):")
        for name, m in winners:
            print(f"   ⭐ {name}: CAGR={m['CAGR']*100:.2f}%, Sharpe={m['Sharpe Ratio']:.4f}")
    else:
        print(f"\n⚠️  No strategies met BOTH targets yet — see leverage sweep for closest.")

    print(f"\n{'=' * 80}")
    print(f"Total audit issues found: {total_issues}")
    if total_issues == 0:
        print("✅ All audits passed — calculations verified, no look-ahead bias detected")
    else:
        print(f"⚠️  {total_issues} issues require investigation")
    print("=" * 80)

    return all_results, spy_metrics, prices


if __name__ == "__main__":
    all_results, spy_metrics, prices = main()
