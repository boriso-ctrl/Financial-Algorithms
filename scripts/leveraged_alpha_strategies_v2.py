"""
Leveraged Alpha Strategies v2 — Targeted High-Sharpe Approaches
================================================================
Targets: CAGR > SPY (~13.6%), Sharpe > 1.95
Period: 2010-2025 (15 years across multiple regimes)

Key design improvements over v1:
  - Concentrated equity (QQQ+SPY) instead of over-diversification
  - Binary risk-on/risk-off switching (not gradual)
  - Volatility-managed sizing (Moreira & Muir 2017)
  - Cash as safe haven (avoids 2022 bond crash)
  - Multi-speed timing signals
  - Multiple leverage levels tested per strategy
  - Multiple test periods (5/10/15 years)

Self-auditing: look-ahead bias, calculation integrity, max leverage consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from financial_algorithms.backtest.metrics import compute_metrics

# ── Configuration ──────────────────────────────────────────────────────────

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "GLD"]
START_DATE = "2010-01-01"
END_DATE = "2025-03-01"

TX_COST_BPS = 5
LEVERAGE_COST_ANNUAL = 0.015
SHORT_COST_ANNUAL = 0.005
RISK_FREE_RATE = 0.0


# ── Data Loading ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    tickers = list(set(UNIVERSE))
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.dropna(how="all").ffill().bfill()


# ── Custom Backtest Engine ─────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000.0,
    rebalance_freq: int = 5,
) -> dict:
    """Vectorized backtest with proper leverage, costs, and weekly rebalancing."""
    common = prices.columns.intersection(target_weights.columns)
    prices = prices[common]
    w = target_weights[common].reindex(prices.index).fillna(0)

    # Weekly rebalancing: only update on rebalance days, hold between
    mask = pd.Series(False, index=w.index)
    mask.iloc[::rebalance_freq] = True
    w = w.where(mask).ffill().fillna(0)

    # SHIFT by 1 day — prevents look-ahead bias
    w = w.shift(1).fillna(0)

    ret = prices.pct_change().fillna(0)
    gross_ret = (w * ret).sum(axis=1)

    # Costs
    turnover = w.diff().fillna(0).abs().sum(axis=1)
    tx = turnover * TX_COST_BPS / 10_000
    gross_exp = w.abs().sum(axis=1)
    lev_cost = (gross_exp - 1.0).clip(lower=0) * LEVERAGE_COST_ANNUAL / 252
    short_exp = w.clip(upper=0).abs().sum(axis=1)
    short_cost = short_exp * SHORT_COST_ANNUAL / 252

    net_ret = gross_ret - tx - lev_cost - short_cost
    equity = initial_capital * (1 + net_ret).cumprod()
    equity.name = "Equity"

    metrics = compute_metrics(
        net_ret, equity, initial_capital,
        risk_free_rate=RISK_FREE_RATE,
        turnover=turnover, gross_exposure=gross_exp,
    )
    return {
        "equity_curve": equity,
        "portfolio_returns": net_ret,
        "weights": w,
        "turnover": turnover,
        "gross_exposure": gross_exp,
        "metrics": metrics,
    }


# ── Signal Helpers ────────────────────────────────────────────────────────

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, min_periods=n).mean()

def realized_vol(series: pd.Series, lookback: int = 21) -> pd.Series:
    return series.pct_change().rolling(lookback, min_periods=10).std() * np.sqrt(252)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-10)
    return 100 - 100 / (1 + rs)

def momentum(series: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    return series.shift(skip).pct_change(lookback - skip)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY A: Dual-MA Trend Timer with Leverage
#   - Long equity when SPY > 200-day MA (golden cross not required)
#   - Cash when SPY < 200-day MA
#   - Apply configurable leverage when long
# ═══════════════════════════════════════════════════════════════════════════

def strategy_a_trend_timer(prices: pd.DataFrame, leverage: float = 1.0,
                           equity_split: dict | None = None) -> pd.DataFrame:
    """Binary trend timing: leveraged equity or cash."""
    if equity_split is None:
        equity_split = {"SPY": 0.55, "QQQ": 0.45}

    spy = prices["SPY"]
    ma200 = sma(spy, 200)

    risk_on = (spy > ma200).astype(float)

    available = [t for t in equity_split if t in prices.columns]
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ticker in available:
        weights[ticker] = risk_on * equity_split[ticker] * leverage

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY B: Multi-Speed Trend Timer
#   - Fast MA (50) + Slow MA (200) = 3 regimes
#   - Both bullish: leverage equity
#   - Fast bearish only: reduce equity + add hedge
#   - Both bearish: cash/gold
# ═══════════════════════════════════════════════════════════════════════════

def strategy_b_multi_speed(prices: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
    """3-regime trend timing with adaptive exposure."""
    spy = prices["SPY"]
    ma50 = sma(spy, 50)
    ma200 = sma(spy, 200)

    # Regimes
    both_bull = (ma50 > ma200) & (spy > ma200)
    pullback = (ma50 > ma200) & (spy <= ma200)  # minor dip in uptrend
    correction = (spy > ma200) & (ma50 <= ma200)  # momentum weakening
    bearish = (ma50 <= ma200) & (spy <= ma200)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    eq = ["SPY", "QQQ"]
    eq_avail = [t for t in eq if t in prices.columns]

    for t in eq_avail:
        w_per = leverage / len(eq_avail)
        weights[t] = np.where(both_bull, w_per,
                     np.where(pullback, w_per * 0.5,
                     np.where(correction, w_per * 0.3, 0.0)))

    # Hedge: GLD when bearish
    if "GLD" in prices.columns:
        weights["GLD"] = np.where(bearish, 0.4 * leverage,
                         np.where(correction, 0.3 * leverage, 0.0))
    # Bonds as mild hedge
    if "IEF" in prices.columns:
        weights["IEF"] = np.where(bearish, 0.3 * leverage,
                         np.where(correction, 0.2 * leverage, 0.0))

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY C: Volatility-Managed Equity
#   - Always long SPY+QQQ but size inversely to realized vol
#   - Targets specific portfolio vol → high leverage in calm, low in crisis
#   - Proven to improve Sharpe (Moreira & Muir, 2017)
# ═══════════════════════════════════════════════════════════════════════════

def strategy_c_vol_managed(prices: pd.DataFrame, target_vol: float = 0.12,
                           max_leverage: float = 2.0) -> pd.DataFrame:
    """Volatility-managed portfolio with dynamic leverage."""
    spy = prices["SPY"]
    qqq = prices["QQQ"] if "QQQ" in prices.columns else spy

    # Portfolio: 55/45 SPY/QQQ
    port_ret = 0.55 * spy.pct_change() + 0.45 * qqq.pct_change()
    port_vol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)

    # Scale factor: target_vol / realized_vol
    scale = (target_vol / port_vol.clip(lower=0.03)).clip(upper=max_leverage)
    # Smooth to reduce turnover
    scale = scale.rolling(5, min_periods=1).mean()

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in prices.columns:
        weights["SPY"] = 0.55 * scale
    if "QQQ" in prices.columns:
        weights["QQQ"] = 0.45 * scale

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY D: Vol-Managed + Trend Filter (Best of B+C)
#   - Vol-managed sizing (from C)
#   - Trend filter: zero out during bearish trends (from A)
#   - Add GLD hedge during risk-off
# ═══════════════════════════════════════════════════════════════════════════

def strategy_d_vol_trend(prices: pd.DataFrame, target_vol: float = 0.12,
                         max_leverage: float = 2.0) -> pd.DataFrame:
    """Vol-managed equity with trend filter and hedging."""
    spy = prices["SPY"]
    ma200 = sma(spy, 200)
    ma50 = sma(spy, 50)

    # Vol-managed base
    qqq = prices["QQQ"] if "QQQ" in prices.columns else spy
    port_ret = 0.55 * spy.pct_change() + 0.45 * qqq.pct_change()
    port_vol = port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    scale = (target_vol / port_vol.clip(lower=0.03)).clip(upper=max_leverage)
    scale = scale.rolling(5, min_periods=1).mean()

    # Trend filter
    uptrend = (spy > ma200).astype(float)
    pullback = ((spy <= ma200) & (ma50 > ma200)).astype(float) * 0.3
    trend_mult = uptrend + pullback  # 1.0 in uptrend, 0.3 in pullback, 0 in bear

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in prices.columns:
        weights["SPY"] = 0.55 * scale * trend_mult
    if "QQQ" in prices.columns:
        weights["QQQ"] = 0.45 * scale * trend_mult

    # GLD hedge when risk-off (bear)
    if "GLD" in prices.columns:
        bear = (1 - trend_mult).clip(lower=0)
        weights["GLD"] = bear * 0.4

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY E: Momentum Rotation + Leverage
#   - Each month: long the top-2 momentum assets from {SPY, QQQ, IWM, EFA, GLD, TLT}
#   - Zero out assets with negative 12-month momentum (absolute filter)
#   - Vol-normalized sizing
# ═══════════════════════════════════════════════════════════════════════════

def strategy_e_momentum_rotation(prices: pd.DataFrame, leverage: float = 1.0,
                                 top_n: int = 2) -> pd.DataFrame:
    """Momentum rotation across full universe with absolute filter."""
    available = [c for c in UNIVERSE if c in prices.columns]
    p = prices[available]

    mom = p.shift(21).pct_change(231)  # 12-month momentum, skip 1 month
    vol = p.pct_change().rolling(63, min_periods=20).std() * np.sqrt(252)

    # Absolute momentum filter: only long if positive
    positive = mom > 0

    # Rank (higher = better momentum)
    rank = mom.rank(axis=1, ascending=True)
    n_assets = len(available)
    top_mask = rank >= (n_assets - top_n + 1)

    # Final selection: must have positive momentum AND be in top N
    selected = positive & top_mask

    # Vol-normalized weights
    inv_vol = 1.0 / vol.clip(lower=0.03)
    raw_weights = selected.astype(float) * inv_vol
    total = raw_weights.sum(axis=1).clip(lower=1e-8)
    weights = raw_weights.div(total, axis=0) * leverage

    # When nothing selected, go to least-volatile asset
    nothing = ~selected.any(axis=1)
    if "IEF" in available:
        weights.loc[nothing, "IEF"] = 0.5 * leverage
    if "GLD" in available:
        weights.loc[nothing, "GLD"] = 0.3 * leverage

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY F: Hedged Leveraged Equity (permanent hedge)
#   - Core: leveraged SPY+QQQ (adjustable)
#   - Permanent hedge: 20-30% in GLD + IEF (always)
#   - Rebalance weekly
#   - Trend filter reduces equity in downtrends
# ═══════════════════════════════════════════════════════════════════════════

def strategy_f_hedged_leveraged(prices: pd.DataFrame, equity_lev: float = 1.2,
                                hedge_pct: float = 0.25) -> pd.DataFrame:
    """Leveraged equity core with permanent hedge allocation."""
    spy = prices["SPY"]
    ma200 = sma(spy, 200)
    ma50 = sma(spy, 50)

    # Trend multiplier for equity
    uptrend = (spy > ma200).astype(float)
    strong_trend = (uptrend * ((ma50 > ma200).astype(float))).clip(lower=0.3)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Equity core (leveraged)
    eq_base = equity_lev * (1 - hedge_pct)
    if "SPY" in prices.columns:
        weights["SPY"] = eq_base * 0.55 * strong_trend
    if "QQQ" in prices.columns:
        weights["QQQ"] = eq_base * 0.45 * strong_trend

    # Permanent hedge
    if "GLD" in prices.columns:
        weights["GLD"] = hedge_pct * equity_lev * 0.5
    if "IEF" in prices.columns:
        weights["IEF"] = hedge_pct * equity_lev * 0.5

    return weights


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY G: Ensemble of D + E (low correlation expected)
# ═══════════════════════════════════════════════════════════════════════════

def strategy_g_ensemble(prices: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
    """Blend of vol-managed trend (D) + momentum rotation (E)."""
    w_d = strategy_d_vol_trend(prices, target_vol=0.12, max_leverage=2.0)
    w_e = strategy_e_momentum_rotation(prices, leverage=1.3, top_n=2)

    common = w_d.columns.intersection(w_e.columns)
    combined = (w_d[common] * 0.5 + w_e[common] * 0.5) * leverage
    return combined


# ── Audit Functions ────────────────────────────────────────────────────────

def audit_lookahead(weights: pd.DataFrame, prices: pd.DataFrame) -> int:
    """Check for look-ahead bias via same-day return correlation."""
    ret = prices.pct_change()
    issues = 0
    for col in weights.columns:
        if col in ret.columns and weights[col].abs().sum() > 0:
            corr = weights[col].corr(ret[col])
            if abs(corr) > 0.25:
                print(f"    ⚠️  {col}: same-day corr = {corr:.4f} — POSSIBLE LOOKAHEAD")
                issues += 1
            else:
                print(f"    ✅ {col}: {corr:.4f}")
    return issues


def audit_metrics(result: dict, name: str) -> int:
    """Verify calculation consistency."""
    issues = 0
    equity = result["equity_curve"]
    ret = result["portfolio_returns"]
    m = result["metrics"]

    # Equity curve check
    recon = 100_000 * (1 + ret).cumprod()
    diff = (equity - recon).abs().max()
    if diff > 1.0:
        print(f"    ⚠️  {name}: Equity drift = ${diff:.2f}")
        issues += 1

    # Sharpe check
    excess = ret - RISK_FREE_RATE / 252
    sharpe_check = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    if abs(sharpe_check - m["Sharpe Ratio"]) > 0.05:
        print(f"    ⚠️  {name}: Sharpe mismatch ({sharpe_check:.4f} vs {m['Sharpe Ratio']:.4f})")
        issues += 1

    # CAGR check
    yrs = len(ret) / 252
    total_ret = equity.iloc[-1] / 100_000 - 1
    cagr_check = (1 + total_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
    if abs(cagr_check - m["CAGR"]) > 0.005:
        print(f"    ⚠️  {name}: CAGR mismatch ({cagr_check*100:.2f}% vs {m['CAGR']*100:.2f}%)")
        issues += 1

    # Sanity
    if m["CAGR"] > 0.80:
        print(f"    ⚠️  {name}: CAGR {m['CAGR']*100:.1f}% unrealistically high")
        issues += 1
    if m["Sharpe Ratio"] > 4.5:
        print(f"    ⚠️  {name}: Sharpe {m['Sharpe Ratio']:.2f} unrealistically high")
        issues += 1

    if issues == 0:
        print(f"    ✅ {name}: all checks pass")
    return issues


# ── Main Execution ─────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("LEVERAGED ALPHA v2 — TARGETED HIGH-SHARPE EXPLORATION")
    print(f"Targets: CAGR > SPY, Sharpe > 1.95  |  Period: {START_DATE} to {END_DATE}")
    print("=" * 90)

    print("\n📊 Loading data...")
    prices = load_data()
    print(f"   {len(prices.columns)} tickers, {len(prices)} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)/252:.1f}yr)")

    # ── Benchmark ──
    spy_ret = prices["SPY"].pct_change().dropna()
    spy_eq = 100_000 * (1 + spy_ret).cumprod()
    spy_m = compute_metrics(spy_ret, spy_eq, 100_000, risk_free_rate=RISK_FREE_RATE)
    print(f"\n📈 SPY Benchmark: CAGR={spy_m['CAGR']*100:.2f}%, Sharpe={spy_m['Sharpe Ratio']:.4f}, "
          f"MaxDD={spy_m['Max Drawdown']*100:.2f}%")

    # ── Strategy Definitions with leverage grid ──
    strategies = {
        "A_TrendTimer": (strategy_a_trend_timer, {"leverage": [1.0, 1.3, 1.5, 1.8, 2.0]}),
        "B_MultiSpeed": (strategy_b_multi_speed, {"leverage": [1.0, 1.3, 1.5, 1.8, 2.0]}),
        "C_VolManaged": (strategy_c_vol_managed, {"target_vol": [0.10, 0.12, 0.15, 0.18, 0.20],
                                                   "max_leverage": [2.0]}),
        "D_VolTrend": (strategy_d_vol_trend, {"target_vol": [0.10, 0.12, 0.15, 0.18, 0.20],
                                               "max_leverage": [2.0]}),
        "E_MomRotation": (strategy_e_momentum_rotation, {"leverage": [1.0, 1.3, 1.5, 1.8],
                                                          "top_n": [2, 3]}),
        "F_HedgedLev": (strategy_f_hedged_leveraged, {"equity_lev": [1.0, 1.3, 1.5, 1.8, 2.0],
                                                       "hedge_pct": [0.20, 0.25, 0.30]}),
        "G_Ensemble": (strategy_g_ensemble, {"leverage": [0.8, 1.0, 1.2, 1.5]}),
    }

    all_results = {}
    best_per_strategy = {}

    for strat_name, (fn, param_grid) in strategies.items():
        print(f"\n{'─' * 70}")
        print(f"🔄 {strat_name}")

        # Generate all parameter combos
        keys = list(param_grid.keys())
        combos = [{}]
        for k in keys:
            new_combos = []
            for combo in combos:
                for v in param_grid[k]:
                    new_combos.append({**combo, k: v})
            combos = new_combos

        best_sharpe = -999
        best_combo = None
        best_result = None

        for combo in combos:
            target_w = fn(prices, **combo)
            result = run_backtest(prices, target_w)
            m = result["metrics"]
            label = f"{strat_name}({', '.join(f'{k}={v}' for k, v in combo.items())})"
            all_results[label] = result

            beat_cagr = "✓" if m["CAGR"] > spy_m["CAGR"] else " "
            beat_sharpe = "✓" if m["Sharpe Ratio"] > 1.95 else " "
            star = " ⭐" if beat_cagr == "✓" and beat_sharpe == "✓" else ""

            params_str = ", ".join(f"{k}={v}" for k, v in combo.items())
            print(f"   {params_str:<30} CAGR={m['CAGR']*100:>7.2f}%[{beat_cagr}] "
                  f"Sharpe={m['Sharpe Ratio']:>7.4f}[{beat_sharpe}] "
                  f"MaxDD={m['Max Drawdown']*100:>7.2f}% "
                  f"AvgLev={m['Avg Gross Leverage']:>5.2f}x{star}")

            if m["Sharpe Ratio"] > best_sharpe:
                best_sharpe = m["Sharpe Ratio"]
                best_combo = combo
                best_result = result

        best_per_strategy[strat_name] = (best_combo, best_result)
        bm = best_result["metrics"]
        print(f"   ► Best: {best_combo} → Sharpe={bm['Sharpe Ratio']:.4f}, CAGR={bm['CAGR']*100:.2f}%")

    # ── Audit Phase ──
    print(f"\n{'=' * 90}")
    print("🔍 AUDIT PHASE")
    print("=" * 90)
    total_issues = 0

    # Check best of each strategy for lookahead
    for strat_name, (combo, result) in best_per_strategy.items():
        fn = strategies[strat_name][0]
        w = fn(prices, **combo)
        print(f"\n  [{strat_name}] Look-ahead bias:")
        total_issues += audit_lookahead(w, prices)
        print(f"  [{strat_name}] Calculations:")
        total_issues += audit_metrics(result, strat_name)

    # ── Correlation Matrix ──
    print(f"\n  Strategy Return Correlations (best configs):")
    corr_data = {}
    for name, (_, result) in best_per_strategy.items():
        corr_data[name] = result["portfolio_returns"]
    corr_df = pd.DataFrame(corr_data).corr()
    print(corr_df.round(3).to_string())

    # ── Look for lowest-correlation pairs for potential ensemble ──
    print(f"\n  Best diversification pairs:")
    for i, s1 in enumerate(corr_df.columns):
        for j, s2 in enumerate(corr_df.columns):
            if i < j:
                c = corr_df.loc[s1, s2]
                if c < 0.5:
                    print(f"    {s1} × {s2}: ρ = {c:.3f} (good diversification)")

    # ── Sub-period Analysis ──
    print(f"\n{'=' * 90}")
    print("📅 SUB-PERIOD ANALYSIS (best config per strategy)")
    print("=" * 90)

    periods = {
        "Full (2010-2025)": (None, None),
        "2015-2025": ("2015-01-01", None),
        "2018-2025": ("2018-01-01", None),
        "2020-2025": ("2020-01-01", None),
    }

    for period_name, (start, end) in periods.items():
        print(f"\n  ── {period_name} ──")
        sub_prices = prices.copy()
        if start:
            sub_prices = sub_prices[sub_prices.index >= start]
        if end:
            sub_prices = sub_prices[sub_prices.index <= end]

        # SPY benchmark for sub-period
        sub_spy_ret = sub_prices["SPY"].pct_change().dropna()
        sub_spy_eq = 100_000 * (1 + sub_spy_ret).cumprod()
        sub_spy_m = compute_metrics(sub_spy_ret, sub_spy_eq, 100_000,
                                     risk_free_rate=RISK_FREE_RATE)
        print(f"  SPY: CAGR={sub_spy_m['CAGR']*100:.2f}%, Sharpe={sub_spy_m['Sharpe Ratio']:.4f}, "
              f"MaxDD={sub_spy_m['Max Drawdown']*100:.2f}%")

        for strat_name, (combo, _) in best_per_strategy.items():
            fn = strategies[strat_name][0]
            w = fn(sub_prices, **combo)
            result = run_backtest(sub_prices, w)
            m = result["metrics"]
            flags = ""
            if m["CAGR"] > sub_spy_m["CAGR"]:
                flags += "CAGR✓ "
            if m["Sharpe Ratio"] > 1.95:
                flags += "Sharpe✓ "
            if m["CAGR"] > sub_spy_m["CAGR"] and m["Sharpe Ratio"] > 1.95:
                flags += "⭐"
            print(f"  {strat_name:<25} CAGR={m['CAGR']*100:>7.2f}% Sharpe={m['Sharpe Ratio']:>7.4f} "
                  f"MaxDD={m['Max Drawdown']*100:>7.2f}% {flags}")

    # ── Final Summary ──
    print(f"\n{'=' * 90}")
    print("📊 FINAL SUMMARY — ALL RESULTS")
    print("=" * 90)
    print(f"{'Config':<55} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'AvgLev':>6}")
    print("─" * 82)
    print(f"{'SPY (benchmark)':<55} {spy_m['CAGR']*100:>6.2f}% {spy_m['Sharpe Ratio']:>7.4f} "
          f"{spy_m['Max Drawdown']*100:>6.2f}% {'1.00x':>6}")
    print("─" * 82)

    winners = []
    # Sort by Sharpe descending
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["metrics"]["Sharpe Ratio"],
                            reverse=True)
    for label, result in sorted_results[:30]:  # Top 30
        m = result["metrics"]
        star = " ⭐" if m["CAGR"] > spy_m["CAGR"] and m["Sharpe Ratio"] > 1.95 else ""
        if star:
            winners.append((label, m))
        print(f"{label:<55} {m['CAGR']*100:>6.2f}% {m['Sharpe Ratio']:>7.4f} "
              f"{m['Max Drawdown']*100:>6.2f}% {m['Avg Gross Leverage']:>5.2f}x{star}")

    if winners:
        print(f"\n🏆 WINNERS (CAGR > SPY & Sharpe > 1.95):")
        for label, m in winners:
            print(f"   ⭐ {label}: CAGR={m['CAGR']*100:.2f}%, Sharpe={m['Sharpe Ratio']:.4f}, "
                  f"MaxDD={m['Max Drawdown']*100:.2f}%")
    else:
        print(f"\n⚠️  No full-period winners. Check sub-period analysis for promising approaches.")

    print(f"\nTotal audit issues: {total_issues}")
    if total_issues == 0:
        print("✅ All audits passed")
    print("=" * 90)

    return all_results, best_per_strategy, spy_m, prices


if __name__ == "__main__":
    all_results, best_per_strategy, spy_m, prices = main()
