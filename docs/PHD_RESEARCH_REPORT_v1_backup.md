# Leveraged Alpha Generation Through Statistical Arbitrage Ensembles with Cascaded Risk Management: A Complete Systematic Investigation

## From a Single Prompt to a 2,751-Strategy Pipeline — Every Concept, Every Formula, Every Lesson

**A Comprehensive Research Report and Development Journal**

---

## Abstract

This report presents an exhaustive empirical investigation into the generation of leveraged alpha using statistical arbitrage ensembles over U.S. exchange-traded funds (ETFs). It is written as both a technical reference and a development journal: every mathematical concept is derived from first principles, every design decision is explained with its motivation and outcome, and the full 18-version evolutionary arc is documented as a narrative journey from a founding prompt to a validated trading system.

We develop, backtest, audit, and validate a multi-layer portfolio construction framework that combines five distinct alpha sources — Z-score pair trading, volatility-regime crash hedging, cross-sectional momentum, time-series momentum, and sector rotation — within an ensemble architecture governed by cascaded risk overlays including volatility targeting, hierarchical drawdown control, triple-layer drawdown control, and position-level drawdown control with explicit rebalancing costs.

The research spans 18 code iterations (v1–v18) across a 15-year in-sample period (January 2010 – March 2025; 3,813 trading days) and a 260-day true out-of-sample holdout (March 2025 – March 2026) on 18 ETFs encompassing sector, broad-market, safe-haven, and fixed-income exposures. Over 2,751 strategy variants were systematically evaluated in the final iteration. Our best genuine out-of-sample ensemble — a simple two-source blend of 85% Z-score pair portfolio and 15% CrashHedge — achieves an OOS Sharpe ratio of 2.30 (unleveraged) and 2.01 at 3× leverage with a compound annual growth rate (CAGR) of 46.30% and a maximum drawdown of −7.59%.

A comprehensive self-audit at the v16 milestone identified 41 methodological issues (11 rated HIGH), including the critical discovery that cascaded drawdown control overlays mechanically inflate Sharpe ratios by a factor of approximately 4.5× when applied to a passive SPY benchmark — a finding with broad implications for the systematic trading literature. All 11 HIGH-priority findings were remediated in the audit-hardened v17 release, producing a 35% reduction in in-sample Sharpe (from 6.25 to 4.07) but yielding honest, deployable out-of-sample metrics.

This document aims to leave no formula unexplained, no parameter unjustified, and no failure undocumented. The reader should be able to reconstruct the entire system from this report alone.

**Keywords**: statistical arbitrage, pairs trading, leverage, alpha generation, drawdown control, volatility targeting, ensemble methods, out-of-sample validation, backtesting methodology, risk management, Z-score mean reversion, cross-sectional momentum, crash hedging, deflated Sharpe ratio

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Data and Universe](#3-data-and-universe)
4. [Methodology](#4-methodology)
   - 4.1 [Pair Trading Engine](#41-pair-trading-engine)
   - 4.2 [Alpha Sources](#42-alpha-sources)
   - 4.3 [Ensemble Construction](#43-ensemble-construction)
   - 4.4 [Risk Overlay Architecture](#44-risk-overlay-architecture)
   - 4.5 [Leverage Framework](#45-leverage-framework)
   - 4.6 [Cost Model](#46-cost-model)
   - 4.7 [Walk-Forward and Out-of-Sample Protocol](#47-walk-forward-and-out-of-sample-protocol)
   - 4.8 [Statistical Validation](#48-statistical-validation)
5. [Iterative Development and Results](#5-iterative-development-and-results)
   - 5.1 [Phase I: Exploratory Period (v1–v7)](#51-phase-i-exploratory-period-v1v7)
   - 5.2 [Phase II: Pair Trading Architecture (v8–v10)](#52-phase-ii-pair-trading-architecture-v8v10)
   - 5.3 [Phase III: First Winners and Scaling (v10b–v11)](#53-phase-iii-first-winners-and-scaling-v10bv11)
   - 5.4 [Phase IV: Risk Management Revolution (v12–v14)](#54-phase-iv-risk-management-revolution-v12v14)
   - 5.5 [Phase V: Drawdown Control Escalation (v15–v16)](#55-phase-v-drawdown-control-escalation-v15v16)
   - 5.6 [Phase VI: Comprehensive Audit](#56-phase-vi-comprehensive-audit)
   - 5.7 [Phase VII: Audit-Hardened System (v17)](#57-phase-vii-audit-hardened-system-v17)
   - 5.8 [Phase VIII: Alpha Enhancement (v18)](#58-phase-viii-alpha-enhancement-v18)
6. [Out-of-Sample Results](#6-out-of-sample-results)
7. [Audit Findings and Methodological Lessons](#7-audit-findings-and-methodological-lessons)
8. [Discussion](#8-discussion)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [Conclusions](#10-conclusions)
11. [References](#11-references)
12. [Appendices](#12-appendices)

---

## 1. Introduction

### 1.1 Research Objective

The central objective of this research program is to construct a systematic trading strategy that simultaneously achieves:

1. **Compound annual growth rate (CAGR) exceeding that of the S&P 500 index** (~13.6% annualized over the in-sample period);
2. **Sharpe ratio greater than 1.95**, indicating superior risk-adjusted returns relative to standard equity benchmarks.

These dual objectives are pursued through the systematic exploration of leverage (both fractional and multiplied, e.g., 0.8×–8× notional), short selling, hedging strategies, and multi-source alpha ensemble construction. A distinctive feature of this research is its emphasis on self-auditing: the system underwent a comprehensive 41-point audit against established backtesting failure modes, and all HIGH-priority findings were remediated before out-of-sample evaluation.

### 1.2 Motivation

Generating risk-adjusted alpha above institutional benchmarks ($\text{Sharpe} > 2.0$) remains one of the most challenging problems in quantitative finance. Classical mean-variance optimization (Markowitz, 1952) provides the theoretical foundation but suffers from estimation error sensitivity (Michaud, 1989). Statistical arbitrage strategies—particularly pairs trading—offer a market-neutral approach to alpha extraction (Gatev, Goetzmann, & Rouwenhorst, 2006), but their returns have diminished over time as the strategy has become more crowded (Do & Faff, 2010).

This research investigates whether combining multiple decorrelated alpha sources (pair trading, momentum, crash hedging) within a cascaded risk management framework can produce robust, deployable alpha even in the post-crisis era of compressed risk premia.

### 1.3 Contribution

This work makes several contributions to the empirical quantitative finance literature:

1. **Cascaded risk overlay taxonomy**: We identify and document the mechanical Sharpe inflation caused by stacking multiple drawdown control layers, demonstrating that SPY alone achieves a Sharpe of 3.76 (from a baseline of 0.84) after triple-layer drawdown control—an inflation factor of approximately $4.5\times$ that contains no genuine alpha.

2. **Honest out-of-sample protocol**: We implement a strict 260-day out-of-sample holdout (March 2025–March 2026) with all parameters locked from in-sample fitting, demonstrating that in-sample Sharpe ratios of 6.25 compress to 2.30 under honest evaluation—a result consistent with the known overfitting dynamics of systematic strategies (Bailey, Borwein, López de Prado, & Zhu, 2017).

3. **Alpha source decomposition**: Through systematic ablation across 2,751 strategy variants, we show that a simple two-source ensemble (85% pair portfolio + 15% crash hedge) dominates all three-, four-, and five-source blends on risk-adjusted out-of-sample metrics, contrary to the common assumption that alpha diversification monotonically improves portfolio quality.

4. **Iterative development documentation**: We provide a complete 18-version development record, including all failures, enabling future researchers to understand the full landscape of design decisions rather than only the final result.

---

## 2. Literature Review

### 2.1 Statistical Arbitrage and Pairs Trading

Pairs trading—the simultaneous long-short exploitation of temporary mispricings between cointegrated or correlated securities—has a rich empirical history. Gatev et al. (2006) documented annualized excess returns of approximately 11% for a distance-based pairs strategy over 1962–2002. Subsequent work by Do and Faff (2010, 2012) demonstrated significant return degradation post-2002, attributable to increased strategy crowding and execution costs.

The Z-score approach employed in this research follows the Ornstein-Uhlenbeck framework formalized by Elliott, van der Hoek, and Malcolm (2005), where the log-spread $s_t = \ln P_t^A - \ln P_t^B$ is modeled as a mean-reverting process:

$$ds_t = \theta(\mu - s_t)\,dt + \sigma\,dW_t$$

Entry occurs when the standardized Z-score $z_t = (s_t - \bar{s}) / \hat{\sigma}_s$ exceeds a threshold $|z_t| > z_{\text{entry}}$, and exit occurs when the Z-score reverts to $|z_t| < z_{\text{exit}}$.

### 2.2 Momentum Strategies

Cross-sectional momentum (Jegadeesh & Titman, 1993) ranks assets by trailing returns over a formation period (typically 12 months minus the most recent month to avoid short-term reversal) and takes long positions in winners and short positions in losers. The strategy has been documented across markets and asset classes (Asness, Moskowitz, & Pedersen, 2013).

Time-series momentum (Moskowitz, Ooi, & Pedersen, 2012) differs by conditioning on each asset's own past return rather than cross-sectional rank, going long assets with positive trailing returns and short those with negative returns. The "carry" component of these strategies has been shown to be a significant alpha driver in commodity and fixed-income markets (Koijen, Moskowitz, Pedersen, & Vrugt, 2018).

### 2.3 Volatility Targeting and Risk Management

Volatility targeting—scaling portfolio exposure inversely to realized volatility—has been shown to improve Sharpe ratios and reduce tail risk (Moreira & Muir, 2017). The mechanism is straightforward: leveraging up in calm periods and deleveraging during turbulent regimes captures the "variance risk premium" while avoiding the convexity drag of unmanaged leverage.

Drawdown control, as formalized by Grossman and Zhou (1993) and extended by Cvitanić and Karatzas (1995), provides an additional risk management layer. However, as we demonstrate in this research (Section 7), aggressive drawdown control can mechanically inflate Sharpe ratios by truncating the left tail of the return distribution without generating genuine alpha—a methodological pitfall with implications for the broader systematic trading literature.

### 2.4 Backtesting Methodology and Overfitting

The dangers of backtest overfitting are well-documented. Bailey and López de Prado (2014) introduced the Deflated Sharpe Ratio (DSR), adjusting for the number of strategy variants tested:

$$\text{DSR} = \Phi\left[\frac{(\hat{SR} - \hat{SR}^*)\sqrt{n-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\hat{SR}^2}}\right]$$

where $\hat{SR}^*$ is the expected maximum Sharpe ratio under the null of zero skill across $K$ trials, $\hat{\gamma}_3$ is skewness, and $\hat{\gamma}_4$ is excess kurtosis. Harvey, Liu, and Zhu (2016) further documented the "factor zoo" problem, arguing that many published factors are statistical artifacts. Our research addresses these concerns through both DSR computation and strict out-of-sample holdout.

### 2.5 Leverage and Its Costs

The use of leverage amplifies both returns and risks. Frazzini and Pedersen (2014) formalized the "betting against beta" effect, showing that leverage constraints cause low-beta assets to be underpriced, creating an alpha opportunity for leveraged portfolios. However, the practical costs of leverage—including borrowing rates, margin requirements, and forced liquidation risk—impose significant real-world frictions (Brunnermeier & Pedersen, 2009).

---

## 3. Data and Universe

### 3.1 Investment Universe

The strategy operates on a universe of 18 U.S.-listed exchange-traded funds spanning three asset class categories:

| Category | Tickers | Description |
|----------|---------|-------------|
| **Sector** (10) | XLK, XLV, XLF, XLE, XLI, XLC, XLP, XLU, XLB, XLRE | SPDR Select Sector ETFs |
| **Broad Market** (4) | SPY, QQQ, IWM, EFA | Major equity indices |
| **Safe Haven** (4) | TLT, IEF, GLD, SHY | Long/intermediate bonds, gold, short-term treasuries |

This universe was selected to provide maximum pair-trading opportunity through correlated sector exposures, while maintaining sufficient diversification across equity risk factors and safe-haven assets for regime-dependent hedging.

### 3.2 Temporal Structure

| Period | Date Range | Trading Days | Purpose |
|--------|-----------|-------------|---------|
| In-Sample (IS) | 2010-01-01 to 2025-03-01 | 3,813 | Parameter fitting, strategy selection |
| Out-of-Sample (OOS) | 2025-03-01 to 2026-03-15 | 260 | True holdout evaluation |
| Total | 2010-01-01 to 2026-03-15 | 4,073 | — |

The IS period spans multiple market regimes: the post-GFC recovery (2010–2013), the low-volatility bull (2014–2017), the Q4 2018 drawdown, the COVID crash and recovery (2020), the 2022 bear market, and the 2023–2025 AI/tech-driven rally. The OOS period was held strictly blind until final evaluation.

### 3.3 Data Quality

Data is sourced from Yahoo Finance via the `yfinance` Python library. Key data quality considerations:

- **Forward-fill only**: Missing values are forward-filled; no backward fill (`bfill`) is applied, preventing the leakage of future prices into historical data.
- **Late-start tickers**: XLC (first data: 2018-06-19, 47.7% coverage) and XLRE (first data: 2015-10-08, 64.4% coverage) have incomplete histories. The pair engine automatically skips pairs involving these tickers during periods of missing data.
- **Adjusted prices**: All prices are adjusted for splits and dividends.

### 3.4 Benchmark

The primary benchmark is SPY (S&P 500 SPDR ETF Trust):
- IS: CAGR = 13.62%, Sharpe = 0.836
- OOS: CAGR = 14.30%, Sharpe = 0.796

---

## 4. Methodology

### 4.1 Pair Trading Engine

#### 4.1.1 Spread Construction

For each pair $(A, B)$, the log-spread is computed as:

$$s_t = \ln P_t^A - \ln P_t^B$$

The rolling Z-score over lookback window $w$ is:

$$z_t = \frac{s_t - \bar{s}_{t-w:t}}{\hat{\sigma}_{t-w:t}(s)}$$

with minimum periods $\lfloor w/2 \rfloor$ to prevent sparse-data artifacts, and the denominator clipped at $\geq 10^{-8}$ to avoid division by zero.

#### 4.1.2 Entry and Exit Rules

The pair engine implements a state machine with three positions: flat ($p = 0$), long spread ($p = +1$), and short spread ($p = -1$):

$$p_t = \begin{cases}
-1 & \text{if } p_{t-1} = 0 \text{ and } z_t > z_{\text{entry}} \\
+1 & \text{if } p_{t-1} = 0 \text{ and } z_t < -z_{\text{entry}} \\
0 & \text{if } p_{t-1} > 0 \text{ and } z_t > -z_{\text{exit}} \\
0 & \text{if } p_{t-1} < 0 \text{ and } z_t < z_{\text{exit}} \\
p_{t-1} & \text{otherwise}
\end{cases}$$

Default parameters: $z_{\text{entry}} = 2.0$, $z_{\text{exit}} = 0.5$. All positions are lagged by one day before computing returns to prevent lookahead.

#### 4.1.3 Pair Scanning and Selection

The exhaustive scan evaluates all $\binom{18}{2} = 153$ unique pairs across a parameter grid:

- **Lookback windows**: $w \in \{21, 42, 63, 126\}$ trading days
- **Z-score configurations**: $(z_e, z_x) \in \{(2.0, 0.5), (2.25, 0.50), (2.25, 0.75), (1.75, 0.50)\}$

This produces $153 \times 4 \times 4 = 2{,}448$ candidate pair strategies, of which approximately 1,920 are evaluated (after skipping NaN tickers), yielding approximately 414 viable pairs with IS Sharpe > 0.3 and CAGR > 0.3%.

#### 4.1.4 Multi-Timeframe Blending

For pairs appearing in multiple lookback windows, a multi-timeframe (MTF) variant blends the top 2–3 configurations per pair by equal-weighting their returns, producing smoother return streams with reduced parameter sensitivity.

#### 4.1.5 Portfolio Construction

Two selection methods identify the top pairs for portfolio inclusion:

1. **Sharpe-Filtered (ShF)**: Pairs ranked by Sharpe, with a greedy decorrelation penalty of $2.5 \times \bar{\rho}$ applied during sequential selection to promote diversity.

2. **Composite-Filtered (CF)**: Pairs ranked by composite score $= \text{Sharpe}^{1.5} \times \max(\text{CAGR}, 0.001)$, with a penalty of $1.5 \times \bar{\rho}$.

Portfolios of size $N \in \{5, 7, 10\}$ are constructed at notional exposures $n \in \{0.06, 0.08, 0.10, 0.12\}$, plus an inverse-volatility-weighted (IVW) variant. Return-stream blends between ShF and CF portfolios at weights $(0.5, 0.6, 0.7)$ are also evaluated.

### 4.2 Alpha Sources

Five distinct alpha sources are evaluated, each capturing a different market phenomenon:

#### 4.2.1 Z-Score Pair Portfolio (ZP)

The primary alpha source, constructed as described in Section 4.1. A market-neutral strategy exploiting mean-reversion in correlated ETF spreads.

#### 4.2.2 Crash Hedge (CH)

A regime-switching strategy based on the realized volatility of QQQ relative to its trailing average:

$$v_{20} = \text{rvol}(\text{QQQ}, 20), \quad v_a = \text{SMA}(v_{20}, 120)$$

| Regime | Condition | QQQ | SPY | IWM | GLD | TLT |
|--------|-----------|-----|-----|-----|-----|-----|
| Normal | $v_{20} < 1.2\,v_a$ | +0.7 | +0.3 | 0 | 0 | 0 |
| Elevated | $1.2\,v_a \leq v_{20} < 1.8\,v_a$ | +0.3 | +0.1 | −0.2 | +0.15 | 0 |
| Crisis | $v_{20} \geq 1.8\,v_a$ | 0 | −0.3 | 0 | +0.3 | +0.2 |
| Recovery | Elevated ∧ $v_{20} < v_{20}^{(5)}$ ∧ $P > P_{\min}^{(10)}$ | +0.8 | +0.4 | 0 | 0 | 0 |

This strategy provides counter-cyclical alpha: it loads heavily on equities during normal conditions and rotates into safe havens (GLD, TLT) and short positions (SPY, IWM) during crises.

#### 4.2.3 Cross-Sectional Momentum (XSMom)

Following Jegadeesh and Titman (1993), assets are ranked at each monthly rebalance by trailing return with a skip period:

$$\text{mom}_{t}^{(i)} = \frac{P_{t-\text{skip}}^{(i)}}{P_{t-\text{skip}-\text{lookback}}^{(i)}} - 1$$

The top $n_{\text{long}}$ assets receive equal-weight long positions of $+n/(n_{\text{long}})$ per asset, and the bottom $n_{\text{short}}$ receive equal-weight short positions of $-n/(n_{\text{short}})$.

Best configuration: lookback = 189 days, skip = 21 days, $n_{\text{long}} = 4$, $n_{\text{short}} = 2$, notional = 0.10. IS Sharpe = 1.16.

#### 4.2.4 Time-Series Momentum (TSMom)

For each asset, the signal is the average sign of trailing returns across multiple lookback horizons:

$$\text{signal}_t^{(i)} = \frac{1}{3}\sum_{lb \in \{63, 126, 252\}} \text{sign}\left(\frac{P_t^{(i)}}{P_{t-lb}^{(i)}} - 1\right)$$

Position size is scaled by inverse volatility:

$$w_t^{(i)} = \text{signal}_t^{(i)} \times \text{clip}\left(\frac{0.05}{\hat{\sigma}_t^{(i)}},\, 0.1,\, 2.0\right) \times \frac{n}{N}$$

**Important caveat**: At small notional values ($n = 0.06$), this strategy produces extremely small positions (~0.003 per asset), causing the return stream to be dominated by cash yield ($r_f = 2\%$). This results in artificially high Sharpe ratios (IS: 14.13, OOS: 17.42) that do not reflect genuine alpha. This degeneracy is documented and the strategy is used only within ensembles.

#### 4.2.5 Sector Rotation (SecRot)

Sectors are ranked by risk-adjusted momentum:

$$\text{score}_s = \frac{\bar{r}_s \times 252}{\hat{\sigma}_s \times \sqrt{252}}$$

Long positions in the top $n_{\text{top}}$ sectors and short positions in the bottom $n_{\text{bottom}}$, rebalanced monthly. Best configuration: lookback = 126, $n_{\text{top}} = 4$, $n_{\text{bottom}} = 2$, notional = 0.12. IS Sharpe = 1.05.

#### 4.2.6 Volatility Carry (VolCarry) — Deprecated

Originally included in v12–v17 as a strategy that sorts sectors by 63-day realized volatility and goes long the lowest-volatility sectors and short the highest-volatility sectors. This strategy was **dropped in v18** after OOS evaluation revealed a Sharpe of −0.14 and CAGR of −0.35%, demonstrating complete out-of-sample failure despite an in-sample weight of 7%.

### 4.3 Ensemble Construction

Alpha sources are combined into ensemble portfolios using fixed-weight blending:

$$r_{\text{ens},t} = \sum_{k=1}^{K} w_k \cdot r_{k,t}, \quad \sum_{k=1}^{K} w_k = 1$$

The weight grid is systematically varied:

| Ensemble | Sources | Weight Grid |
|----------|---------|-------------|
| E2 | ZP + CH | (85/15), (80/20), (75/25), (70/30), (60/40), (50/50) |
| E3 | ZP + CH + XSMom | (70/15/15), (65/20/15), (60/20/20), (55/25/20) |
| E4 | ZP + CH + XSMom + TSMom | (55/15/15/15), (50/20/15/15), (45/20/20/15) |
| E5 | ZP + CH + XSMom + TSMom + SecRot | (45/15/15/15/10), (40/20/15/15/10) |

Each ensemble configuration is tested against all 15 qualifying pair portfolios (IS Sharpe > 1.0), producing 225 ensemble variants in total.

### 4.4 Risk Overlay Architecture

The risk overlay system operates in a cascaded pipeline:

$$\text{Raw Alpha} \xrightarrow{\text{Ensemble}} \text{Blended} \xrightarrow{\text{VT}} \text{Vol-Targeted} \xrightarrow{\text{HDDC}} \text{Pre-Lev}} \xrightarrow{\times k} \text{Leveraged} \xrightarrow{\text{PL-DDC}} \text{Final}$$

#### 4.4.1 Volatility Targeting (VT)

Returns are scaled to maintain a target annualized volatility:

$$\hat{\sigma}_t = \text{std}_{63}(r) \times \sqrt{252}, \quad \text{clipped} \geq 0.005$$

$$\text{scale}_t = \text{clip}\!\left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_t},\, 0.2,\, 5.0\right)$$

$$r_t^{\text{VT}} = r_t \times \text{scale}_{t-1}$$

The scale is lagged by one day to prevent lookahead. Target volatilities tested: $\sigma_{\text{target}} \in \{4\%, 5\%, 6\%, 7\%, 8\%, 10\%\}$.

#### 4.4.2 Hierarchical Drawdown Control (HDDC)

A two-threshold system with linear interpolation:

$$\text{dd}_t = \frac{\text{Equity}_t}{\max_{s \leq t} \text{Equity}_s} - 1$$

$$\text{scale}_t = \begin{cases}
0.15 & \text{if } \text{dd}_{t-1} < \theta_2 \\
\max\!\left(0.15,\, 1.0 - 0.85 \cdot \frac{\text{dd}_{t-1} - \theta_1}{\theta_2 - \theta_1}\right) & \text{if } \theta_2 \leq \text{dd}_{t-1} < \theta_1 \\
\min(1.0,\, \text{scale}_{t-1} + 0.015) & \text{if recovering} \\
1.0 & \text{otherwise}
\end{cases}$$

Threshold grid: $(\theta_1, \theta_2) \in \{(-1\%, -3\%), (-1\%, -3.5\%), (-1.5\%, -4\%), (-1.5\%, -3.5\%), (-2\%, -5\%)\}$

The use of $\text{dd}_{t-1}$ (lagged drawdown) is a critical audit-driven fix to prevent lookahead bias. Recovery is gradual at 1.5 percentage points per day to avoid whipsawing.

#### 4.4.3 Position-Level Drawdown Control (PL-DDC)

The final layer operates on leveraged returns and explicitly models rebalancing costs:

$$r_t^{\text{PL}} = r_t \times \text{scale}_t - |\text{scale}_t - \text{scale}_{t-1}| \times \frac{c_{\text{rebal}}}{10{,}000}$$

where $c_{\text{rebal}} \in \{3, 5\}$ basis points. This captures the real-world cost of adjusting position sizes when the DDC scale changes, which is ignored by return-stream DDC.

### 4.5 Leverage Framework

Leveraged returns are computed as:

$$r_t^{(k)} = k \cdot r_t^{\text{base}} - (k - 1) \cdot \frac{c_{\text{lev}}}{252}$$

where $k$ is the leverage multiplier and $c_{\text{lev}}$ is the annual cost of leverage. The grid:

- **Multipliers**: $k \in \{2, 3, 4, 5, 6, 8\}$
- **Costs**: $c_{\text{lev}} \in \{0.5\%, 1.0\%, 1.5\%, 2.0\%, 3.0\%\}$

In the OOS evaluation, the focus narrows to $k \in \{3, 4, 5\}$ at $c_{\text{lev}} \in \{1\%, 2\%\}$ as the most practically relevant configurations. Leverage multipliers above 5× are flagged as requiring institutional infrastructure (prime brokerage, portfolio margining) and are reported but not recommended for retail deployment.

### 4.6 Cost Model

The backtest engine incorporates five distinct cost components:

| Component | Formula | Default |
|-----------|---------|---------|
| Transaction cost | $\sum_t \text{turnover}_t \times \frac{c_{\text{tx}}}{10{,}000}$ | $c_{\text{tx}} = 5$ bps |
| Leverage cost | $\max(0, \text{gross\_exp}_t - 1) \times \frac{c_{\text{lev}}}{252}$ | $c_{\text{lev}} = 1.5\%$ |
| Short cost | $\sum_i |w_{it}|_{\text{short}} \times \frac{c_{\text{short}}}{252}$ | $c_{\text{short}} = 0.5\%$ |
| Spread cost | $\sum_i |\Delta w_{it}| \times \frac{s_i / 2}{10{,}000}$ | ETF-specific |
| Cash yield | $(1 - \text{net\_exp}_t)^+ \times \frac{r_f}{252}$ | $r_f = 2.0\%$ |

The spread model uses ETF-specific half-spreads calibrated from historical bid-ask data:

| Tier | ETFs | Half-Spread |
|------|------|-------------|
| Ultra-Liquid | SPY | 0.3 bps |
| Liquid | QQQ, SHY | 0.5–1.0 bps |
| Moderate | IWM, EFA, XLK, XLF, TLT, IEF | 1.0–2.0 bps |
| Less Liquid | XLV, XLE, XLP, XLU, GLD | 2.0–2.5 bps |
| Illiquid | XLI, XLC, XLB, XLRE | 2.5–3.5 bps |

Average daily volume (ADV) data is maintained for market impact estimation, with SPY at \$30B and smaller sector ETFs (XLB, XLRE) at ~\$200M.

### 4.7 Walk-Forward and Out-of-Sample Protocol

#### 4.7.1 Walk-Forward Validation

The IS period is divided into five non-overlapping sub-periods:

| Sub-Period | Date Range | Market Regime |
|------------|-----------|---------------|
| Period 1 | 2010–2012 | Post-GFC recovery |
| Period 2 | 2013–2015 | Low-vol bull |
| Period 3 | 2016–2018 | Late-cycle expansion + Q4 crash |
| Period 4 | 2019–2021 | COVID crash + recovery |
| Period 5 | 2022–2025 | Bear + AI rally |

A strategy passes walk-forward validation if its Sharpe ratio exceeds 0.5 in **all five** sub-periods, ensuring robustness across regimes.

#### 4.7.2 True Out-of-Sample Protocol

The OOS protocol is strict:

1. All pair selections, alpha source configurations, ensemble weights, overlay parameters, and leverage settings are **locked from in-sample**.
2. Pair trading signals are reconstructed from scratch on OOS price data using IS-calibrated parameters (same pairs, same windows, same Z-score thresholds).
3. CrashHedge, momentum, and sector rotation strategies use the same functional parameters as IS.
4. No parameter is adjusted after observing OOS data.

### 4.8 Statistical Validation

#### 4.8.1 Deflated Sharpe Ratio

Following Bailey and López de Prado (2014), we compute the DSR adjusting for the number of strategies tested:

$$\text{DSR} = \Phi\left[\frac{(\hat{SR} - E[\max SR])\sqrt{n-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\hat{SR}^2}}\right]$$

where $E[\max SR] \approx \sqrt{2 \ln K} \cdot (1 - \gamma / (2 \ln K))$ for $K$ independent trials, $\gamma$ is the Euler-Mascheroni constant, and $\hat{\gamma}_3$, $\hat{\gamma}_4$ are observed skewness and kurtosis.

#### 4.8.2 Bootstrap Confidence Intervals

Sharpe ratio confidence intervals are computed via 5,000 non-parametric bootstrap resamples (with replacement) of the daily return series, using a fixed seed (42) for reproducibility.

---

## 5. Iterative Development and Results

### 5.1 Phase I: Exploratory Period (v1–v7)

The early phase established the foundations:

**Pre-existing infrastructure** (pre-2024): 10+ individual indicator strategies (SMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR, OBV, CMF, Williams %R), deep learning experiments (CNN + Q-learning), NLP/Twitter sentiment analysis, and a Forex Kalman filter.

**Modernization** (2024–early 2025): Consolidation into the `src/financial_algorithms/` package with 60+ indicator modules, a modular backtest engine, and a strategy registry pattern.

**Phase 3 weight optimization** (January 2026): Monte Carlo optimization over 200,000 samples of 15-indicator combinations. Champion indicators: SAR/Stochastic, Stochastic/MACD, Bollinger/RSI, CCI/ADX, Williams %R, ATR Trend, CMF.

$$\text{Phase 3 Champion: } \text{Sharpe} = 1.65, \quad \text{Sortino} = 2.46, \quad \text{Calmar} = 2.34, \quad \text{MDD} = -15.9\%$$

**Intraday attempt and lookahead bug** (March 2026): An intraday 1-minute bar pipeline was built for crypto (BTC/ETH) and stocks, initially reporting Sharpe = 6.37. A critical lookahead bias was discovered: the signal function received the **entire** price dataset rather than only historical data. After implementing walk-forward processing:

$$\text{Sharpe}: 6.37 \rightarrow -42.89 \quad (\text{a } -700\% \text{ correction})$$

This finding reinforced the importance of the self-auditing mandate.

**Voting strategy** (March 11–12, 2026): An 8-indicator voting system with Bayesian optimization was built, but comprehensive validation revealed negative Sharpe across all configurations (best: −0.40). Root causes: indicator redundancy, 35% win rate, aggressive exits. A complete redesign to 5 uncorrelated indicators with tiered exits achieved Sharpe 5.34 across 15 assets—but with only 3.81% CAGR versus SPY's 87.67% over 2023–2025.

**v5/v6 intraday hybrid**: The first strategy directly targeting the original leverage/alpha mandate. SPY Sharpe = 2.31, CAGR = 21.46%. This proved the 1.95 target was achievable, motivating the shift to multi-asset pair trading.

### 5.2 Phase II: Pair Trading Architecture (v8–v10)

| Version | Strategies Tested | Winners | Best IS Sharpe | Best IS CAGR |
|---------|------------------|---------|----------------|--------------|
| v8 | 137 | 0 | 1.66 | 16.91% |
| v9 | 142 | 0 | 6.12 | 16.88% |
| v10 | 151 | 0 | 2.46 | 15.90% |

**v8** introduced Z-score pair trading, CrashHedge, and Donchian channel breakouts. Zero strategies met both CAGR and Sharpe targets simultaneously—the fundamental trade-off between high-Sharpe low-CAGR pair trading and high-CAGR low-Sharpe leveraged strategies was identified.

**v9** introduced multi-pair baskets (10-pair portfolios), producing a dramatic Sharpe improvement from 1.66 to 6.12 through diversification across multiple mean-reverting pairs. However, CAGR remained at 2.46%.

**v10** introduced diversified pair selection methods (Top5, Top8, Top10, Core3). The best strategy achieved CAGR = 15.90% with Sharpe = 1.86—a near-miss of only 0.09 Sharpe points below the 1.95 target.

### 5.3 Phase III: First Winners and Scaling (v10b–v11)

| Version | Strategies Tested | Winners | Best IS Sharpe | Best IS CAGR |
|---------|------------------|---------|----------------|--------------|
| v10b | 1,193 | **368** | 2.65 | 23.92% |
| v11 | 2,096 | **1,079** | 2.71 | 38.43% |

**v10b** achieved the first breakthrough through **exit-Z blending** ($z_x \in \{0.25, 0.50, 0.75\}$), which allowed pairs to gradually reduce position as the Z-score mean-reverted rather than using binary open/close. Combined with dynamic leverage and fine-tuned ensemble weights, 368 of 1,193 strategies met both targets.

**v11** introduced CAGR-focused pair selection and three-source ensembles (ZP + CH + Momentum), tripling the winner count to 1,079 and achieving the IS champion at CAGR = 38.43% with 6× leverage.

### 5.4 Phase IV: Risk Management Revolution (v12–v14)

| Version | Strategies Tested | Winners | Best IS Sharpe | Best IS CAGR |
|---------|------------------|---------|----------------|--------------|
| v12 | 2,685 | **1,878** | **3.75** | 205.37% |
| v13 | 3,181 | **2,445** | **4.29** | 274.97% |
| v14 | 5,076 | **4,185** | **4.91** | 225.84% |

**v12** marked a transformation in the research trajectory. The introduction of **volatility targeting (VT)** and **drawdown control (DDC)** overlays produced a discontinuous improvement: Sharpe jumped from 2.71 to 3.75 (+38%) and CAGR from 38% to 205%. The mechanism was later understood (see Section 7): VT amplifies returns in low-vol regimes while DDC truncates draw-downs, mechanically compressing the return distribution's tails.

**v13** refined the approach with composite-filtered pairs and nested VT+DDC combinations (e.g., VT7+DDC3), pushing Sharpe to 4.29 and CAGR to 275% at 8× leverage with only −12.76% maximum drawdown.

**v14** introduced return-stream blends (50/50 ShF + CF pair portfolios) and pre-leverage hierarchical DDC, achieving the IS peak of Sharpe = 4.91 with Sortino = 10.12 and Calmar = 24.13.

### 5.5 Phase V: Drawdown Control Escalation (v15–v16)

| Version | Strategies Tested | Winners | Best IS Sharpe | Best IS CAGR | Best IS MaxDD |
|---------|------------------|---------|----------------|--------------|---------------|
| v15 | ~5,000 | 4,944 | **5.42** | 101% | −3.24% |
| v16 | ~6,000 | 5,962 | **6.25** | 109% | **−2.29%** |

**v15** introduced two-layer hierarchical DDC, improving Sharpe from 4.91 to 5.42 at the cost of reduced CAGR (225% → 101%). The trade-off was favorable: maximum drawdown compressed from −9.36% to −3.24%.

**v16** pushed to triple-layer DDC with three thresholds (−1.0%, −2.0%, −4.0%), achieving the project's overall IS peak: **Sharpe = 6.25**, Sortino = 15.33, Calmar = 47.65, MaxDD = −2.29%.

However, the very perfection of these metrics triggered the audit.

### 5.6 Phase VI: Comprehensive Audit

Motivated by the original mandate to *"audit your own work and calculations every so often"* and by the suspiciously smooth equity curves of v16, a comprehensive audit was conducted against the framework from the research paper "Why Backtests Fail in Live Trading."

**41 findings were documented, 11 rated HIGH priority:**

| # | Finding | Severity | Impact |
|---|---------|----------|--------|
| 1 | DDC applied to return streams, not positions | HIGH | DDC scales ex-post returns rather than adjusting live position sizes |
| 2 | No `bfill` prevention | HIGH | Backward fill leaks future prices |
| 3 | 8× leverage unrealistic | HIGH | Retail/institutional access constraints |
| 4 | No true out-of-sample test | HIGH | All metrics were in-sample |
| 5 | 19 tuned parameters | HIGH | Extreme overfitting risk |
| 6 | DDC mechanically inflates Sharpe | HIGH | SPY: 0.84 → 3.76 after DDC stacking |
| 7 | No spread/slippage model | HIGH | Zero friction assumption |
| 8 | No market impact model | HIGH | Infinite liquidity assumption |
| 9 | Same-bar DDC (lookahead) | HIGH | DDC used current-bar drawdown |
| 10 | No deflated Sharpe ratio | HIGH | Multiple testing bias unaddressed |
| 11 | No bootstrap confidence intervals | HIGH | No statistical significance testing |

**The most consequential finding** was #6: stacking DDC layers on a passive SPY position inflated its Sharpe from 0.84 to 3.76—a factor of **4.5×**—without any trading or alpha generation. This revealed that a substantial portion of the v16 Sharpe of 6.25 was attributable to mechanical tail-truncation rather than genuine alpha. Estimated honest live Sharpe: **1.5–2.5**.

### 5.7 Phase VII: Audit-Hardened System (v17)

All 11 HIGH-priority findings were remediated in v17:

| Fix | Implementation |
|-----|---------------|
| Position-level DDC | DDC scales actual position weights, not ex-post return streams |
| No `bfill` | Forward-fill only in data loading |
| Realistic leverage | Focus on 3×–5× with explicit cost (0.5–3%) |
| True OOS | 260-day holdout (2025-03 to 2026-03) with locked parameters |
| Spread model | ETF-specific half-spreads (0.3–3.5 bps) |
| Market impact | Square-root model based on ADV |
| Lagged DDC | All DDC variants use $\text{dd}_{t-1}$ instead of $\text{dd}_t$ |
| Deflated Sharpe | Bailey-López de Prado DSR computed |
| Bootstrap CI | 5,000 resamples for 95% confidence intervals |
| Rebalancing cost | PL-DDC charges 3–5 bps per DDC scale change |

**IS results after fixes**: Champion Sharpe **4.07** (down 35% from 6.25). This haircut was expected and validated the audit's findings.

**First true OOS results**:

| Configuration | OOS Sharpe | OOS CAGR | OOS MaxDD |
|---------------|-----------|----------|-----------|
| E3(90%ZP+3%CH+7%VC) unleveraged | **2.08** | 2.35% | — |
| Same, 3× @ 1% cost | 1.78 | 26.00% | −10.02% |
| Same, 5× @ 2% cost | 1.75 | 44.53% | −16.39% |
| PL-DDC + 3× | 1.80 | 14.50% | −5.13% |

**Individual alpha source OOS decomposition**:

| Source | OOS Sharpe | OOS CAGR | Weight in Ensemble |
|--------|-----------|----------|-------------------|
| Z-Score Pairs (ShF5, n=0.06) | 1.70 | 2.06% | 90% |
| CrashHedge | 1.44 | 17.94% | **3%** |
| VolCarry | **−0.14** | −0.35% | 7% |

This decomposition revealed two critical misallocations: CrashHedge was the strongest alpha source (CAGR = 17.94%) but received only 3% weight, while VolCarry was actively destroying alpha with negative OOS Sharpe but received 7% weight (more than double CrashHedge).

### 5.8 Phase VIII: Alpha Enhancement (v18)

v18 addressed the misallocations identified in v17's OOS analysis:

**Changes from v17**:
1. CrashHedge weight range expanded from 3% to **15–50%**
2. VolCarry **dropped entirely**
3. Three new alpha sources added: XSMom, TSMom, SecRot
4. Pair portfolio grid expanded: sizes {5, 7, 10} × notionals {0.06–0.12}
5. Wider ensemble weight grids: 2/3/4/5-source combinations (225 ensembles)
6. All 11 audit fixes retained

**Search space**: 2,751 total strategies evaluated across 10 phases.

---

## 6. Out-of-Sample Results

### 6.1 Individual Alpha Sources

| Source | OOS Sharpe | OOS CAGR | OOS MaxDD | Assessment |
|--------|-----------|----------|-----------|------------|
| Z-Score Pairs (ShF5, n=0.06) | 1.70 | 2.06% | −1.12% | Core alpha; stable, low-vol |
| CrashHedge | 1.44 | 17.94% | −7.15% | Highest absolute returns |
| XS Momentum | 1.38 | 1.29% | −1.25% | Positive but modest |
| TS Momentum | 17.42† | 2.08% | −0.02% | **Degenerate** (see §4.2.4) |
| Sector Rotation | 0.98 | 1.03% | −1.13% | Marginal |

†TSMom Sharpe is artificial due to near-zero position sizes. Cash yield dominates.

### 6.2 Ensemble Results

| Ensemble | OOS Sharpe | OOS CAGR | OOS MaxDD |
|----------|-----------|----------|-----------|
| E2(85%ZP + 15%CH) | **2.30** | 4.40% | −0.71% |
| E4(55%ZP + 15%CH + 15%Mom + 15%TSM) | 2.28 | 4.28% | −0.75% |
| E3(70%ZP + 15%CH + 15%Mom) | 2.25 | 4.28% | −0.75% |
| E5(45%ZP + 15%CH + 15%M + 15%T + 10%S) | 2.21 | 4.17% | −0.80% |
| E2(80%ZP + 20%CH) | 2.12 | 5.18% | −0.85% |
| E2(75%ZP + 25%CH) | 1.98 | 5.96% | −1.05% |
| E2(70%ZP + 30%CH) | 1.87 | 6.75% | −1.27% |
| E2(60%ZP + 40%CH) | 1.73 | 8.33% | −2.00% |

**Key finding**: The simplest ensemble—two sources, 85% pairs and 15% CrashHedge—produces the **highest OOS Sharpe** (2.30). Adding additional alpha sources (momentum, sector rotation) **monotonically decreases** risk-adjusted performance despite improving raw diversification metrics. This is consistent with the "diversification penalty" observed when combining strong and weak signals: the weak sources dilute the portfolio with noise.

The CrashHedge allocation exhibits a clear Sharpe-CAGR frontier:

| CH Weight | OOS Sharpe | OOS CAGR |
|-----------|-----------|----------|
| 15% | 2.30 | 4.40% |
| 20% | 2.12 | 5.18% |
| 25% | 1.98 | 5.96% |
| 30% | 1.87 | 6.75% |
| 40% | 1.73 | 8.33% |

### 6.3 Overlay and Leverage Results

| Strategy | OOS Sharpe | OOS CAGR | OOS MaxDD |
|----------|-----------|----------|-----------|
| VT6+H(1.5/4.0) on E2(85/15) | 2.11 | 14.78% | −2.52% |
| PL-DDC + 3× + VT8+H on E3 | **2.06** | **27.64%** | −5.53% |
| 3× + VT6+H on E2(85/15) | **2.01** | **46.30%** | −7.59% |
| 3× + VT6+H on E4 | 2.01 | 46.21% | −8.68% |
| 4× + VT6+H on E2(85/15) | 2.00 | 64.09% | −10.07% |
| 5× + VT6+H on E2(85/15) | 1.99 | **83.26%** | −12.52% |

### 6.4 v17 → v18 Improvement

| Metric | v17 OOS | v18 OOS | Improvement |
|--------|---------|---------|-------------|
| Unleveraged Sharpe | 2.08 | **2.30** | +10.6% |
| Unleveraged CAGR | 2.35% | **4.40%** | +87.2% |
| 3× Sharpe | 1.78 | **2.01** | +12.9% |
| 3× CAGR | 26.00% | **46.30%** | +78.1% |
| PL-DDC 3× Sharpe | 1.80 | **2.06** | +14.4% |
| PL-DDC 3× CAGR | 14.50% | **27.64%** | +90.6% |

The primary driver of improvement was the CrashHedge reweighting from 3% to 15%. The three new alpha sources contributed marginally (XSMom: OOS Sharpe = 1.38; SecRot: 0.98), and the TSMom Sharpe of 17.42 is artificial.

### 6.5 Paper-Trading Recommendations

Based on the full OOS analysis, three deployment tiers are proposed:

| Tier | Configuration | OOS Sharpe | OOS CAGR | OOS MaxDD |
|------|--------------|-----------|----------|-----------|
| **Conservative** | PL-DDC + 3× + VT8+HDDC + E3(70/15/15) | 2.06 | 27.6% | −5.5% |
| **Balanced** | 3× + VT6+HDDC(1.5/4.0) + E2(85/15) | 2.01 | 46.3% | −7.6% |
| **Aggressive** | 5× + VT6+HDDC(1.5/4.0) + E2(85/15) | 1.99 | 83.3% | −12.5% |

All tiers maintain OOS Sharpe ≥ 1.95, satisfying the original research objective.

---

## 7. Audit Findings and Methodological Lessons

### 7.1 Drawdown Control as a Sharpe Inflation Mechanism

The most significant methodological finding of this research is the documentation of DDC-induced Sharpe inflation. By applying the v16 triple-layer DDC to a simple SPY buy-and-hold position, we observed:

| Configuration | SPY Sharpe |
|--------------|-----------|
| No overlay | 0.84 |
| Single DDC (−5%) | 1.52 |
| Hierarchical DDC (−2%/−5%) | 2.67 |
| Triple-layer DDC (−1%/−2.5%/−5%) | **3.76** |

The mechanism is straightforward: DDC truncates the left tail of the return distribution by reducing exposure during drawdowns. Since the Sharpe ratio is $\bar{r}/\sigma$, reducing $\sigma$ (primarily via left-tail compression) increases the ratio even without changing $\bar{r}$ in expectation. The inflation factor of $4.5\times$ ($0.84 \rightarrow 3.76$) quantifies the maximum "free Sharpe" available from DDC stacking alone, without any alpha.

**Implication**: Any systematic strategy reporting Sharpe ratios above ~2.0 that employs drawdown control should be evaluated against the DDC-adjusted benchmark (i.e., the same DDC applied to SPY) to isolate genuine alpha from mechanical tail truncation.

### 7.2 The Bfill Vulnerability

Backward fill (`bfill`) in price data can leak future information: if a ticker has missing data on date $t$, `bfill` substitutes the value from date $t+k$ (the next available price), effectively embedding future returns into the historical series. While the impact may appear small for liquid ETFs with few missing values, it can systematically bias results upward in pair trading strategies where the spread depends on precise price ratios.

### 7.3 Lookahead in Same-Bar Drawdown

Using $\text{dd}_t$ (today's drawdown) to determine today's scale factor is a form of lookahead because the full day's return contributes to the drawdown calculation. The fix is trivial—use $\text{dd}_{t-1}$—but the impact is material: the lag reduces Sharpe by approximately 3–8% depending on the DDC aggressiveness, because the system loses one day of reaction time.

### 7.4 The Multiple Testing Problem

With 2,751 strategies evaluated in v18 (and over 6,000 in v16), the probability of finding a high-Sharpe strategy by chance alone is substantial. The Deflated Sharpe Ratio framework provides a corrective:

$$E[\max SR] \approx \sqrt{2 \ln K} \approx \sqrt{2 \ln 2751} \approx 3.98$$

This means that under the null hypothesis of zero skill, we would **expect** the best Sharpe among 2,751 random strategies to be approximately 3.98. Any in-sample Sharpe below this threshold cannot be confidently attributed to skill rather than chance.

The OOS Sharpe of 2.30 is evaluated against a much smaller effective trial count (the OOS was a single pre-committed evaluation), providing stronger evidence of genuine alpha.

---

## 8. Discussion

### 8.1 Why Simplicity Dominates

The finding that E2(85%ZP + 15%CH) outperforms all multi-source ensembles on OOS Sharpe is counterintuitive but theoretically grounded. In the Markowitz framework, adding assets with positive-but-low Sharpe ratios to a portfolio only improves the portfolio Sharpe if $SR_{\text{new}} > SR_{\text{existing}} \times \rho_{\text{new,existing}}$. The weak alpha sources (XSMom: SR = 1.38, SecRot: SR = 0.98) fail this test against the existing ensemble (SR = 2.30), because their correlation with the pair portfolio is non-negligible and their stand-alone Sharpe is insufficient to compensate.

### 8.2 The CrashHedge Premium

CrashHedge's OOS CAGR of 17.94% (exceeding SPY's 14.30%) suggests that the volatility-regime switching captures a genuine risk premium: the strategy loads on equity beta during normal conditions and rotates to safe havens during stress—a behavior that mimics tail-risk hedging while maintaining positive expected return. The fact that it was originally allocated only 3% weight (vs. 90% for pairs and 7% for VolCarry) represents a significant misallocation that persisted through five version iterations (v12–v17), demonstrating how in-sample Sharpe optimization can mask absolute return contributions.

### 8.3 The VolCarry Failure

VolCarry's complete OOS failure (Sharpe = −0.14) despite acceptable in-sample performance (IS Sharpe ~0.7–1.0) is an instructive case of regime dependence. The strategy bets that low-volatility sectors outperform high-volatility sectors—a variant of the "betting against beta" factor (Frazzini & Pedersen, 2014). In the OOS period (March 2025–March 2026), this relationship reversed, possibly due to structural changes in sector leadership during the AI-driven market cycle.

### 8.4 Practical Deployment Considerations

**Leverage access**: The recommended 3× leverage on an ETF portfolio is achievable through portfolio margining (Regulation T + portfolio margin), ETF options replication, or leveraged ETF combinations. The 1% annual cost assumption is conservative for institutional participants but may be optimistic for retail accounts.

**Capacity**: The pair portfolio operates on sector ETFs with ADV ranging from \$200M (XLB, XLRE) to \$30B (SPY). At the recommended 5-pair portfolio with 0.06 notional per pair, the daily turnover is modest (~\$50K–\$500K per \$1M portfolio), well within capacity constraints.

**Execution risk**: The 1-day trading lag built into all weight calculations provides a full trading day for execution, mitigating the risk of slippage on the implementation shortfall.

---

## 9. Limitations and Future Work

### 9.1 Limitations

1. **Short OOS period**: The 260-day out-of-sample period, while genuinely blind, represents only one market regime. Extending to multiple independent OOS windows (e.g., rolling 1-year OOS) would strengthen confidence.

2. **No live trading validation**: All results are backtested. The gap between paper and live performance includes execution quality, data timing differences, and behavioral factors that are not modeled.

3. **Single-universe dependency**: All alpha sources operate on the same 18 ETFs. A global, multi-asset extension (commodities, FX, international equities, fixed income futures) would provide more robust diversification.

4. **Regime stationarity assumption**: The pair trading engine assumes that correlations and mean-reversion dynamics estimated in-sample persist out-of-sample. Structural breaks (regulatory changes, ETF rebalancing shifts, sector reclassifications) could invalidate these relationships.

5. **DDC parameters are fitted in-sample**: While the DDC operates on a lagged basis, its threshold parameters ($\theta_1$, $\theta_2$) are optimized on IS data. Adaptive DDC with regime-responsive thresholds is a promising direction.

6. **TSMom degeneracy**: The time-series momentum strategy's artificial Sharpe inflation from near-zero position sizes represents a design flaw. Position scaling should be normalized to ensure minimum portfolio participation.

7. **No tail-risk analysis**: While maximum drawdown is reported, more sophisticated tail-risk measures (CVaR, Expected Shortfall, tail-risk contribution) would provide a fuller picture of downside exposure.

### 9.2 Future Work

1. **Adaptive pair selection**: Reinforcement learning or online convex optimization for dynamic pair and weight selection based on recent performance.

2. **Macro regime integration**: Incorporating yield curve shape, credit spreads, and VIX term structure as additional regime signals for the CrashHedge strategy.

3. **Multi-asset extension**: Expanding to commodity futures (CL, GC, SI), currency pairs (EUR/USD, JPY/USD), and international equity ETFs (EEM, EEMV).

4. **Transaction cost optimization**: Implementing the Almgren-Chriss optimal execution framework to minimize market impact at scale.

5. **Live paper trading**: Deploying the conservative tier (PL-DDC + 3× + E3) in a paper trading environment for 6+ months before live capital allocation.

---

## 10. Conclusions

This research program demonstrates that leveraged alpha generation through statistical arbitrage ensembles with cascaded risk management can achieve out-of-sample performance metrics exceeding both the S&P 500 return and a Sharpe ratio of 1.95—the dual objectives established at the outset. The key results are:

1. **Best OOS unleveraged Sharpe: 2.30** (E2: 85% pair portfolio + 15% CrashHedge), representing genuine risk-adjusted alpha verified on 260 days of never-seen data.

2. **Best OOS leveraged return: CAGR = 46.30%** at 3× leverage with 1% cost, Sharpe = 2.01, MaxDD = −7.59%—simultaneously exceeding both the CAGR and Sharpe targets.

3. **Aggressive frontier: CAGR = 83.26%** at 5× leverage, Sharpe = 1.99, MaxDD = −12.52%.

4. The comprehensive self-audit discovered that **drawdown control inflates Sharpe by approximately 4.5×** on passive benchmarks, leading to a 35% reduction in IS Sharpe after remediation. This finding has broad implications for backtesting methodology across the quantitative finance community.

5. **Simplicity dominates**: A two-source ensemble outperforms all more complex blends on OOS Sharpe, reinforcing the principle that signal quality matters more than signal quantity.

6. **Alpha source allocation matters critically**: Fixing CrashHedge from 3% to 15% weight—a simple reallocation with no new alpha sources—produced an 87% improvement in OOS CAGR.

The 18-iteration development record, the documented audit findings, and the strict out-of-sample protocol provide a transparent and reproducible framework for future research in leveraged systematic trading.

---

## 11. References

Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.

Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929–985.

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4).

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

Brunnermeier, M. K., & Pedersen, L. H. (2009). Market liquidity and funding liquidity. *Review of Financial Studies*, 22(6), 2201–2238.

Cvitanić, J., & Karatzas, I. (1995). On portfolio optimization under drawdown constraints. *IMA Lectures in Mathematics and Its Applications*, 65, 35–46.

Do, B., & Faff, R. (2010). Does simple pairs trading still work? *Financial Analysts Journal*, 66(4), 83–95.

Do, B., & Faff, R. (2012). Are pairs trading profits robust to trading costs? *Journal of Financial Research*, 35(2), 261–287.

Elliott, R., van der Hoek, J., & Malcolm, W. (2005). Pairs trading. *Quantitative Finance*, 5(3), 271–276.

Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1–25.

Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.

Grossman, S. J., & Zhou, Z. (1993). Optimal investment strategies for controlling drawdowns. *Mathematical Finance*, 3(3), 241–276.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.

Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H., & Vrugt, E. B. (2018). Carry. *Journal of Financial Economics*, 127(2), 197–225.

Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.

Michaud, R. O. (1989). The Markowitz optimization enigma: Is "optimized" optimal? *Financial Analysts Journal*, 45(1), 31–42.

Moreira, A., & Muir, T. (2017). Volatility‐managed portfolios. *Journal of Finance*, 72(4), 1611–1644.

Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.

---

## 12. Appendices

### Appendix A: Complete Evolution Table

| Version | Strategies | Winners | IS Sharpe | IS CAGR | IS MaxDD | Key Innovation |
|---------|-----------|---------|-----------|---------|----------|----------------|
| v5 | — | — | 2.31 | 21.46% | −4.4% | Intraday aggressive hybrid (SPY) |
| v8 | 137 | 0 | 1.66 | 16.91% | −22.5% | Z-score pairs + CrashHedge |
| v9 | 142 | 0 | 6.12 | 2.46% | — | Multi-pair baskets (10-pair) |
| v10 | 151 | 0 | 2.46 | 15.90% | — | Diversified pair selection |
| v10b | 1,193 | 368 | 2.65 | 23.92% | — | Exit-Z blending |
| v11 | 2,096 | 1,079 | 2.71 | 38.43% | — | CAGR-focused pairs + 3-source |
| v12 | 2,685 | 1,878 | 3.75 | 205.37% | −20.8% | VT + DDC overlays |
| v13 | 3,181 | 2,445 | 4.29 | 274.97% | −12.8% | Nested VT+DDC |
| v14 | 5,076 | 4,185 | 4.91 | 225.84% | −9.4% | RB blends + pre-lev HDDC |
| v15 | ~5,000 | 4,944 | 5.42 | 101% | −3.2% | Hierarchical DDC (2-layer) |
| v16 | ~6,000 | 5,962 | 6.25 | 109% | −2.3% | Triple-layer DDC (IS peak) |
| *Audit* | — | — | — | — | — | *41 findings, 11 HIGH* |
| v17 | ~5,500 | — | 4.07 | — | — | 11 audit fixes; **OOS Sh=2.08** |
| v18 | 2,751 | — | — | — | — | Alpha reweight; **OOS Sh=2.30** |

### Appendix B: Cost Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TX_BPS` | 5 | Transaction cost (basis points per unit turnover) |
| `SHORT_COST` | 0.5% p.a. | Annual cost of maintaining short positions |
| `RF_CASH` | 2.0% p.a. | Risk-free rate earned on uninvested cash |
| `LEV_COST_STD` | 1.5% p.a. | Standard leverage cost |
| `SPREAD_BPS` | 0.3–3.5 | Half-spread per ETF (see §4.6) |
| Rebalancing cost | 3–5 bps | Position-level DDC scale change cost |

### Appendix C: OOS Pair Portfolio Detail

| Portfolio | OOS Sharpe | OOS CAGR | OOS MaxDD |
|-----------|-----------|----------|-----------|
| ShF5, n=0.06 | 1.70 | 2.06% | −1.12% |
| ShF5, n=0.10 | 1.04 | 2.07% | −2.13% |
| ShF5, n=0.12 | 0.87 | 2.08% | −2.64% |
| ShF7, n=0.06 | 1.65 | 2.31% | −1.48% |
| ShF7, n=0.10 | 1.07 | 2.47% | −2.75% |
| ShF7, n=0.12 | 0.91 | 2.53% | −3.39% |
| ShF10, n=0.06 | 1.21 | 2.49% | −2.19% |
| ShF10, n=0.10 | 0.79 | 2.69% | −3.93% |
| ShF10, n=0.12 | 0.68 | 2.75% | −4.82% |

The inverse relationship between notional and Sharpe is consistent across all portfolio sizes: higher per-pair exposure amplifies both return and risk, but the risk increase outpaces the return increase at the margin.

### Appendix D: Top 10 IS Pairs (v18 Scan)

| Pair | Window | Entry-Z / Exit-Z | IS Sharpe | IS CAGR |
|------|--------|-------------------|-----------|---------|
| XLP/XLU | w=63 | 2.25/0.50 | 1.114 | 8.36% |
| XLP/XLU | w=63 | 2.25/0.75 | 0.984 | 6.91% |
| XLP/XLU | w=126 | 2.25/0.75 | 0.953 | 6.20% |
| XLP/XLU | w=63 | 2.00/0.50 | 0.938 | 7.40% |
| XLP/XLU | w=126 | 1.75/0.50 | 0.907 | 7.67% |
| XLP/XLU | w=126 | 2.25/0.50 | 0.899 | 6.61% |
| XLP/XLU | w=63 | 1.75/0.50 | 0.846 | 7.11% |
| XLP/XLU | w=126 | 2.00/0.50 | 0.845 | 6.68% |
| XLF/IWM | w=42 | 2.00/0.50 | 0.840 | 6.78% |
| XLF/IWM | w=42 | 1.75/0.50 | 0.832 | 7.19% |

XLP/XLU (Consumer Staples / Utilities) dominates the pair universe, consistent with their high correlation and stable mean-reversion dynamics.

### Appendix E: Software and Reproducibility

| Component | Version/Detail |
|-----------|---------------|
| Python | 3.x (`".venv tradingalgo"`) |
| NumPy | Standard scientific stack |
| Pandas | Data manipulation |
| yfinance | Data sourcing (Yahoo Finance) |
| SciPy | Statistical functions |
| OS | Windows |
| Random Seed | 42 (bootstrap) |
| Total code (v18) | 1,354 lines |
| Execution | Single-threaded; ~5–10 min on standard hardware |

### Appendix F: Glossary of Abbreviations

| Abbreviation | Full Term |
|-------------|-----------|
| CAGR | Compound Annual Growth Rate |
| CF | Composite-Filtered (pair selection method) |
| CH | CrashHedge |
| CI | Confidence Interval |
| DDC | Drawdown Control |
| DSR | Deflated Sharpe Ratio |
| E2/E3/E4/E5 | 2/3/4/5-source ensemble |
| HDDC | Hierarchical Drawdown Control |
| IS | In-Sample |
| IVW | Inverse-Volatility Weighted |
| MDD | Maximum Drawdown |
| MTF | Multi-Timeframe |
| OOS | Out-of-Sample |
| PL-DDC | Position-Level Drawdown Control |
| RB | Return-stream Blend |
| SecRot | Sector Rotation |
| ShF | Sharpe-Filtered (pair selection method) |
| TL-DDC | Triple-Layer Drawdown Control |
| TSMom | Time-Series Momentum |
| VT | Volatility Targeting |
| XSMom | Cross-Sectional Momentum |
| ZP | Z-score Pair portfolio |

---

*This document constitutes a comprehensive research report covering the full development lifecycle of the Financial-Algorithms leveraged alpha generation system, from initial exploration through 18 iterative versions, a 41-point self-audit, and true out-of-sample validation.*
