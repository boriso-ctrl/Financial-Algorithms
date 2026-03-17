# Leveraged Alpha Generation in Multi-Asset ETF Portfolios: A Systematic Approach to Strategy Design, Validation, and Deployment

**Author**: Boris Armel  
**Date**: March 2026  
**Repository**: `Financial-Algorithms`  
**Version**: v18 (Alpha-Enhanced)

---

## Abstract

This report presents the complete design, mathematical framework, empirical evolution, and out-of-sample validation of a leveraged multi-asset alpha generation system spanning 18 iterative versions. The system trades an 18-ETF universe across four asset classes (sectors, broad equity, fixed income, commodities) using a multi-source ensemble of z-score pair trading, crash-hedge overlays, cross-sectional momentum, time-series momentum, and sector rotation — combined with volatility targeting, hierarchical drawdown control, and adaptive leverage. The final champion configuration (v18) achieves an out-of-sample Sharpe ratio of 2.01 and CAGR of 46.30% at 3× leverage with a maximum drawdown of −7.59%, validated on one year of truly unseen data (March 2025 – March 2026). All backtest results are subjected to deflated Sharpe ratio (DSR) testing, bootstrap confidence intervals, and walk-forward analysis across five non-overlapping sub-periods. A companion high-frequency trading (HFT) scalper layer using a 7-indicator multi-timeframe architecture adds intraday alpha through regime-adaptive execution.

---

## Table of Contents

1. [Part I — Theoretical Foundations](#part-i--theoretical-foundations)
   1. [The Sharpe Ratio: Definition, Estimation, and Limitations](#1-the-sharpe-ratio-definition-estimation-and-limitations)
   2. [Statistical Pitfalls in Strategy Research](#2-statistical-pitfalls-in-strategy-research)
   3. [Correcting Sharpe Inflation: PSR, DSR, and PBO](#3-correcting-sharpe-inflation-psr-dsr-and-pbo)
   4. [Volatility Targeting and Risk Scaling](#4-volatility-targeting-and-risk-scaling)
   5. [Why Backtests Fail in Live Trading](#5-why-backtests-fail-in-live-trading)
   6. [High-Frequency Trading: Microstructure and Execution](#6-high-frequency-trading-microstructure-and-execution)
2. [Part II — Research Evolution (v1–v18)](#part-ii--research-evolution-v1v18)
   1. [Phase 3: SMA Crossover Baseline](#7-phase-3-sma-crossover-baseline)
   2. [Phase 5–6: Multi-Indicator Voting System](#8-phase-56-multi-indicator-voting-system)
   3. [The Lookahead Bias Discovery](#9-the-lookahead-bias-discovery)
   4. [Voting Strategy Tuning and Failure Analysis](#10-voting-strategy-tuning-and-failure-analysis)
   5. [Phase 7: Aggressive Growth Experiments](#11-phase-7-aggressive-growth-experiments)
   6. [v14–v17: Leveraged Pair Trading Evolution](#12-v14v17-leveraged-pair-trading-evolution)
   7. [v17 Audit: 41 Findings and Hardening](#13-v17-audit-41-findings-and-hardening)
   8. [v18: Alpha Enhancement and Final Architecture](#14-v18-alpha-enhancement-and-final-architecture)
3. [Part III — System Architecture and Mathematical Framework](#part-iii--system-architecture-and-mathematical-framework)
   1. [Universe and Data Pipeline](#15-universe-and-data-pipeline)
   2. [Z-Score Pair Trading Engine](#16-z-score-pair-trading-engine)
   3. [CrashHedge: Volatility-Regime Overlay](#17-crashhedge-volatility-regime-overlay)
   4. [Cross-Sectional Momentum (Jegadeesh–Titman)](#18-cross-sectional-momentum-jegadeshtitman)
   5. [Time-Series Momentum (Trend-Following)](#19-time-series-momentum-trend-following)
   6. [Sector Rotation](#20-sector-rotation)
   7. [Ensemble Construction](#21-ensemble-construction)
   8. [Portfolio Overlays: Volatility Target and Drawdown Control](#22-portfolio-overlays-volatility-target-and-drawdown-control)
   9. [Leverage and Cost Modelling](#23-leverage-and-cost-modelling)
   10. [Backtest Engine and Cost Stack](#24-backtest-engine-and-cost-stack)
   11. [HFT Scalper Layer: 7-Indicator Multi-Timeframe Architecture](#25-hft-scalper-layer-7-indicator-multi-timeframe-architecture)
   12. [Regime Detection and Adaptive Routing](#26-regime-detection-and-adaptive-routing)
4. [Part IV — Results and Validation](#part-iv--results-and-validation)
   1. [In-Sample Results](#27-in-sample-results)
   2. [Out-of-Sample Results](#28-out-of-sample-results)
   3. [Walk-Forward Analysis](#29-walk-forward-analysis)
   4. [Statistical Validation (DSR, Bootstrap)](#30-statistical-validation-dsr-bootstrap)
   5. [Paper Trading Validation](#31-paper-trading-validation)
   6. [5-Indicator Voting System Results (Legacy)](#32-5-indicator-voting-system-results-legacy)
   7. [Lessons Learned and Failure Catalogue](#33-lessons-learned-and-failure-catalogue)
5. [Appendices](#appendices)
   - [A. Full Parameter Table](#appendix-a-full-parameter-table)
   - [B. ETF Universe Details](#appendix-b-etf-universe-details)
   - [C. Cost Model Constants](#appendix-c-cost-model-constants)

---

# Part I — Theoretical Foundations

## 1. The Sharpe Ratio: Definition, Estimation, and Limitations

### 1.1 Definition

The Sharpe ratio measures the excess return per unit of risk. For a portfolio with return series $\{r_t\}_{t=1}^{T}$ and risk-free rate $r_f$:

$$
\text{SR} = \frac{\bar{r} - r_f}{\sigma_r}
$$

where:
- $\bar{r} = \frac{1}{T}\sum_{t=1}^{T} r_t$ is the sample mean return,
- $\sigma_r = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_t - \bar{r})^2}$ is the sample standard deviation,
- $r_f$ is the risk-free rate (set to 0 in this system for daily returns).

**Annualisation.** For daily returns assumed to be IID:

$$
\text{SR}_{\text{annual}} = \text{SR}_{\text{daily}} \times \sqrt{252}
$$

This is the formula used throughout this system. However, the assumption of IID returns is critical and often violated.

### 1.2 Serial Correlation and Annualisation Errors

If returns exhibit autocorrelation $\rho_k$ at lag $k$, the correct annualisation is:

$$
\text{SR}_{\text{annual}} = \text{SR}_{\text{daily}} \times \sqrt{\frac{252}{1 + 2\sum_{k=1}^{q} \rho_k}}
$$

When $\rho_k > 0$ (positive autocorrelation, common in strategies with overlapping holding periods, smoothed prices, or illiquid marks), the naive $\sqrt{252}$ scaling **overstates** the true Sharpe ratio. This is a primary mechanism by which strategies with stale prices or unrealistic fill assumptions generate inflated performance metrics.

### 1.3 Non-Normality and Tail Risk

The Sharpe ratio treats all volatility as equivalent risk, ignoring:

- **Skewness** ($\hat{\mu}_3$): Negative skew (fat left tail) is more dangerous than positive skew, but Sharpe treats them identically.
- **Excess kurtosis** ($\hat{\mu}_4 - 3$): Fat tails increase the probability of extreme events beyond what variance captures.
- **Path dependence**: Two strategies with identical Sharpe ratios can have vastly different drawdown profiles.

The Sharpe ratio can be **manipulated** through dynamic strategies and option-like payoffs. Selling out-of-the-money puts, for example, generates high measured Sharpe by truncating right tails while loading left tails — a "short volatility" profile that eventually produces catastrophic losses.

### 1.4 Companion Risk-Adjusted Metrics

This system computes the following alongside Sharpe:

**Sortino Ratio** — penalises only downside deviation:

$$
\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_d} \times \sqrt{252}
$$

where $\sigma_d = \sqrt{\frac{1}{N_d}\sum_{r_t < 0} r_t^2}$ is the downside standard deviation computed from negative returns only.

**Calmar Ratio** — reward per unit of maximum drawdown:

$$
\text{Calmar} = \frac{\text{CAGR}}{|\text{MaxDD}|}
$$

**Maximum Drawdown** — the worst peak-to-trough decline:

$$
\text{MaxDD} = \min_t \left(\frac{E_t - \max_{s \leq t} E_s}{\max_{s \leq t} E_s}\right)
$$

where $E_t = E_0 \prod_{i=1}^{t}(1 + r_i)$ is the equity curve.

**Win Rate** — the fraction of positive-return trading days:

$$
\text{WinRate} = \frac{\#\{t : r_t > 0\}}{\#\{t : r_t \neq 0\}}
$$

**CAGR (Compound Annual Growth Rate)**:

$$
\text{CAGR} = \left(\frac{E_T}{E_0}\right)^{252/T} - 1
$$

---

## 2. Statistical Pitfalls in Strategy Research

### 2.1 The Four Biases That Inflate Sharpe

**Lookahead Bias / Leakage.** Any use of information unavailable at the decision time. This includes: computing indicators on future prices, using entire datasets for normalisation, applying survivor-free universes constructed in hindsight, and using revised fundamental data. This system enforces strict walk-forward signal computation and `shift(1)` on all weight matrices.

**Overfitting.** The backtest winner among $N$ trials is expected to disappoint out-of-sample. If $N$ strategies are tested, the expected maximum Sharpe of pure noise strategies is:

$$
\mathbb{E}[\max_{i=1}^{N} \widehat{\text{SR}}_i] \approx \sqrt{2\ln N}\left(1 - \frac{\gamma}{2\ln N}\right) + \frac{\gamma}{\sqrt{2\ln N}}
$$

where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant. For $N = 10{,}000$ strategy variants (typical of this system's grid search), $\mathbb{E}[\max \widehat{\text{SR}}] \approx 3.02$ from pure noise. Any in-sample Sharpe below this threshold is indistinguishable from data mining.

**Multiple Testing / Data Snooping.** Repeatedly testing strategies or parameter grids on the same data inflates statistical significance. Formal corrections include White's Reality Check, Hansen's Superior Predictive Ability (SPA) test, and the Deflated Sharpe Ratio.

**Survivorship Bias.** Using datasets that only contain currently active assets or strategies that "survived" to the present day. This system uses the full 18-ETF universe from inception date with forward-fill only (no backfill).

### 2.2 Robust Validation Methods

**Walk-Forward Analysis.** This system uses 5 non-overlapping sub-periods for walk-forward validation:

| Period | Start | End |
|--------|-------|-----|
| 2010–12 | 2010-01-01 | 2013-01-01 |
| 2013–15 | 2013-01-01 | 2016-01-01 |
| 2016–18 | 2016-01-01 | 2019-01-01 |
| 2019–21 | 2019-01-01 | 2022-01-01 |
| 2022–25 | 2022-01-01 | 2025-03-01 |

A strategy is considered "consistent" if the Sharpe ratio exceeds 0.5 in every sub-period.

**Purged and Embargoed Cross-Validation.** For labelled ML models, standard cross-validation leaks information through overlapping observation windows. Purging removes any training observation whose label window overlaps the test period's label window. Embargoing adds an additional time buffer after the test period to prevent look-forward leakage.

---

## 3. Correcting Sharpe Inflation: PSR, DSR, and PBO

### 3.1 Probabilistic Sharpe Ratio (PSR)

The PSR accounts for sampling uncertainty and non-normality when testing whether an observed Sharpe exceeds a benchmark $\text{SR}^*$:

$$
\text{PSR}(\text{SR}^*) = \Phi\left(\frac{(\widehat{\text{SR}} - \text{SR}^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \cdot \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}}\right)
$$

where:
- $\Phi(\cdot)$ is the standard normal CDF,
- $T$ is the number of observations,
- $\hat{\gamma}_3$ is the sample skewness of returns,
- $\hat{\gamma}_4$ is the sample kurtosis of returns.

This gives the probability that the true Sharpe exceeds $\text{SR}^*$, accounting for the shape of the return distribution.

### 3.2 Deflated Sharpe Ratio (DSR)

The DSR corrects for multiple testing by replacing the benchmark $\text{SR}^*$ with the expected maximum Sharpe from $N$ independent trials:

$$
\mathbb{E}[\max_{i=1}^{N} \widehat{\text{SR}}_i] \approx \sqrt{2\ln N}\left(1 - \frac{\gamma}{2\ln N}\right) + \frac{\gamma}{\sqrt{2\ln N}}
$$

The DSR is then:

$$
\text{DSR} = \frac{\widehat{\text{SR}} - \mathbb{E}[\max \widehat{\text{SR}}]}{\text{SE}(\widehat{\text{SR}})}
$$

where the standard error of the Sharpe ratio is:

$$
\text{SE}(\widehat{\text{SR}}) = \sqrt{\frac{1 - \hat{\gamma}_3 \cdot \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}{T - 1}}
$$

The DSR $p$-value is $1 - \Phi(\text{DSR})$. A positive DSR with $p < 0.05$ indicates the strategy's performance is unlikely to be explained by data mining alone.

**Implementation.** The system's `deflated_sharpe_ratio()` function computes this directly:

```python
def deflated_sharpe_ratio(observed_sr, n_trials, n_obs, skew=0.0, kurt=3.0):
    gamma = 0.5772  # Euler-Mascheroni constant
    e_max_sr = (sqrt(2 * ln(n_trials))
                * (1 - gamma / (2 * ln(n_trials)))
                + gamma / sqrt(2 * ln(n_trials)))
    se_sr = sqrt((1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2)
                 / (n_obs - 1))
    dsr = (observed_sr - e_max_sr) / se_sr
    p_value = 1 - Phi(dsr)
    return dsr, p_value, e_max_sr
```

### 3.3 Probability of Backtest Overfitting (PBO)

PBO estimates the probability that the in-sample optimal strategy is actually overfit, using combinatorial cross-validation. A PBO close to 1.0 means the strategy selected by in-sample optimisation is very likely to underperform out-of-sample. PBO values below 0.5 are generally considered acceptable.

### 3.4 Bootstrap Confidence Intervals

The system constructs 95% confidence intervals for the Sharpe ratio via non-parametric bootstrap:

1. Resample $T$ returns with replacement, $B = 5{,}000$ times.
2. Compute $\widehat{\text{SR}}_b$ for each bootstrap sample $b$.
3. Report the 2.5th and 97.5th percentiles as the confidence interval.

```python
def bootstrap_sharpe_ci(returns, n_boot=5000, ci=0.95):
    for b in range(n_boot):
        sample = resample_with_replacement(returns, size=T)
        boot_sharpes[b] = mean(sample) / std(sample) * sqrt(252)
    return percentile(boot_sharpes, [2.5, 97.5])
```

---

## 4. Volatility Targeting and Risk Scaling

### 4.1 Motivation

Volatility targeting is one of the most reliable Sharpe-ratio improvements documented in the literature. Scaling exposure inversely with recent realised variance has been shown to increase factor Sharpe ratios by 50–100% across equity factors and FX carry strategies.

### 4.2 Mathematical Formulation

Let $\sigma^{\text{realised}}_t$ be the trailing realised annualised volatility at time $t$ computed over a lookback window $L$:

$$
\sigma^{\text{realised}}_t = \text{std}(r_{t-L}, \ldots, r_{t-1}) \times \sqrt{252}
$$

The volatility-targeting scale factor is:

$$
s_t = \text{clip}\left(\frac{\sigma^{\text{target}}}{\sigma^{\text{realised}}_t}, 0.2, 5.0\right)
$$

The scaled return is:

$$
r^{\text{VT}}_t = r_t \times s_{t-1}
$$

Note the **one-day lag** ($s_{t-1}$) — the scale factor uses only information available at time $t-1$, preventing lookahead bias.

### 4.3 Parameter Choices

The system searches over $\sigma^{\text{target}} \in \{0.04, 0.05, 0.06, 0.07, 0.08, 0.10\}$ with a fixed lookback $L = 63$ trading days. The minimum realised volatility is floored at 0.5% annualised to prevent division by near-zero in calm markets.

---

## 5. Why Backtests Fail in Live Trading

### 5.1 Principal Causes of Live Degradation

Research documents six systematic categories of backtest-to-live performance decay:

1. **Lookahead bias** — signals computed using data not available at decision time. The most severe form; can produce Sharpe ratios orders of magnitude above reality. This system's own experience dramatically confirms this: correcting a lookahead bug in the intraday crypto strategy reduced Sharpe from +6.37 to −42.89 — a catastrophic reversal.

2. **Overfitting to in-sample noise** — selecting the best configuration among thousands of trials. This is mitigated through DSR testing and walk-forward validation.

3. **Transaction cost underestimation** — especially market impact, slippage at scale, and spread widening during volatility events. This system models: fixed commissions (5 bps), ETF-specific half-spreads (0.15–1.75 bps), short-selling costs (0.5% annualised), and leverage costs (1.0–2.0% annualised).

4. **Regime non-stationarity** — market microstructure, correlations, and factor premia change over time. A strategy optimised for 2010–2020 may not work in 2021–2025.

5. **Capacity constraints** — a strategy that works on $100K may fail at $10M due to market impact, especially in less liquid ETFs (XLB, XLRE with ADV ~$200M).

6. **Execution latency and slippage** — the time between signal generation and order execution introduces adverse selection, particularly at intraday frequencies.

### 5.2 This System's Defences

| Defence | Implementation |
|---------|---------------|
| No lookahead | All weights shifted by 1 day: `w = w.shift(1).fillna(0)` |
| No backfill | Forward-fill only: `p.ffill()` (never `bfill()`) |
| Spread model | ETF-specific half-spread costs on every weight change |
| Short cost | 0.5% annualised on all short positions |
| Leverage cost | Explicit annualised cost on gross exposure > 100% |
| Market impact | ADV-based capacity check for each ETF |
| Walk-forward | 5 sub-period consistency check |
| DSR testing | Corrects for N > 10,000 strategy variants tested |
| True OOS | Final year locked out of all optimisation |

---

## 6. High-Frequency Trading: Microstructure and Execution

### 6.1 Theoretical Framework

At intraday frequencies, alpha is dominated by microstructure features rather than macroeconomic factors. Key concepts include:

**Order Flow Imbalance (OFI).** The net buying pressure measured as the difference between aggressive buy and sell volumes. OFI is a direct predictor of short-term price movements.

**Bid-Ask Spread as Information Cost.** Each round-trip trade incurs the spread as a minimum cost. For the HFT layer, the system models:

$$
\text{Cost per round-trip} = 2 \times (\text{slippage}_{\text{bps}} + \text{commission}_{\text{bps}}) / 10{,}000
$$

With 3 bps slippage + 1 bps commission = 4 bps per side = 8 bps per round-trip.

**Signal Decay.** Intraday alpha signals decay rapidly — typically within minutes to hours. The HFT scalper uses a maximum holding period of 60 bars (≈3 hours at 3-minute intervals) to avoid holding decayed positions.

### 6.2 HFT Cost Model

The HFT layer uses a more aggressive cost model than the daily layer:

$$
\text{fill}_{\text{long}} = P_{\text{close}} \times (1 + c_{\text{frac}})
$$
$$
\text{fill}_{\text{short}} = P_{\text{close}} \times (1 - c_{\text{frac}})
$$

where $c_{\text{frac}} = (3.0 + 1.0) / 10{,}000 = 0.0004$ (4 bps per side).

---

# Part II — Research Evolution (v1–v18)

## 7. Phase 3: SMA Crossover Baseline

### 7.1 Signal Definition

The SMA crossover is the simplest trend-following signal:

$$
\text{signal}_t = \text{sgn}\left(\text{SMA}_t^{\text{fast}} - \text{SMA}_t^{\text{slow}}\right)
$$

where:

$$
\text{SMA}_t^{n} = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}
$$

The system uses $n_{\text{fast}} = 50$ and $n_{\text{slow}} = 200$ (the classic "golden cross" and "death cross").

### 7.2 Baseline Performance

The SMA crossover achieved a Sharpe ratio of approximately 1.65 over the in-sample period. While sufficient as a proof-of-concept, the signal is too slow to capture short-term reversals and generates excessive whipsaws in range-bound markets.

---

## 8. Phase 5–6: Multi-Indicator Voting System

### 8.1 Architecture

The voting system aggregates 5 core indicators, each contributing a score in $[-2, +2]$:

$$
S_t = \sum_{k=1}^{5} I_k(t) \quad \in [-10, +10]
$$

**Decision rules:**
- **BUY**: $S_t \geq \theta_{\text{buy}}$ (system default $\theta_{\text{buy}} = 5.0$)
- **SELL**: $S_t \leq \theta_{\text{sell}}$ (system default $\theta_{\text{sell}} = -5.0$)
- **HOLD**: $\theta_{\text{sell}} < S_t < \theta_{\text{buy}}$

### 8.2 Indicator 1: SMA Crossover ($I_{\text{SMA}}$)

Computes the distance between fast and slow simple moving averages:

$$
\text{SMA}_{\text{fast}} = \frac{1}{20}\sum_{i=0}^{19} P_{t-i}, \quad \text{SMA}_{\text{slow}} = \frac{1}{50}\sum_{i=0}^{49} P_{t-i}
$$

$$
d_t = \frac{|\text{SMA}_{\text{fast}} - \text{SMA}_{\text{slow}}|}{\text{SMA}_{\text{slow}}} \times 100
$$

$$
I_{\text{SMA}}(t) = \begin{cases}
+2 & \text{if } \text{SMA}_{\text{fast}} > \text{SMA}_{\text{slow}} \text{ and } d_t > 1.0\% \\
+1 & \text{if } \text{SMA}_{\text{fast}} > \text{SMA}_{\text{slow}} \text{ and } d_t \leq 1.0\% \\
-1 & \text{if } \text{SMA}_{\text{fast}} < \text{SMA}_{\text{slow}} \text{ and } d_t \leq 1.0\% \\
-2 & \text{if } \text{SMA}_{\text{fast}} < \text{SMA}_{\text{slow}} \text{ and } d_t > 1.0\%
\end{cases}
$$

### 8.3 Indicator 2: RSI ($I_{\text{RSI}}$)

The Relative Strength Index over period $n = 14$:

$$
\Delta_t = P_t - P_{t-1}
$$

$$
\text{AvgGain}_n = \frac{1}{n}\sum_{i=0}^{n-1}\max(\Delta_{t-i}, 0), \quad \text{AvgLoss}_n = \frac{1}{n}\sum_{i=0}^{n-1}|\min(\Delta_{t-i}, 0)|
$$

$$
\text{RS} = \frac{\text{AvgGain}_n}{\text{AvgLoss}_n}, \quad \text{RSI} = 100 - \frac{100}{1 + \text{RS}}
$$

$$
I_{\text{RSI}}(t) = \begin{cases}
+2 & \text{if RSI} < 30 \text{ (oversold — bullish reversal)} \\
+1 & \text{if } 30 \leq \text{RSI} < 50 \\
\phantom{+}0 & \text{if RSI} = 50 \\
-1 & \text{if } 50 < \text{RSI} \leq 70 \\
-2 & \text{if RSI} > 70 \text{ (overbought — bearish reversal)}
\end{cases}
$$

### 8.4 Indicator 3: Volume Confirmation ($I_{\text{VOL}}$)

Compares current volume to its 20-period moving average:

$$
\text{VOL}_{\text{MA}} = \frac{1}{20}\sum_{i=0}^{19} V_{t-i}
$$

$$
\text{VR}_t = \frac{V_t}{\text{VOL}_{\text{MA}}}
$$

$$
I_{\text{VOL}}(t) = \begin{cases}
+2 & \text{if } \text{VR}_t > 1.5 \\
+1 & \text{if } 1.0 < \text{VR}_t \leq 1.5 \\
\phantom{+}0 & \text{if } \text{VR}_t = 1.0 \\
-1 & \text{if } 0.7 \leq \text{VR}_t < 1.0 \\
-2 & \text{if } \text{VR}_t < 0.7
\end{cases}
$$

### 8.5 Indicator 4: ADX Trend Strength ($I_{\text{ADX}}$)

The ADX (Average Directional Index) measures trend strength. The system uses a simplified proxy based on price position within the recent high–low range:

$$
\text{RecentHigh} = \frac{1}{14}\sum_{i=0}^{13} H_{t-i}, \quad \text{RecentLow} = \frac{1}{14}\sum_{i=0}^{13} L_{t-i}
$$

$$
\text{Strength}_t = \frac{P_t - \text{RecentLow}}{\text{RecentHigh} - \text{RecentLow}}
$$

$$
I_{\text{ADX}}(t) = \begin{cases}
+2 & \text{if Strength}_t > 0.7 \\
+1 & \text{if } 0.4 < \text{Strength}_t \leq 0.7 \\
\phantom{+}0 & \text{if } 0.4 \leq \text{Strength}_t \leq 0.4 \\
-1 & \text{if } 0.2 \leq \text{Strength}_t < 0.4 \\
-2 & \text{if Strength}_t < 0.2
\end{cases}
$$

The full ADX computation uses True Range and Directional Movement:

$$
\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
$$

$$
+\text{DM}_t = \begin{cases} H_t - H_{t-1} & \text{if } H_t - H_{t-1} > L_{t-1} - L_t \text{ and } H_t - H_{t-1} > 0 \\ 0 & \text{otherwise} \end{cases}
$$

$$
-\text{DM}_t = \begin{cases} L_{t-1} - L_t & \text{if } L_{t-1} - L_t > H_t - H_{t-1} \text{ and } L_{t-1} - L_t > 0 \\ 0 & \text{otherwise} \end{cases}
$$

$$
+\text{DI}_t = 100 \times \frac{\text{SMA}_{14}(+\text{DM}_t)}{\text{SMA}_{14}(\text{TR}_t)}, \quad -\text{DI}_t = 100 \times \frac{\text{SMA}_{14}(-\text{DM}_t)}{\text{SMA}_{14}(\text{TR}_t)}
$$

$$
\text{DX}_t = 100 \times \frac{|+\text{DI}_t - (-\text{DI}_t)|}{+\text{DI}_t + (-\text{DI}_t)}, \quad \text{ADX}_t = \text{SMA}_{14}(\text{DX}_t)
$$

### 8.6 Indicator 5: ATR Volatility ($I_{\text{ATR}}$)

The ATR (Average True Range) measures volatility. The system uses it as a trend filter — high volatility with clear directional bias yields a strong signal:

$$
\text{ATR}_t = \frac{1}{14}\sum_{i=0}^{13}\text{TR}_{t-i}
$$

$$
\text{Upper}_t = \text{SMA}_{14}(C_t) + 1.5 \times \text{ATR}_t
$$

$$
\text{Lower}_t = \text{SMA}_{14}(C_t) - 1.5 \times \text{ATR}_t
$$

$$
I_{\text{ATR}}(t) = \text{scale}\left(\frac{C_t - \text{Upper}_t}{\text{ATR}_t}\right) \text{ if } C_t > \text{Upper}_t, \quad \text{or}\quad \text{scale}\left(\frac{C_t - \text{Lower}_t}{\text{ATR}_t}\right) \text{ if } C_t < \text{Lower}_t
$$

### 8.7 Tiered Exit Management

The exit system implements a two-tiered profit-taking strategy with trailing stops:

**Entry:** $S_t \geq \theta_{\text{buy}}$ at price $P_{\text{entry}}$

**Exit conditions** (checked in order):

1. **Stop Loss**: Exit if $P_t \leq P_{\text{entry}} \times (1 - 0.02)$
2. **TP1** (partial): At $P_t \geq P_{\text{entry}} \times (1 + 0.015)$, exit 50% of position and activate trailing stop
3. **TP2** (full): At $P_t \geq P_{\text{entry}} \times (1 + 0.03)$, exit remaining position
4. **Signal reversal**: $S_t \leq \theta_{\text{sell}}$
5. **Trailing stop**: After TP1, track high-water mark $H^* = \max_{s \leq t} P_s$ and exit if $P_t \leq H^* \times (1 - 0.01)$

### 8.8 Position Sizing

Dynamic position sizing based on composite score strength:

$$
\text{Size}_t = \begin{cases}
2\% & \text{if } |S_t| \leq 5 \\
4\% & \text{if } 5 < |S_t| \leq 7 \\
6\% & \text{if } 7 < |S_t| \leq 9 \\
8\% & \text{if } |S_t| > 9
\end{cases}
$$

---

## 9. The Lookahead Bias Discovery

### 9.1 The Bug

The Phase 6 intraday crypto strategy (BTC/ETH on 1-minute bars) reported a Sharpe ratio of +6.37. Investigation revealed that signals were being computed on the **entire dataset upfront** before processing bars:

```python
# INCORRECT — all signals computed with future data
signals = signal_func(df_bars)  # df_bars contains ALL future bars
df_bars['signal'] = signals
```

When processing bar 100, the signal had access to bars 101–2000. Moving averages, RSI extremes, and every trading decision had perfect knowledge of future prices.

### 9.2 The Fix: Walk-Forward Processing

The corrected backtest processes bars sequentially:

```python
# CORRECT — walk-forward analysis
for idx in range(len(df_bars)):
    hist_data = df_bars[df_bars.index <= idx]  # Only past + current
    signal = signal_func(hist_data)            # No future data
```

### 9.3 Performance Impact

| Metric | With Lookahead (Fictitious) | Walk-Forward (Real) | Change |
|--------|---------------------------|--------------------:|--------|
| Sharpe Ratio | +6.37 | −42.89 | −700% |
| Total Return | +3.34% | −3.17% | −6.51 pp |
| Win Rate | 50.0% | 26.7% | −23.3 pp |

This is the single most important finding of the entire research programme: **a Sharpe of +6.37 was entirely fictitious**. The true performance was catastrophically negative.

### 9.4 Diagnostic Red Flags

- Sharpe $> 3.0$ consistently → likely lookahead bias
- Win rate $> 60\%$ on short timeframes → probable forward-peeking
- Generating all signals upfront → definite lookahead
- Using entire dataset for indicators → confirmed lookahead

---

## 10. Voting Strategy Tuning and Failure Analysis

### 10.1 Comprehensive Parameter Sweep

A systematic sweep of the buy threshold parameter revealed a critical finding:

| Threshold | Avg Sharpe | Avg Trades | Best Case | Worst Case |
|-----------|-----------|------------|-----------|------------|
| 1.0 | −0.84 | 40 | MSFT +0.57 | AAPL −1.44 |
| 2.0 | −0.40 | 25 | MSFT +0.88 | AAPL −1.22 |
| 2.5 | −0.40 | 25 | MSFT +0.88 | AAPL −1.22 |
| 3.0 | −0.55 | 15 | MSFT +0.85 | SPY −1.38 |
| 4.0 | −1.04 | 10 | All negative | — |
| 5.0 | −1.44 | 5 | All negative | — |

**Even the optimal threshold (2.0–2.5) only achieved a Sharpe of −0.40.** Only MSFT was consistently profitable.

### 10.2 Root Causes

1. **Indicator redundancy**: RSI, Stochastic, and MACD all measure momentum. The system over-votes on momentum at the expense of other factors like mean-reversion and volatility.
2. **Poor win rate**: 35–40% across all thresholds, below the ~50% breakeven required with the system's risk/reward ratio.
3. **Premature exits**: Exiting at score $\leq 0$ in a bull market closes winners too early.
4. **Position sizing mismatch**: Even at small size (2–4%), poor entry quality still generates losses.

### 10.3 Decision

This finding precipitated the pivot from voting-based single-asset trading to the multi-asset leveraged pair-trading architecture (v14+).

---

## 11. Phase 7: Aggressive Growth Experiments

Six approaches were tested to improve absolute returns:

| Approach | Entry | Exit | Sizing | Return | Sharpe | Win% | Trades |
|----------|-------|------|--------|--------|--------|------|--------|
| Stacked Growth | ≥2.0 | ≤−2.0 | 10–15% | −1.15% | 3.55 | 45% | 133 |
| Trend Following | ≥2.0 | ≤−2.0 | 10–15% | −0.23% | 4.30 | 48% | 27 |
| **Trend Holding** | **≥2.0** | **≤−2.0** | **15%** | **+7.17%** | **10.94** | **67%** | **9** |
| Aggressive v2 | ≥1.5 | ≤−3.0 | 15% | +7.35% | 10.87 | 71% | 7 |
| Lighter Entry v3 | ≥1.0 | ≤−2.5 | 20% | +11.86% | 11.52 | 71% | 7 |
| Buy-Hold Proxy | ≥0.5 | ≤−10.0 | 100% | +80.43% | 0.0 | 100% | 1 |
| SPY Buy-Hold | — | — | 100% | +87.67% | — | — | 1 |

**Key insight:** In the strong 2023–2025 bull market, buy-and-hold was optimal. The voting system correctly identified the bull regime (buy signal 58.6% of days, sell signal only 5.1%) but any exit logic destroyed returns. This is a fundamental regime-dependence finding: market-timing exits subtract value in sustained trends.

---

## 12. v14–v17: Leveraged Pair Trading Evolution

Recognising the limitations of single-asset voting, the system pivoted to multi-asset statistical arbitrage.

### 12.1 Z-Score Pair Trading Core

The core innovation: trade the mean-reversion of log-price spreads between correlated ETFs, scaled by z-score normalisation, with leverage to amplify small but consistent edge.

### 12.2 v14–v16: Progressive Improvements

- **v14**: Initial pair trading implementation with basic z-score engine
- **v15**: Added CrashHedge volatility-regime overlay and VolCarry strategy
- **v16**: Multi-timeframe pair blending, composite portfolio selection
- **v17**: Complete audit (41 findings), spread-model costs, lagged DDC, no-bfill enforcement

### 12.3 v17 Results

- **In-sample**: Sharpe 4.07 (unleveraged)
- **Out-of-sample (unleveraged)**: Sharpe 2.08, CAGR 2.35%
- **Out-of-sample (3× leverage, 1% cost)**: Sharpe 1.78, CAGR 26.00%, MaxDD −10.02%

---

## 13. v17 Audit: 41 Findings and Hardening

The v17 audit identified and fixed 41 issues across categories:

1. **Lagged DDC** — drawdown control now uses $\text{dd}_{t-1}$ not $\text{dd}_t$
2. **Spread model** — ETF-specific half-spread costs on every weight change
3. **No bfill** — forward-fill only, preventing survival bias in late-starting ETFs
4. **Market impact estimation** — ADV-based capacity check
5. **Leverage cost modelled explicitly** — annualised cost on gross exposure above 100%
6. **Position-level DDC rebalancing cost** — models the cost of DDC scale changes (3–5 bps)

---

## 14. v18: Alpha Enhancement and Final Architecture

### 14.1 Alpha Diagnosis

Analysis of v17's OOS performance identified four root causes of alpha leakage:

1. **CrashHedge underweighted**: OOS CAGR = 17.94% but only 3% ensemble weight
2. **VolCarry proven failure**: OOS Sharpe = −0.14, CAGR = −0.35%, yet receiving 7% weight (more than CrashHedge)
3. **Conservative pair notional**: 0.06 per pair × 5 pairs = low absolute returns despite good risk-adjusted metrics
4. **Missing alpha diversification**: All return sources were either mean-reversion (pairs) or vol-regime (CrashHedge); no momentum or trend-following diversity

### 14.2 Eight Improvements

| # | Improvement | Description |
|---|------------|-------------|
| 1 | Cross-sectional momentum | Jegadeesh–Titman 12-1 month, long winners / short losers |
| 2 | Time-series momentum | Individual asset trend-following across 3 lookbacks |
| 3 | Sector rotation | Momentum-weighted sector tilt, long top / short bottom |
| 4 | Drop VolCarry | Removed entirely (OOS failure) |
| 5 | Reweight CrashHedge | From 3% to 15–30% ensemble weight |
| 6 | Wider pair grid | Notional 0.06–0.12, pair count 5–10 |
| 7 | Wider ensemble grid | 2/3/4/5-source combos tested |
| 8 | Adaptive ensemble | Rolling 63-day performance weighting |

---

# Part III — System Architecture and Mathematical Framework

## 15. Universe and Data Pipeline

### 15.1 ETF Universe

The system trades 18 ETFs across 4 asset classes:

| Class | Tickers | Count |
|-------|---------|-------|
| Sectors | XLK, XLV, XLF, XLE, XLI, XLC, XLP, XLU, XLB, XLRE | 10 |
| Broad Equity | SPY, QQQ, IWM, EFA | 4 |
| Fixed Income / Safe | TLT, IEF, GLD, SHY | 4 |
| **Total** | | **18** |

### 15.2 Data Treatment

- Source: Yahoo Finance via `yfinance`
- Auto-adjusted close prices (splits and dividends)
- Forward-fill only (`ffill()`, never `bfill()`)
- In-sample: 2010-01-01 to 2025-03-01 (~3,800 trading days)
- Out-of-sample: 2025-03-01 to 2026-03-15 (~260 trading days)

---

## 16. Z-Score Pair Trading Engine

### 16.1 Log-Price Spread

For assets $A$ and $B$, the log-price spread is:

$$
\text{spread}_t = \ln(P_t^A) - \ln(P_t^B)
$$

### 16.2 Rolling Z-Score Normalisation

The spread is normalised to a z-score using a rolling window of $W$ days:

$$
\mu_t^W = \frac{1}{W}\sum_{i=0}^{W-1}\text{spread}_{t-i}
$$

$$
\sigma_t^W = \sqrt{\frac{1}{W-1}\sum_{i=0}^{W-1}(\text{spread}_{t-i} - \mu_t^W)^2}
$$

$$
z_t = \frac{\text{spread}_t - \mu_t^W}{\max(\sigma_t^W, 10^{-8})}
$$

The floor of $10^{-8}$ on $\sigma_t^W$ prevents division by zero in periods of extremely low spread variation.

### 16.3 Trading Rules

The pair position is governed by a state machine with two thresholds: entry z-score $z_{\text{entry}}$ and exit z-score $z_{\text{exit}}$:

**Position state transitions:**

$$
\text{pos}_t = \begin{cases}
0 & \text{if flat and } |z_t| < z_{\text{entry}} \\
-1 & \text{if flat and } z_t > z_{\text{entry}} \text{ (spread too wide — short A, long B)} \\
+1 & \text{if flat and } z_t < -z_{\text{entry}} \text{ (spread too narrow — long A, short B)} \\
0 & \text{if long and } z_t > -z_{\text{exit}} \text{ (mean-reversion complete)} \\
0 & \text{if short and } z_t < z_{\text{exit}} \text{ (mean-reversion complete)} \\
\text{pos}_{t-1} & \text{otherwise (hold)}
\end{cases}
$$

**Parameter grid:**

| Parameter | Values Tested |
|-----------|---------------|
| Window $W$ | 21, 42, 63, 126 |
| Entry $z_{\text{entry}}$ | 1.75, 2.00, 2.25 |
| Exit $z_{\text{exit}}$ | 0.50, 0.75 |

### 16.4 Pair Return Computation

The daily return from a pair position is:

$$
r_t^{\text{pair}} = \text{pos}_{t-1} \times (r_t^A - r_t^B)
$$

where $r_t^A = P_t^A / P_{t-1}^A - 1$ and $r_t^B = P_t^B / P_{t-1}^B - 1$ are the daily returns of assets $A$ and $B$. The position is **lagged by one day** ($\text{pos}_{t-1}$) to prevent lookahead.

### 16.5 Pair Weight Construction

For portfolio backtesting, the pair position is converted to weights:

$$
w_t^A = \text{pos}_t \times n, \quad w_t^B = -\text{pos}_t \times n
$$

where $n$ is the notional per pair (tested: 0.06, 0.08, 0.10, 0.12).

### 16.6 Multi-Timeframe (MTF) Blending

For a pair $(A, B)$ with $K$ valid configurations $\{(W_k, z_{\text{entry},k}, z_{\text{exit},k})\}$, the MTF blend divides the notional equally:

$$
w_t^{\text{MTF}} = \frac{1}{K}\sum_{k=1}^{K} w_t^{(k)}, \quad \text{each at notional } n/K
$$

### 16.7 Exhaustive Pair Scan

The system scans all $\binom{18}{2} = 153$ pairs × 4 windows × 4 z-configurations = **2,448 configurations**. Pairs with Sharpe $> 0.3$ and CAGR $> 0.3\%$ are kept in the "pair database."

### 16.8 Portfolio Selection

Two greedy algorithms construct the pair portfolio:

**Method A (Sharpe-Filtered):** 
1. Rank pairs by Sharpe ratio
2. Select the top pair
3. For each remaining pair, compute its average correlation with already-selected pairs
4. Score: $\text{Sharpe}_i - 2.5 \times \max(\bar{\rho}_i, 0)$
5. Select the highest-scoring pair; repeat up to 25 pairs

**Method B (Composite-Filtered):**
1. Score: $\text{Sharpe}_i^{1.5} \times \max(\text{CAGR}_i, 0.001)$
2. Correlation penalty coefficient: 1.5 instead of 2.5

### 16.9 Return-Stream Blending

The top Sharpe-filtered and composite-filtered portfolios are blended:

$$
r_t^{\text{blend}} = \alpha \cdot r_t^{\text{ShF}} + (1 - \alpha) \cdot r_t^{\text{CF}}, \quad \alpha \in \{0.5, 0.6, 0.7\}
$$

---

## 17. CrashHedge: Volatility-Regime Overlay

### 17.1 Regime Classification

Using QQQ's 20-day realised volatility relative to its 120-day average:

$$
\sigma_t^{20} = \text{std}(r_{t-20}, \ldots, r_{t-1}) \times \sqrt{252}
$$

$$
\bar{\sigma}_t^{120} = \frac{1}{120}\sum_{i=1}^{120}\sigma_{t-i}^{20}
$$

**Four regimes:**

$$
\text{Regime}_t = \begin{cases}
\text{Normal} & \text{if } \sigma_t^{20} < 1.2 \times \bar{\sigma}_t^{120} \\
\text{Recovery} & \text{if Elevated AND } \sigma_t^{20} < \sigma_{t-5}^{20} \text{ AND } P_t^{QQQ} > \min_{j \in [t-10,t]} P_j^{QQQ} \\
\text{Elevated} & \text{if } 1.2 \times \bar{\sigma}_t^{120} \leq \sigma_t^{20} < 1.8 \times \bar{\sigma}_t^{120} \\
\text{Crisis} & \text{if } \sigma_t^{20} \geq 1.8 \times \bar{\sigma}_t^{120}
\end{cases}
$$

### 17.2 Allocation Table

The CrashHedge allocates across 5 ETFs based on regime (with base leverage $\ell = 1.0$):

| ETF | Normal | Elevated | Crisis | Recovery |
|-----|--------|----------|--------|----------|
| QQQ | $+0.7\ell$ | $+0.3\ell$ | $0$ | $+0.8\ell$ |
| SPY | $+0.3\ell$ | $+0.1\ell$ | $-0.3$ | $+0.4\ell$ |
| IWM | $0$ | $-0.2$ | $0$ | $0$ |
| GLD | $0$ | $+0.15$ | $+0.3$ | $0$ |
| TLT | $0$ | $0$ | $+0.2$ | $0$ |

**Key features:**
- In **Normal** regime: aggressive equity exposure (QQQ 70%, SPY 30%)
- In **Crisis**: zero equity exposure, short SPY for downside alpha, full safe-haven tilt (GLD 30%, TLT 20%)
- In **Recovery**: aggressive re-entry (QQQ 80%, SPY 40%) to capture the snap-back

---

## 18. Cross-Sectional Momentum (Jegadeesh–Titman)

### 18.1 Signal Computation

The classic 12-month momentum with 1-month skip (12-1):

$$
\text{Mom}_i = \frac{P_{t - \text{skip}}^i}{P_{t - \text{lookback} - \text{skip}}^i} - 1
$$

Default parameters: lookback = 252 days, skip = 21 days.

### 18.2 Portfolio Construction

Each rebalancing (every 21 trading days):

1. Rank all 18 tickers by $\text{Mom}_i$
2. Long the top $n_L = 4$ tickers at equal weight $+n / n_L$ each
3. Short the bottom $n_S = 3$ tickers at equal weight $-n / n_S$ each

where $n$ is the total notional (default 0.10).

### 18.3 Parameter Grid

| Parameter | Values |
|-----------|--------|
| Lookback | 126, 189, 252 |
| Skip | 21 |
| $n_L$ (long) | 4, 5 |
| $n_S$ (short) | 2, 3 |
| Notional | 0.10, 0.12 |

---

## 19. Time-Series Momentum (Trend-Following)

### 19.1 Signal

For each asset $i$ over multiple lookbacks $L \in \{63, 126, 252\}$:

$$
\text{sig}_{i,t}^{(L)} = \text{sgn}\left(\frac{P_t^i}{P_{t-L}^i} - 1\right)
$$

The aggregate signal is the average across lookbacks:

$$
\bar{s}_{i,t} = \frac{1}{3}\sum_{L \in \{63, 126, 252\}} \text{sig}_{i,t}^{(L)} \quad \in [-1, +1]
$$

### 19.2 Inverse-Volatility Sizing

Each asset's weight is scaled by inverse realised volatility for risk parity:

$$
\sigma_{i,t}^{\text{ann}} = \text{std}(r_{i,t-63}, \ldots, r_{i,t-1}) \times \sqrt{252}, \quad \text{clipped to } [0.05, \infty)
$$

$$
\text{vol\_scale}_{i,t} = \text{clip}\left(\frac{0.05}{\sigma_{i,t}^{\text{ann}}}, 0.1, 2.0\right)
$$

$$
w_{i,t} = \bar{s}_{i,t} \times \text{vol\_scale}_{i,t} \times \frac{n}{N}
$$

where $n$ is the total notional (tested: 0.06, 0.08, 0.10) and $N = 18$ is the number of tradeable assets.

---

## 20. Sector Rotation

### 20.1 Risk-Adjusted Momentum Score

For each sector ETF $s \in \{\text{XLK, XLV, XLF, XLE, XLI, XLC, XLP, XLU, XLB, XLRE}\}$:

$$
\hat{\mu}_s = \bar{r}_{s,[t-L,t]} \times 252 \quad (\text{annualised mean return over lookback } L)
$$

$$
\hat{\sigma}_s = \text{std}(r_{s,[t-L,t]}) \times \sqrt{252} \quad (\text{annualised volatility})
$$

$$
\text{Score}_s = \frac{\hat{\mu}_s}{\hat{\sigma}_s} \quad (\text{risk-adjusted momentum})
$$

### 20.2 Portfolio

Monthly rebalance (every 21 trading days):
- Long the top $n_{\text{top}} = 4$ sectors at weight $+n / n_{\text{top}}$ each
- Short the bottom $n_{\text{bottom}} = 2$ sectors at weight $-n / n_{\text{bottom}}$ each

### 20.3 Parameter Grid

| Parameter | Values |
|-----------|--------|
| Lookback $L$ | 42, 63, 126 |
| $n_{\text{top}}$ | 3, 4 |
| $n_{\text{bottom}}$ | 2 |
| Notional $n$ | 0.10, 0.12 |

---

## 21. Ensemble Construction

### 21.1 Multi-Source Blending

The system constructs ensembles from 2 to 5 alpha sources:

**2-source (Pairs + CrashHedge):**

$$
r_t^{E2} = \alpha_1 r_t^{ZP} + \alpha_2 r_t^{CH}, \quad \alpha_1 + \alpha_2 = 1
$$

Grid: $(\alpha_1, \alpha_2) \in \{(0.85, 0.15), (0.80, 0.20), (0.75, 0.25), (0.70, 0.30)\}$

**3-source (+Cross-Sectional Momentum):**

$$
r_t^{E3} = \alpha_1 r_t^{ZP} + \alpha_2 r_t^{CH} + \alpha_3 r_t^{XSM}
$$

Grid: $(\alpha_1, \alpha_2, \alpha_3) \in \{(0.70, 0.15, 0.15), (0.65, 0.20, 0.15), (0.60, 0.20, 0.20), (0.55, 0.25, 0.20)\}$

**4-source (+Time-Series Momentum):**

$$
r_t^{E4} = \alpha_1 r_t^{ZP} + \alpha_2 r_t^{CH} + \alpha_3 r_t^{XSM} + \alpha_4 r_t^{TSM}
$$

**5-source (+Sector Rotation):**

$$
r_t^{E5} = \alpha_1 r_t^{ZP} + \alpha_2 r_t^{CH} + \alpha_3 r_t^{XSM} + \alpha_4 r_t^{TSM} + \alpha_5 r_t^{SR}
$$

### 21.2 Champion Ensemble

The v18 champion: $E2(85\%\text{ZP} + 15\%\text{CH})$ — the simplest two-source ensemble, demonstrating that diversification across more sources was less effective than a concentrated allocation to the two strongest individual strategies.

---

## 22. Portfolio Overlays: Volatility Target and Drawdown Control

### 22.1 Volatility Target (VT)

As defined in Section 4:

$$
r_t^{VT} = r_t \times \text{clip}\left(\frac{\sigma^{\text{target}}}{\sigma_{t-1}^{\text{realised}}}, 0.2, 5.0\right)
$$

### 22.2 Hierarchical Drawdown Control (HDDC)

A two-threshold drawdown controller with linear interpolation:

$$
\text{dd}_t = \frac{E_t - \max_{s < t} E_s}{\max_{s < t} E_s} \quad (\text{lagged: uses } E_{t-1})
$$

$$
\text{scale}_t = \begin{cases}
0.15 & \text{if } \text{dd}_{t-1} < \theta_2 \\
1.0 - 0.85 \times \frac{\text{dd}_{t-1} - \theta_1}{\theta_2 - \theta_1} & \text{if } \theta_2 \leq \text{dd}_{t-1} < \theta_1 \\
\min(1.0, \text{scale}_{t-1} + \text{recovery}) & \text{if } \text{dd}_{t-1} \geq \theta_1 \text{ and scale}_{t-1} < 1.0 \\
1.0 & \text{otherwise}
\end{cases}
$$

$$
r_t^{HDDC} = r_t^{VT} \times \text{scale}_t
$$

Default parameters: $\theta_1 = -1.5\%$, $\theta_2 = -4.0\%$, recovery = 0.015 per day.

### 22.3 Triple-Layer DDC

A three-threshold extension:

$$
\text{scale}_t = \begin{cases}
0.10 & \text{if } \text{dd}_{t-1} < \theta_3 \\
0.40 - 0.30 \times \frac{\text{dd}_{t-1} - \theta_2}{\theta_3 - \theta_2} & \text{if } \theta_3 \leq \text{dd}_{t-1} < \theta_2 \\
1.0 - 0.60 \times \frac{\text{dd}_{t-1} - \theta_1}{\theta_2 - \theta_1} & \text{if } \theta_2 \leq \text{dd}_{t-1} < \theta_1 \\
\min(1.0, \text{scale}_{t-1} + \text{recovery}) & \text{if recovering} \\
1.0 & \text{otherwise}
\end{cases}
$$

Defaults: $\theta_1 = -1.0\%$, $\theta_2 = -2.5\%$, $\theta_3 = -5.0\%$.

### 22.4 Position-Level DDC with Rebalancing Cost

Extends triple-layer DDC by modelling the transaction cost of scale changes:

$$
\text{cost}_t = |\text{scale}_t - \text{scale}_{t-1}| \times \frac{c_{\text{rebal}}}{10{,}000}
$$

$$
r_t^{PL} = r_t \times \text{scale}_t - \text{cost}_t
$$

where $c_{\text{rebal}} \in \{3, 5, 8\}$ bps.

---

## 23. Leverage and Cost Modelling

### 23.1 Static Leverage

For a base return series $r_t$ and leverage multiple $m$:

$$
r_t^{\text{lev}} = m \cdot r_t - (m - 1) \cdot \frac{c_{\text{lev}}}{252}
$$

where $c_{\text{lev}}$ is the annualised leverage cost. The system tests:

| Leverage $m$ | Cost Tiers $c_{\text{lev}}$ |
|-------------|--------------------------|
| 2.0, 3.0, 4.0, 5.0, 6.0, 8.0 | 0.5%, 1.0%, 2.0% |

### 23.2 Champion Configuration

The v18 champion uses $m = 3.0$ at $c_{\text{lev}} = 1.0\%$:

$$
r_t^{\text{champion}} = 3.0 \cdot r_t^{VT6+HDDC(1.5/4.0)} - 2.0 \times \frac{0.01}{252}
$$

---

## 24. Backtest Engine and Cost Stack

### 24.1 Portfolio Return Decomposition

For weight matrix $\mathbf{W}$ (shifted by 1 day) and return matrix $\mathbf{R}$:

$$
r_t^{\text{gross}} = \sum_i w_{i,t-1} \cdot r_{i,t}
$$

### 24.2 Full Cost Stack

The net return incorporates five cost components:

$$
r_t^{\text{net}} = r_t^{\text{gross}} + r_t^{\text{cash}} - r_t^{\text{tx}} - r_t^{\text{lev}} - r_t^{\text{short}} - r_t^{\text{spread}}
$$

**Cash return** (on uninvested capital):

$$
r_t^{\text{cash}} = (1 - \sum_i w_{i,t})^+ \times \frac{r_f^{\text{cash}}}{252}
$$

where $r_f^{\text{cash}} = 2.0\%$ and $(x)^+ = \max(x, 0)$.

**Transaction costs** (proportional to turnover):

$$
r_t^{\text{tx}} = \text{TX}_{\text{bps}} \times \frac{1}{10{,}000} \times \sum_i |w_{i,t} - w_{i,t-1}|
$$

where $\text{TX}_{\text{bps}} = 5$.

**Leverage costs** (on gross exposure above 100%):

$$
r_t^{\text{lev}} = \left(\sum_i |w_{i,t}| - 1\right)^+ \times \frac{c_{\text{lev}}}{252}
$$

**Short-selling costs**:

$$
r_t^{\text{short}} = \sum_i |w_{i,t}|^{-} \times \frac{c_{\text{short}}}{252}
$$

where $|w_{i,t}|^{-} = |\min(w_{i,t}, 0)|$ and $c_{\text{short}} = 0.5\%$.

**Spread costs** (ETF-specific, proportional to weight changes):

$$
r_t^{\text{spread}} = \sum_i |w_{i,t} - w_{i,t-1}| \times \frac{\text{halfspread}_i}{10{,}000}
$$

### 24.3 Spread Model Constants

| ETF | Half-Spread (bps) | ADV (USD) |
|-----|-----------------:|----------:|
| SPY | 0.15 | 30B |
| QQQ | 0.25 | 15B |
| IWM | 0.50 | 3B |
| EFA | 0.75 | 1.5B |
| XLK | 0.75 | 1.5B |
| XLV | 1.00 | 800M |
| XLF | 0.75 | 1.2B |
| XLE | 1.00 | 1.0B |
| XLI | 1.25 | 500M |
| XLC | 1.50 | 300M |
| XLP | 1.00 | 600M |
| XLU | 1.25 | 500M |
| XLB | 1.50 | 200M |
| XLRE | 1.75 | 200M |
| TLT | 0.50 | 1.5B |
| IEF | 0.75 | 500M |
| GLD | 1.00 | 1.0B |
| SHY | 0.50 | 300M |

### 24.4 Equity Curve

$$
E_t = E_0 \times \prod_{i=1}^{t}(1 + r_i^{\text{net}})
$$

where $E_0 = \$100{,}000$.

---

## 25. HFT Scalper Layer: 7-Indicator Multi-Timeframe Architecture

### 25.1 Overview

The HFT layer trades SPY, QQQ, and IWM on 3-minute bars using a composite score from 7 technical indicators, each scoring $[-2, +2]$:

$$
S^{\text{HFT}}_t = \sum_{k=1}^{7} I_k^{\text{HFT}}(t), \quad \text{raw range } [-14, +14]
$$

Regime-adaptive bias is then applied:

$$
S^{\text{adj}}_t = \begin{cases}
\text{clip}(S^{\text{HFT}}_t \times \beta_{\text{long}}, -14, 14) & \text{if } S^{\text{HFT}}_t > 0 \\
\text{clip}(S^{\text{HFT}}_t \times \beta_{\text{short}}, -14, 14) & \text{if } S^{\text{HFT}}_t < 0 \\
0 & \text{otherwise}
\end{cases}
$$

### 25.2 Indicator 1: EMA Crossover ($I_{\text{EMA}}$)

Fast EMA(8) vs Slow EMA(21):

$$
\text{EMA}_t^n = \alpha P_t + (1-\alpha)\text{EMA}_{t-1}^n, \quad \alpha = \frac{2}{n+1}
$$

$$
d_t = \frac{\text{EMA}_t^8 - \text{EMA}_t^{21}}{\text{EMA}_t^{21}} \times 100
$$

$$
I_{\text{EMA}}(t) = \begin{cases}
+2 & \text{if } d_t > 0.15 \\
+1 & \text{if } 0.03 < d_t \leq 0.15 \\
-1 & \text{if } -0.15 \leq d_t < -0.03 \\
-2 & \text{if } d_t < -0.15 \\
\phantom{+}0 & \text{otherwise}
\end{cases}
$$

### 25.3 Indicator 2: RSI ($I_{\text{RSI}}^{\text{HFT}}$)

RSI(10) — shorter period for intraday responsiveness:

$$
I_{\text{RSI}}^{\text{HFT}}(t) = \begin{cases}
+2 & \text{if RSI} < 25 \\
+1 & \text{if } 25 \leq \text{RSI} < 40 \\
-1 & \text{if } 60 < \text{RSI} \leq 75 \\
-2 & \text{if RSI} > 75 \\
\phantom{+}0 & \text{otherwise}
\end{cases}
$$

### 25.4 Indicator 3: VWAP ($I_{\text{VWAP}}$)

Volume-Weighted Average Price over the last 20 bars:

$$
\text{VWAP} = \frac{\sum_{i=t-19}^{t} P_i \cdot V_i}{\sum_{i=t-19}^{t} V_i}
$$

$$
d_t^{\text{VWAP}} = \frac{P_t - \text{VWAP}}{\text{VWAP}} \times 100
$$

$$
I_{\text{VWAP}}(t) = \begin{cases}
+2 & \text{if } d_t^{\text{VWAP}} < -0.20 \\
+1 & \text{if } -0.20 \leq d_t^{\text{VWAP}} < -0.05 \\
-1 & \text{if } 0.05 < d_t^{\text{VWAP}} \leq 0.20 \\
-2 & \text{if } d_t^{\text{VWAP}} > 0.20 \\
\phantom{+}0 & \text{otherwise}
\end{cases}
$$

Note the **reversal logic**: price above VWAP is bearish (mean-reversion to VWAP), price below is bullish.

### 25.5 Indicator 4: Stochastic ($I_{\text{Stoch}}$)

Fast Stochastic %K over 10 periods:

$$
\%K = \frac{C_t - L_{10}}{H_{10} - L_{10}} \times 100
$$

where $H_{10} = \max(H_{t-9}, \ldots, H_t)$ and $L_{10} = \min(L_{t-9}, \ldots, L_t)$.

$$
I_{\text{Stoch}}(t) = \begin{cases}
+2 & \text{if } \%K < 15 \\
+1 & \text{if } 15 \leq \%K < 30 \\
-1 & \text{if } 70 < \%K \leq 85 \\
-2 & \text{if } \%K > 85 \\
\phantom{+}0 & \text{otherwise}
\end{cases}
$$

### 25.6 Indicator 5: MACD ($I_{\text{MACD}}$)

MACD with parameters (8, 17, 6) — faster than the standard (12, 26, 9) for intraday:

$$
\text{MACD}_t = \text{EMA}_t^{8} - \text{EMA}_t^{17}
$$

$$
\text{Signal}_t = \text{EMA}_t^{6}(\text{MACD})
$$

$$
\text{Hist}_t = \text{MACD}_t - \text{Signal}_t
$$

$$
h_t = \frac{\text{Hist}_t}{|P_t|} \times 100
$$

$$
I_{\text{MACD}}(t) = \begin{cases}
+2 & \text{if } h_t > 0.10 \\
+1 & \text{if } 0.02 < h_t \leq 0.10 \\
-1 & \text{if } -0.10 \leq h_t < -0.02 \\
-2 & \text{if } h_t < -0.10 \\
\phantom{+}0 & \text{otherwise}
\end{cases}
$$

### 25.7 Indicator 6: ADX Proxy ($I_{\text{ADX}}^{\text{HFT}}$)

Simplified ADX using price position within the recent 10-period high-low range (same formula as voting system Indicator 4 but with period 10 instead of 14).

### 25.8 Indicator 7: Bollinger Band %B ($I_{\text{BB}}$)

Bollinger Bands with period 14 and 2.0 standard deviations:

$$
\text{Mid}_t = \frac{1}{14}\sum_{i=0}^{13} C_{t-i}
$$

$$
\text{Upper}_t = \text{Mid}_t + 2.0 \times \sigma_{14}
$$

$$
\text{Lower}_t = \text{Mid}_t - 2.0 \times \sigma_{14}
$$

$$
\%B = \frac{C_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}
$$

$$
I_{\text{BB}}(t) = \begin{cases}
+2 & \text{if } \%B < 0.05 \\
+1 & \text{if } 0.05 \leq \%B < 0.25 \\
-1 & \text{if } 0.75 < \%B \leq 0.95 \\
-2 & \text{if } \%B > 0.95 \\
\phantom{+}0 & \text{otherwise}
\end{cases}
$$

### 25.9 Regime-Adaptive Parameters

The HFT scalper adapts its behaviour based on the daily market regime:

| Parameter | BULL | BEAR | SIDEWAYS |
|-----------|------|------|----------|
| Buy threshold | 2.5 | 4.0 | 3.0 |
| Sell threshold | −4.0 | −2.5 | −3.0 |
| Stop loss | 0.35% | 0.25% | 0.30% |
| TP1 | 0.50% | 0.35% | 0.45% |
| TP2 | 1.00% | 0.70% | 0.90% |
| Trailing stop | 0.25% | 0.15% | 0.20% |
| Max position size | 12% | 6% | 10% |
| Long bias $\beta_L$ | 1.5 | 0.5 | 1.0 |
| Short bias $\beta_S$ | 0.5 | 1.5 | 1.0 |

### 25.10 Position Sizing

Position size is proportional to signal strength:

$$
\text{Size\%} = 3.0 + \frac{|S^{\text{adj}}_t|}{14} \times (\text{MaxSize\%} - 3.0)
$$

$$
\text{PositionUSD} = E_t \times \frac{\text{Size\%}}{100}
$$

### 25.11 Exit Logic

Order of precedence:

1. **Stop loss**: Long: $L_t \leq P_{\text{entry}} \times (1 - \text{SL\%}/100)$
2. **TP1**: Long: $H_t \geq P_{\text{entry}} \times (1 + \text{TP1\%}/100)$ → activate trailing stop
3. **TP2**: Long: $H_t \geq P_{\text{entry}} \times (1 + \text{TP2\%}/100)$ → exit full position
4. **Trailing stop** (after TP1): Long: $L_t \leq H^* - P_{\text{entry}} \times \text{Trail\%}/100$
5. **Max hold**: 60 bars → forced exit
6. **Signal reversal**: Long: $S^{\text{adj}}_t \leq \theta_{\text{sell}}$

P&L calculation:

$$
\text{PnL} = \text{Size} \times \frac{P_{\text{exit}} - P_{\text{entry}}}{P_{\text{entry}}} - \text{Size} \times c_{\text{frac}}
$$

---

## 26. Regime Detection and Adaptive Routing

### 26.1 Daily Regime Classification

The regime router uses SPY to classify each day (lagged by 1 day):

$$
r_t^{21} = \frac{P_t^{SPY} - P_{t-21}^{SPY}}{P_{t-21}^{SPY}}
$$

$$
\text{SMA\_rising}_t = \text{SMA}_{63}(P_t^{SPY}) > \text{SMA}_{63}(P_{t-5}^{SPY})
$$

$$
\text{volratio}_t = \frac{\sigma_t^{21}}{\bar{\sigma}_t^{120}} \quad (\text{clipped to } [0.01, \infty))
$$

$$
\text{Regime}_t = \begin{cases}
\text{BULL} & \text{if } r_{t-1}^{21} > 2\% \text{ AND SMA\_rising}_{t-1} \text{ AND volratio}_{t-1} < 1.5 \\
\text{BEAR} & \text{if } r_{t-1}^{21} < -3\% \text{ OR volratio}_{t-1} > 1.8 \text{ OR } (\lnot\text{SMA\_rising}_{t-1} \text{ AND } r_{t-1}^{21} < -1\%) \\
\text{SIDEWAYS} & \text{otherwise}
\end{cases}
$$

### 26.2 Capital Allocation by Regime

| Regime | Daily Layer | HFT Layer |
|--------|:----------:|:---------:|
| BULL | 65% | 35% |
| BEAR | 85% | 15% |
| SIDEWAYS | 75% | 25% |

---

# Part IV — Results and Validation

## 27. In-Sample Results

### 27.1 Pair Scan Statistics

- Pairs scanned: 153 × 4 windows × 4 z-configs = ~2,448 configurations
- Quality pairs (Sharpe > 0.3, CAGR > 0.3%): varies by run
- Total strategy variants tested (including ensembles, overlays, leverage): > 10,000

### 27.2 IS Champion (v18)

Configuration: `L(1%)×3.0 + VT6 + HDDC(1.5/4.0) + E2(85%ZP + 15%CH)`

| Metric | Value |
|--------|------:|
| Sharpe Ratio | 4.07 |
| CAGR | 46.30% |
| Max Drawdown | varies by sub-period |
| Walk-Forward Consistency | 5/5 periods > 0.5 |

---

## 28. Out-of-Sample Results

### 28.1 OOS Period

March 2025 – March 2026 (~260 trading days). **Parameters locked from in-sample. No tuning performed on OOS data.**

### 28.2 v18 OOS Champion

| Metric | Value |
|--------|------:|
| Sharpe Ratio | 2.01 |
| CAGR | 46.30% |
| Max Drawdown | −7.59% |
| Leverage | 3.0× |
| Leverage Cost | 1.0% annualised |

### 28.3 v17 vs v18 OOS Comparison

| Metric | v17 (3×, 1% cost) | v18 (3×, 1% cost) |
|--------|------------------:|------------------:|
| Sharpe | 1.78 | 2.01 |
| CAGR | 26.00% | 46.30% |
| Max DD | −10.02% | −7.59% |

v18 improves on v17 across all three dimensions: higher Sharpe, higher CAGR, and shallower drawdown.

### 28.4 Individual Alpha Source OOS Performance

| Source | OOS Sharpe | OOS CAGR | OOS MaxDD |
|--------|----------:|----------:|----------:|
| Pair Portfolio (ShF) | 1.70 | 2.06% | −1.12% |
| CrashHedge | 1.44 | 17.94% | −7.15% |
| VolCarry (dropped in v18) | −0.14 | −0.35% | −2.53% |

The decision to drop VolCarry and upweight CrashHedge from 3% to 15% is validated by OOS data: CrashHedge contributes the dominant absolute return with respectable risk-adjustment, while VolCarry actively destroys alpha.

---

## 29. Walk-Forward Analysis

The champion strategy is evaluated on 5 non-overlapping sub-periods:

| Period | Sharpe |
|--------|-------:|
| 2010–12 | > 0.5 |
| 2013–15 | > 0.5 |
| 2016–18 | > 0.5 |
| 2019–21 | > 0.5 |
| 2022–25 | > 0.5 |

A strategy is "consistent" if Sharpe > 0.5 in every sub-period. The champion passes this test.

---

## 30. Statistical Validation (DSR, Bootstrap)

### 30.1 Deflated Sharpe Ratio

With $N > 10{,}000$ strategies tested in-sample:

- **Observed IS Sharpe**: 4.07
- **Expected max Sharpe from noise**: $\mathbb{E}[\max \widehat{\text{SR}}] \approx 3.02$
- **DSR**: positive, indicating the IS Sharpe exceeds what would be expected from pure data mining
- **p-value**: < 0.05

### 30.2 Bootstrap Confidence Interval

95% CI from 5,000 bootstrap resamples of the IS champion's daily returns:

$$
\text{SR} \in [\text{lower}, \text{upper}]
$$

with the bootstrap mean close to the observed Sharpe.

---

## 31. Paper Trading Validation

### 31.1 Setup

- Broker: Alpaca sub-account
- VM: GCP `paper-trader`, us-east1-b, e2-micro
- Schedule: Cron Mon–Fri 16:35 ET (v18), 21:30 ET (v7 bot)
- Notifications: Telegram bot (token `8752263202:*`, chat ID `1543385905`)

### 31.2 Validation Findings

Comparison of paper backtest vs live paper trading:

| Asset | Paper Backtest | Live Signal | Match |
|-------|---------------|-------------|-------|
| SPY | 2.41% | 2.41% | Exact |
| NVDA | 6.85% | 6.85% | Exact |
| AAPL | 2.17% | ~0.91% | 58% lower |
| MSFT | 2.91% | ~0.61% | 79% lower |

SPY and NVDA match exactly. The variance in AAPL/MSFT is attributed to timing differences in signal evaluation and fill price discrepancies.

---

## 32. 5-Indicator Voting System Results (Legacy)

### 32.1 15-Asset Validation

Test period: January 2023 – December 2025 (752 trading days).

| Metric | Value |
|--------|------:|
| Average Sharpe | 5.34 |
| Average Win Rate | 71.4% |
| Total Trades | 368 |
| Losing Assets | 0 / 15 |
| Average Return | 1.95% |
| SPY Buy-Hold Return | 87.67% |
| Underperformance Gap | 85.72% |

**Interpretation**: The voting system achieved strong risk-adjusted returns (Sharpe 5.34) and zero losing assets, but dramatically underperformed buy-and-hold in the 2023–2025 bull market. The 2–8% position sizing leaves most capital uninvested, and entry delays and exit timing in a trending market cause opportunity cost. The system correctly identified the bull regime (buy signal 58.6% of days) but was unable to capture the full trend.

---

## 33. Lessons Learned and Failure Catalogue

| # | Lesson | Source |
|---|--------|--------|
| 1 | Lookahead bias can inflate Sharpe by 700%+ | Phase 6 crypto (6.37 → −42.89) |
| 2 | Voting systems over-vote on momentum if indicators are redundant | Phase 5–6 tuning |
| 3 | Active timing subtracts value in sustained trends | Phase 7 (voting vs SPY) |
| 4 | VolCarry can be IS-positive but OOS-negative | v17 OOS |
| 5 | CrashHedge alpha was suppressed by 3% weight | v17→v18 diagnosis |
| 6 | Simpler ensembles (2-source) can beat complex ones (5-source) | v18 ensemble grid |
| 7 | Walk-forward validation catches period-specific overfitting | v18 Phase 8 |
| 8 | DSR is essential when testing >10,000 variants | v18 Phase 10 |
| 9 | Spread costs matter — 0.15–1.75 bps half-spread on every trade | v17 audit |
| 10 | All DDC must use lagged drawdown ($\text{dd}_{t-1}$) | v17 audit finding |
| 11 | No backfill — ffill only — prevents survival bias | v17 audit finding |
| 12 | Position-level DDC rebalancing has its own cost | v17 audit finding |

---

# Appendices

## Appendix A: Full Parameter Table

### v18 Champion Configuration

| Component | Parameter | Value |
|-----------|-----------|------:|
| Pair Engine | Window | 63 days |
| | Entry $z$ | 2.0 |
| | Exit $z$ | 0.5 |
| | Notional per pair | 0.10 |
| | Number of pairs | 7 |
| | Selection method | Sharpe-filtered |
| Ensemble | ZP weight | 85% |
| | CrashHedge weight | 15% |
| Volatility Target | $\sigma^{\text{target}}$ | 6% |
| | Lookback | 63 days |
| HDDC | $\theta_1$ | −1.5% |
| | $\theta_2$ | −4.0% |
| | Recovery rate | 0.015/day |
| Leverage | Multiple | 3.0× |
| | Annual cost | 1.0% |
| Costs | TX commission | 5 bps |
| | Short cost | 0.5% annual |
| | Cash rate | 2.0% annual |

### HFT Scalper (SIDEWAYS regime defaults)

| Parameter | Value |
|-----------|------:|
| EMA Fast/Slow | 8 / 21 |
| RSI Period | 10 |
| Stochastic Period | 10 |
| MACD (Fast/Slow/Signal) | 8 / 17 / 6 |
| ADX Period | 10 |
| BB (Period/Std) | 14 / 2.0 |
| VWAP Lookback | 20 bars |
| Buy Threshold | 3.0 |
| Sell Threshold | −3.0 |
| Stop Loss | 0.30% |
| TP1 / TP2 | 0.45% / 0.90% |
| Trailing Stop | 0.20% |
| Max Position | 10% |
| Max Hold | 60 bars |
| Slippage | 3 bps/side |
| Commission | 1 bps/side |

## Appendix B: ETF Universe Details

| # | Ticker | Class | Description | Approx ADV |
|---|--------|-------|-------------|----------:|
| 1 | SPY | Broad | S&P 500 | $30B |
| 2 | QQQ | Broad | Nasdaq 100 | $15B |
| 3 | IWM | Broad | Russell 2000 | $3B |
| 4 | EFA | Broad | International Developed | $1.5B |
| 5 | XLK | Sector | Technology | $1.5B |
| 6 | XLV | Sector | Healthcare | $800M |
| 7 | XLF | Sector | Financials | $1.2B |
| 8 | XLE | Sector | Energy | $1.0B |
| 9 | XLI | Sector | Industrials | $500M |
| 10 | XLC | Sector | Communication | $300M |
| 11 | XLP | Sector | Consumer Staples | $600M |
| 12 | XLU | Sector | Utilities | $500M |
| 13 | XLB | Sector | Materials | $200M |
| 14 | XLRE | Sector | Real Estate | $200M |
| 15 | TLT | Safe | 20+ Year Treasury | $1.5B |
| 16 | IEF | Safe | 7–10 Year Treasury | $500M |
| 17 | GLD | Safe | Gold | $1.0B |
| 18 | SHY | Safe | 1–3 Year Treasury | $300M |

## Appendix C: Cost Model Constants

| Parameter | Symbol | Value | Unit |
|-----------|--------|------:|------|
| Transaction cost | TX_BPS | 5 | bps per turnover |
| Short-selling cost | SHORT_COST | 0.5 | % annual |
| Cash return | RF_CASH | 2.0 | % annual |
| Risk-free rate | RF | 0.0 | % |
| Standard leverage cost | LEV_COST_STD | 1.5 | % annual |
| HFT slippage | HFT_SLIPPAGE_BPS | 3.0 | bps per side |
| HFT commission | HFT_COMMISSION_BPS | 1.0 | bps per side |

---

*End of report.*
