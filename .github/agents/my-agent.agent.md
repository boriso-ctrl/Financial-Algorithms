---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name:
Peter

# My Agent

The repository lacks a GitHub Copilot coding agent specification to guide future strategy development toward world-class performance metrics — specifically Sharpe ≥ 3, CAGR that beats buy-and-hold SPY, high trading frequency, and audited zero lookahead bias.

### Changes

**`.github/agents/quant-agent.md`** — new agent description file:

- **Performance targets** (hard constraints): Sharpe ≥ 3, CAGR > SPY on same window, max drawdown ≤ 25%, ≥ 2 trades/week, zero lookahead bias
- **HF strategy roadmap** — three prioritised approaches to close the current CAGR gap (0.62% vs SPY's 23.35% CAGR) while preserving the existing Sharpe 5.34:
  1. Intraday mean-reversion on 1-hour bars (multi-timeframe anchor)
  2. Momentum burst on 15-minute bars
  3. Daily multi-asset rotation with Kelly sizing
- **Lookahead bias enforcement** — canonical codebase pattern codified, forbidden anti-patterns listed with correct alternatives:
  ```python
  # ❌ Future leak
  signals = compute_signals(prices)          # applied same bar as calculated

  # ✅ Correct: execute at open of bar t+1
  signals = compute_signals(prices).shift(1)
  ```
- **Strategy development workflow** — IS → walk-forward (6m train / 3m OOS folds) → full OOS freeze → no-lookahead audit → PR checklist
- **Repository map** — full `src/financial_algorithms/` tree with file roles, grounded in the actual codebase structure

<!-- START COPILOT CODING AGENT TIPS -->
---

📍 Connect Copilot coding agent with [Jira](https://gh.io/cca-jira-docs), [Azure Boards](https://gh.io/cca-azure-boards-docs) or [Linear](https://gh.io/cca-linear-docs) to delegate work to Copilot in one click without leaving your project management tool.
