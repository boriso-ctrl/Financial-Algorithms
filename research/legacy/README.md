# Legacy / Archived Materials

This folder contains historical or exploratory content that is not part of the current backtesting and signal stack. Use it for reference only.

## Contents
- Financial-Algorithm Contents/ — collection of legacy research (deep learning trading, NLP, robust strategies, technical indicators, etc.).
- Forex/ — FX pair trading examples and historical CSV/result artifacts.

## Active code lives elsewhere
The maintained code paths are:
- backtest/ (engine, CLI, signal blending)
- strategies/, indicators/, volume/ (current signal implementations)
- scripts/ (runnable demos, e.g., scripts/demo_blend.py)
- tests/ (coverage for the active stack)

If you need to revive anything from legacy, copy it into a fresh module under strategies/ or scripts/ and add tests before wiring it into the backtester.
