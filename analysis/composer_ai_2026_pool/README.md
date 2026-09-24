# Composer AI strategy candidate stock pool

Snapshot: 2026-09-23. `candidate_stocks.csv` is the union of stock tickers listed in the current public versions of three Composer strategies:

- [AI Native Biotechs](https://www.composer.trade/trading-strategies/ai-native-biotechs-nMo0uNhpaC0iRG0aklXz)
- [AI Boom Strategy](https://www.composer.trade/trading-strategies/ai-boom-strategy-lbgKxZ1bqbjMyOH8lSWR)
- [AI Focused Risk Managed Strategy](https://www.composer.trade/trading-strategies/ai-focused-risk-managed-strategy-xs4FPuP6F32btaFtKvqQ)

The 19 rows are **eligible strategy constituents**, not a verified list of stocks actually selected on each day of 2026. Public strategy pages show current eligible tickers and current holdings, but do not expose historical allocations. Composer's backtest view has a Historical Allocations table. A precise historical selection list requires that table's export or an authenticated backtest response.

BOTZ (AI and robotics ETF) and IEF (Treasury ETF) are excluded because this file is a stock pool. The six stocks in AI Focused Risk Managed Strategy also appear in AI Boom Strategy. `pool_available_from` uses the creation date of the earliest source strategy containing that ticker; use it as a lower bound when constructing a historical strategy-derived universe. In particular, the seven AI Native Biotechs tickers entered this strategy-derived pool on 2026-06-22, so including them in a January 2026 validation as though they had already been sourced from this strategy would introduce look-ahead bias.

This file does not apply a business-quality screen and does not assert that every constituent has a direct AI revenue stream. It is a source pool for the user's independent algorithm validation.
