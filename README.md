# CrewAI — Quarterly Portfolio Agent Team (Demo)

This repository creates a **CrewAI** team that designs a **quarterly equity portfolio** and **runs a backtest** using mock, offline data.
You can swap in real market data and constraints later.

## Quick Start
```bash
cd crew_portfolio_agents
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# set your keys
cp .env.example .env
# edit .env to include OPENAI_API_KEY and (optionally) OPENAI_MODEL

# run with CrewAI (LLM tasks + tool)
python run.py --start 2020-01-01 --end 2024-12-31

# or, bypass LLM and call the tool directly (for offline demo):
python run.py --start 2020-01-01 --end 2024-12-31 --no-llm
```

Artifacts are written under `reports/`:
- `summary.md`: metrics and compliance checks
- `equity_curve.csv`: daily equity curve
- `weights.csv`: quarter-end portfolio weights
- `trades.csv`: trade list

## Agents (CrewAI)
- **UniverseAgent**: Investable universe and exclusions
- **DataAgent**: Point-in-time prices & fundamentals (mock CSVs here)
- **SignalAgent**: Factor z-scores (Momentum, Value, Quality, Low-Vol)
- **RiskAgent**: Translate policy to constraints/budgets
- **PortfolioAgent**: Convert scores -> target weights under caps
- **ExecutionAgent**: Quarterly rebalance with costs/slippage
- **ComplianceAgent**: Check concentration, liquidity, blacklist/ESG
- **BacktestAgent**: Run simulation and compute metrics
- **ExplainAgent**: Write an investment rationale memo

## Swap in real data
- Edit `src/tools/pipeline_tool.py` → `load_prices()` / `load_fundamentals()` to call your APIs.
- Constraints live in `config/constraints.yaml`.
- Prompts and task specs in `src/agents.py` and `src/tasks.py`.