import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from src.tools.pipeline_tool import RunBacktestTool
from src.tools.portfolio_tools import (
    GetMarketDataTool,
    ComputeFactorSignalsTool,
    BuildPortfolioTool,
    RunCustomBacktestTool,
)


def make_llm():
    # Minimal wrapper for ChatOpenAI (OpenAI 1.x)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    return ChatOpenAI(model=model, temperature=0)


def build_agents(use_llm: bool = True):
    # Create tools
    backtest_tool = RunBacktestTool()
    market_data_tool = GetMarketDataTool()
    factor_signals_tool = ComputeFactorSignalsTool()
    build_portfolio_tool = BuildPortfolioTool()
    custom_backtest_tool = RunCustomBacktestTool()

    llm = make_llm() if use_llm else None

    universe = Agent(
        role="Universe Curator",
        goal=(
            "Define the investable equity universe for a quarterly factor strategy. "
            "Apply exclusions and ensure sector metadata is available."
        ),
        backstory=(
            "You are a data-driven equity PM assistant. You ensure tradability, listing status, and policy exclusions."
        ),
        tools=[market_data_tool],
        llm=llm,
        allow_delegation=False,
    )

    data = Agent(
        role="Market Data Engineer",
        goal=(
            "Provide clean, point-in-time prices and fundamentals. "
            "Ensure no lookahead and align quarterly fundamentals to daily index."
        ),
        backstory=("You specialize in data QA and temporal alignment for backtests."),
        tools=[market_data_tool, factor_signals_tool],
        llm=llm,
        allow_delegation=False,
    )

    signal = Agent(
        role="Factor Researcher",
        goal=(
            "Compute factor z-scores: Momentum (252d, skip 5d), Value (inverse PE), "
            "Quality (ROA), Low-Vol (inverse 63d vol). Blend per config."
        ),
        backstory=("You run robust, simple factors and avoid leakage by design."),
        tools=[factor_signals_tool],
        llm=llm,
        allow_delegation=False,
    )

    risk = Agent(
        role="Risk Officer",
        goal=(
            "Translate policy into constraints and budgets: long-only, gross leverage, per-name/sector caps, and max holdings."
        ),
        backstory=(
            "You ensure risk discipline and concentration limits are always respected."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
    )

    portfolio = Agent(
        role="Portfolio Constructor",
        goal=(
            "Map scores to target weights while respecting constraints and promoting diversification and stability. "
            "Make intelligent decisions about portfolio parameters based on market conditions and factor performance."
        ),
        backstory=(
            "You prefer stability and controlled turnover while achieving exposure to desired factors. "
            "You can adjust portfolio parameters like max names, position limits, and factor weights based on analysis."
        ),
        tools=[build_portfolio_tool, factor_signals_tool],
        llm=llm,
        allow_delegation=False,
    )

    execution = Agent(
        role="Execution Trader",
        goal=(
            "Generate trades to reach target weights on rebalance dates with realistic cost and slippage assumptions."
        ),
        backstory=("You minimize costs and control turnover in quarterly rebalances."),
        tools=[],
        llm=llm,
        allow_delegation=False,
    )

    backtest = Agent(
        role="Backtest Analyst",
        goal=(
            "Run the full quarterly backtest and output metrics: CAGR, Sharpe, Sortino, MaxDD, Calmar, Vol, HitRate. "
            "Use custom portfolio configurations to test different strategies and compare results."
        ),
        backstory=(
            "You deliver auditable results and artifacts for performance review. "
            "You can run multiple backtests with different configurations to find optimal strategies."
        ),
        tools=[backtest_tool, custom_backtest_tool, build_portfolio_tool],
        llm=llm,
        allow_delegation=False,
    )

    compliance = Agent(
        role="Compliance Officer",
        goal=(
            "Check portfolio and trades for concentration, sector caps, liquidity floors, blacklist and ESG flags."
        ),
        backstory=("You enforce firm policy before any result is published."),
        tools=[],
        llm=llm,
        allow_delegation=False,
    )

    explain = Agent(
        role="Investment Writer",
        goal=(
            "Draft a concise investment memo covering top names, factor drivers, risk posture, and key metrics."
        ),
        backstory=("You communicate clearly with PMs and stakeholders."),
        tools=[],
        llm=llm,
        allow_delegation=False,
    )

    return {
        "universe": universe,
        "data": data,
        "signal": signal,
        "risk": risk,
        "portfolio": portfolio,
        "execution": execution,
        "backtest": backtest,
        "compliance": compliance,
        "explain": explain,
    }
