from crewai import Task
from datetime import datetime


def build_tasks(agents, start: str, end: str):
    # Enhanced tasks that allow agents to make actual portfolio decisions
    t1 = Task(
        description=(
            f"Analyze the investable universe for the date range {start} to {end}. "
            "Use the get_market_data tool to examine available stocks, sectors, and data quality. "
            "Identify any potential issues with the universe and suggest improvements."
        ),
        expected_output="A comprehensive analysis of the investable universe with recommendations.",
        agent=agents["universe"],
    )

    t2 = Task(
        description=(
            f"Perform comprehensive data analysis for the period {start} to {end}. "
            "Use the get_market_data and compute_factor_signals tools to examine data quality and factor performance. "
            "Identify any data issues, factor behavior patterns, and market conditions."
        ),
        expected_output="Detailed data analysis with factor performance insights and market condition assessment.",
        agent=agents["data"],
    )

    t3 = Task(
        description=(
            f"Analyze factor signals for the period {start} to {end}. "
            "Use the compute_factor_signals tool to examine how different factors are performing. "
            "Identify which factors are showing strength and which might need adjustment."
        ),
        expected_output="Factor analysis with recommendations for factor weight adjustments based on current market conditions.",
        agent=agents["signal"],
    )

    t4 = Task(
        description=(
            "Review and optimize risk constraints based on current market conditions and factor analysis. "
            "Consider adjusting position limits, sector caps, and other risk parameters to improve portfolio performance."
        ),
        expected_output="Optimized risk policy with specific parameter recommendations.",
        agent=agents["risk"],
    )

    t5 = Task(
        description=(
            f"Design an optimal portfolio strategy for the period {start} to {end}. "
            "Based on the factor analysis and market conditions, determine the best portfolio parameters including: "
            "- Number of positions (max_names) "
            "- Position size limits (max_weight_per_name) "
            "- Sector allocation limits (max_weight_per_sector) "
            "- Factor weights for momentum, value, quality, and low-vol "
            "Use the build_portfolio tool to test your configuration and provide a JSON configuration."
        ),
        expected_output="A JSON portfolio configuration with detailed reasoning for each parameter choice.",
        agent=agents["portfolio"],
    )

    t6 = Task(
        description=(
            "Review the execution strategy and transaction cost assumptions. "
            "Consider if the current quarterly rebalancing and cost parameters are optimal for the designed portfolio."
        ),
        expected_output="Execution strategy review with recommendations for rebalancing frequency and cost optimization.",
        agent=agents["execution"],
    )

    # Run the custom backtest with AI-designed portfolio
    t7 = Task(
        description=(
            f"Run a comprehensive backtest using the AI-designed portfolio configuration for {start} to {end}. "
            "Use the run_custom_backtest tool with the portfolio configuration from the Portfolio Constructor. "
            "Compare results with baseline and provide detailed analysis of performance, risk, and compliance."
        ),
        expected_output="Comprehensive backtest results with performance analysis and comparison to baseline.",
        agent=agents["backtest"],
    )

    t8 = Task(
        description=(
            "Analyze compliance results from the custom backtest. "
            "Identify any violations and suggest adjustments to the portfolio configuration to improve compliance."
        ),
        expected_output="Compliance analysis with specific recommendations for improving portfolio compliance.",
        agent=agents["compliance"],
    )

    t9 = Task(
        description=(
            "Write a comprehensive investment memo based on the AI-designed portfolio and backtest results. "
            "Include analysis of the portfolio strategy, performance metrics, risk management, and recommendations for future optimization."
        ),
        expected_output="A detailed investment memo suitable for portfolio management review and stakeholder communication.",
        agent=agents["explain"],
    )

    return [t1, t2, t3, t4, t5, t6, t7, t8, t9]
