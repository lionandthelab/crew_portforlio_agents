import pandas as pd
import numpy as np
import yaml
import os
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from src.tools.pipeline_tool_impl import (
    load_prices,
    load_fundamentals,
    compute_signals,
    run_backtest,
    compliance_checks,
    explain,
)
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def generate_equity_curve_plot(
    equity_curve: pd.DataFrame,
    portfolio_name: str,
    output_path: str,
    figsize: tuple = (12, 8),
) -> str:
    """
    포트폴리오 수익률 곡선 그래프를 생성합니다.

    Args:
        equity_curve: 수익률 곡선 DataFrame
        portfolio_name: 포트폴리오 이름
        output_path: 그래프 저장 경로
        figsize: 그래프 크기

    Returns:
        저장된 그래프 파일 경로
    """
    try:
        # 그래프 스타일 설정
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )

        # 수익률 곡선
        ax1.plot(
            equity_curve.index,
            equity_curve["portfolio_value"],
            linewidth=2,
            color="#2E86AB",
            label="Portfolio Value",
        )
        ax1.set_title(
            f"{portfolio_name} - Equity Curve", fontsize=16, fontweight="bold", pad=20
        )
        ax1.set_ylabel("Portfolio Value", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # 일일 수익률
        ax2.bar(
            equity_curve.index,
            equity_curve["daily_return"],
            alpha=0.6,
            color="#A23B72",
            width=1,
        )
        ax2.set_title("Daily Returns", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Daily Return", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # x축 날짜 포맷팅
        for ax in [ax1, ax2]:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # 그래프 저장
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Equity curve plot saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate equity curve plot: {e}")
        raise


def generate_performance_comparison_table(
    portfolio_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    포트폴리오 성과 비교 표를 생성합니다.

    Args:
        portfolio_results: 포트폴리오 백테스트 결과 리스트

    Returns:
        성과 비교 DataFrame
    """
    try:
        comparison_data = []

        for result in portfolio_results:
            portfolio_info = result.get("portfolio_info", {})
            metrics = result.get("metrics", {})

            comparison_data.append(
                {
                    "Portfolio Name": portfolio_info.get("name", "Unknown"),
                    "Portfolio Type": portfolio_info.get("type", "Unknown"),
                    "Total Return (%)": f"{metrics.get('total_return', 0) * 100:.2f}",
                    "Annualized Return (%)": f"{metrics.get('annualized_return', 0) * 100:.2f}",
                    "Volatility (%)": f"{metrics.get('volatility', 0) * 100:.2f}",
                    "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
                    "Max Drawdown (%)": f"{metrics.get('max_drawdown', 0) * 100:.2f}",
                    "Win Rate (%)": f"{metrics.get('win_rate', 0) * 100:.1f}",
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        logger.info(
            f"Generated performance comparison table with {len(comparison_df)} portfolios"
        )
        return comparison_df

    except Exception as e:
        logger.error(f"Failed to generate performance comparison table: {e}")
        raise


def save_performance_comparison_table(
    comparison_df: pd.DataFrame, output_path: str
) -> str:
    """
    성과 비교 표를 CSV 파일로 저장합니다.

    Args:
        comparison_df: 성과 비교 DataFrame
        output_path: 저장 경로

    Returns:
        저장된 파일 경로
    """
    try:
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Performance comparison table saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save performance comparison table: {e}")
        raise


class GetMarketDataTool(BaseTool):
    name: str = "get_market_data"
    description: str = (
        "Load market data (prices and fundamentals) for the specified date range. Returns data summary."
    )

    def _run(self, start: str, end: str) -> str:
        try:
            prices = load_prices(start, end, use_real_data=True)
            fundamentals = load_fundamentals(start, end, use_real_data=True)
            
            # Filter by date range (if using mock data)
            if len(prices) > 0 and prices["date"].min() < pd.to_datetime(start):
                prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()

            # Get unique tickers and sectors
            tickers = prices["ticker"].unique()
            sectors = fundamentals[fundamentals["ticker"].isin(tickers)][
                ["ticker", "sector"]
            ].drop_duplicates()

            summary = f"""
Market Data Summary:
- Date Range: {start} to {end}
- Total Tickers: {len(tickers)}
- Available Sectors: {', '.join(sectors['sector'].unique())}
- Price Data Points: {len(prices)}
- Fundamental Data Points: {len(fundamentals[fundamentals['ticker'].isin(tickers)])}

Top 10 Tickers by Market Cap (estimated):
{', '.join(tickers[:10])}
"""
            return summary
        except Exception as e:
            return f"Error loading market data: {str(e)}"

    async def _arun(self, start: str, end: str) -> str:
        return self._run(start, end)


class ComputeFactorSignalsTool(BaseTool):
    name: str = "compute_factor_signals"
    description: str = (
        "Compute factor signals (momentum, value, quality, low-vol) for all stocks in the universe."
    )

    def _run(self, start: str, end: str) -> str:
        try:
            cfg = self._load_config()
            prices = load_prices(start, end, use_real_data=True)
            fundamentals = load_fundamentals(start, end, use_real_data=True)

            # Filter by date range
            if len(prices) > 0 and prices["date"].min() < pd.to_datetime(start):
                prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()

            # Compute signals
            scores = compute_signals(prices, fundamentals, cfg)

            # Get latest scores
            latest_date = scores["date"].max()
            latest_scores = scores[scores["date"] == latest_date].sort_values(
                "score", ascending=False
            )

            summary = f"""
Factor Signals Computed:
- Latest Date: {latest_date}
- Total Stocks: {len(latest_scores)}
- Factor Weights: {cfg.get('factors', {})}

Top 10 Stocks by Composite Score:
{latest_scores.head(10)[['ticker', 'score']].to_string(index=False)}

Bottom 5 Stocks by Composite Score:
{latest_scores.tail(5)[['ticker', 'score']].to_string(index=False)}
"""
            return summary
        except Exception as e:
            return f"Error computing signals: {str(e)}"

    def _load_config(self) -> Dict[str, Any]:
        with open("config/constraints.yaml", "r") as f:
            return yaml.safe_load(f)

    async def _arun(self, start: str, end: str) -> str:
        return self._run(start, end)


class BuildPortfolioTool(BaseTool):
    name: str = "build_portfolio"
    description: str = (
        "Build a portfolio based on factor scores and constraints. Input: JSON with portfolio parameters."
    )

    def _run(self, portfolio_config: str) -> str:
        try:
            import json

            config = json.loads(portfolio_config)

            # Load data
            start = config.get("start", "2020-01-01")
            end = config.get("end", "2025-07-31")
            cfg = self._load_config()

            prices = load_prices(start, end, use_real_data=True)
            fundamentals = load_fundamentals(start, end, use_real_data=True)

            # Filter by date range (if using mock data)
            if len(prices) > 0 and prices["date"].min() < pd.to_datetime(start):
                prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()

            # Compute signals
            scores = compute_signals(prices, fundamentals, cfg)
            sectors = fundamentals[["ticker", "sector"]].drop_duplicates()

            # Apply custom portfolio parameters
            custom_cfg = cfg.copy()
            if "max_names" in config:
                custom_cfg["max_names"] = config["max_names"]
            if "max_weight_per_name" in config:
                custom_cfg["max_weight_per_name"] = config["max_weight_per_name"]
            if "max_weight_per_sector" in config:
                custom_cfg["max_weight_per_sector"] = config["max_weight_per_sector"]
            if "factors" in config:
                custom_cfg["factors"] = config["factors"]

            # Build portfolio using existing logic
            from src.tools.pipeline_tool_impl import build_portfolio, execute_trades

            target_weights = build_portfolio(scores, sectors, custom_cfg)
            weights, trades = execute_trades(target_weights, custom_cfg)

            # Get latest portfolio
            latest_date = weights["date"].max()
            latest_weights = weights[weights["date"] == latest_date].sort_values(
                "weight", ascending=False
            )

            # Calculate sector allocation
            sector_alloc = (
                latest_weights.merge(sectors, on="ticker")
                .groupby("sector")["weight"]
                .sum()
                .sort_values(ascending=False)
            )

            summary = f"""
Portfolio Built Successfully:
- Rebalance Date: {latest_date}
- Total Positions: {len(latest_weights)}
- Portfolio Value: {latest_weights['weight'].sum():.2%}

Top 10 Holdings:
{latest_weights.head(10)[['ticker', 'weight']].to_string(index=False)}

Sector Allocation:
{sector_alloc.to_string()}

Portfolio Configuration Applied:
- Max Names: {custom_cfg.get('max_names', 'default')}
- Max Weight per Name: {custom_cfg.get('max_weight_per_name', 'default'):.1%}
- Max Weight per Sector: {custom_cfg.get('max_weight_per_sector', 'default'):.1%}
"""
            return summary
        except Exception as e:
            return f"Error building portfolio: {str(e)}"

    def _load_config(self) -> Dict[str, Any]:
        with open("config/constraints.yaml", "r") as f:
            return yaml.safe_load(f)

    async def _arun(self, portfolio_config: str) -> str:
        return self._run(portfolio_config)


class RunCustomBacktestTool(BaseTool):
    name: str = "run_custom_backtest"
    description: str = (
        "Run backtest with custom portfolio configuration. Input: JSON with portfolio parameters."
    )

    def _run(self, portfolio_config: str) -> str:
        try:
            import json

            config = json.loads(portfolio_config)

            # Load data
            start = config.get("start", "2020-01-01")
            end = config.get("end", "2025-07-31")
            cfg = self._load_config()

            prices = load_prices(start, end, use_real_data=True)
            fundamentals = load_fundamentals(start, end, use_real_data=True)

            # Filter by date range (if using mock data)
            if len(prices) > 0 and prices["date"].min() < pd.to_datetime(start):
                prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()

            # Compute signals
            scores = compute_signals(prices, fundamentals, cfg)
            sectors = fundamentals[["ticker", "sector"]].drop_duplicates()

            # Apply custom portfolio parameters
            custom_cfg = cfg.copy()
            if "max_names" in config:
                custom_cfg["max_names"] = config["max_names"]
            if "max_weight_per_name" in config:
                custom_cfg["max_weight_per_name"] = config["max_weight_per_name"]
            if "max_weight_per_sector" in config:
                custom_cfg["max_weight_per_sector"] = config["max_weight_per_sector"]
            if "factors" in config:
                custom_cfg["factors"] = config["factors"]

            # Build portfolio and run backtest
            from src.tools.pipeline_tool_impl import (
                build_portfolio,
                execute_trades,
                save_report,
            )

            target_weights = build_portfolio(scores, sectors, custom_cfg)
            weights, trades = execute_trades(target_weights, custom_cfg)

            # Run backtest
            equity, metrics = run_backtest(weights, trades, prices, custom_cfg)

            # Compliance checks
            comp_pass, vios = compliance_checks(weights, sectors, custom_cfg)

            # Generate explanation
            memo = explain(weights, metrics)

            # Save report with custom timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = f"reports/custom_{timestamp}"
            os.makedirs(report_dir, exist_ok=True)

            # Save files with custom names
            equity.to_csv(f"{report_dir}/equity_curve.csv", index=True)
            weights.to_csv(f"{report_dir}/weights.csv", index=False)
            trades.to_csv(f"{report_dir}/trades.csv", index=False)

            # Generate custom summary
            self._save_custom_summary(
                metrics, comp_pass, vios, memo, report_dir, config
            )

            summary = f"""
Custom Backtest Completed:
- Report Directory: {report_dir}
- CAGR: {metrics.get('CAGR', 0):.2%}
- Sharpe Ratio: {metrics.get('Sharpe', 0):.2f}
- Max Drawdown: {metrics.get('MaxDrawdown', 0):.2%}
- Volatility: {metrics.get('Vol', 0):.2%}
- Compliance Passed: {comp_pass}

Configuration Used:
- Max Names: {custom_cfg.get('max_names', 'default')}
- Max Weight per Name: {custom_cfg.get('max_weight_per_name', 'default'):.1%}
- Factor Weights: {custom_cfg.get('factors', {})}

Top Holdings: {memo.split('Top names: ')[1].split('.')[0] if 'Top names:' in memo else 'N/A'}
"""
            return summary
        except Exception as e:
            return f"Error running custom backtest: {str(e)}"

    def _save_custom_summary(self, metrics, comp_pass, vios, memo, report_dir, config):
        from datetime import datetime

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(f"{report_dir}/summary.md", "w", encoding="utf-8") as f:
            f.write("# 🎯 Custom Portfolio — Backtest Summary\n\n")
            f.write(
                f"> **AI-Generated Portfolio Analysis** | *Generated on: {current_date}*\n\n"
            )
            f.write("---\n\n")

            # Configuration used
            f.write("## ⚙️ Portfolio Configuration\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Max Names | {config.get('max_names', 'default')} |\n")
            f.write(
                f"| Max Weight per Name | {config.get('max_weight_per_name', 'default'):.1%} |\n"
            )
            f.write(
                f"| Max Weight per Sector | {config.get('max_weight_per_sector', 'default'):.1%} |\n"
            )
            f.write("\n**Factor Weights:**\n")
            for factor, weight in config.get("factors", {}).items():
                f.write(f"- {factor}: {weight['weight']:.1%}\n")

            f.write("\n---\n\n")

            # Performance metrics
            f.write("## 📊 Performance Metrics\n\n")
            f.write("| Metric | Value | Status |\n")
            f.write("|--------|-------|--------|\n")

            def get_status(metric, value):
                if metric == "CAGR":
                    return (
                        "🟢 Good"
                        if value > 0.05
                        else "🟡 Moderate" if value > 0.02 else "🔴 Poor"
                    )
                elif metric == "Sharpe":
                    return (
                        "🟢 Good"
                        if value > 0.8
                        else "🟡 Moderate" if value > 0.5 else "🔴 Poor"
                    )
                elif metric == "MaxDrawdown":
                    return (
                        "🟢 Good"
                        if value > -0.10
                        else "🟡 Acceptable" if value > -0.20 else "🔴 High"
                    )
                else:
                    return "🟡 Moderate"

            for k, v in metrics.items():
                status = get_status(k, v)
                if k in ["CAGR", "MaxDrawdown"]:
                    formatted_value = f"{v:.2%}"
                else:
                    formatted_value = f"{v:.2f}"
                f.write(f"| **{k}** | {formatted_value} | {status} |\n")

            f.write("\n---\n\n")

            # Compliance
            f.write("## ⚠️ Compliance Status\n\n")
            if comp_pass:
                f.write("✅ **All compliance checks passed**\n")
            else:
                f.write("❌ **Compliance violations detected:**\n")
                for v in vios:
                    f.write(f"- {v}\n")

            f.write("\n---\n\n")

            # Investment rationale
            f.write("## 💡 Investment Rationale\n\n")
            f.write(f"{memo}\n")

            f.write("\n---\n\n")
            f.write("*Report generated by CrewAI Custom Portfolio System*\n")

    def _load_config(self) -> Dict[str, Any]:
        with open("config/constraints.yaml", "r") as f:
            return yaml.safe_load(f)

    async def _arun(self, portfolio_config: str) -> str:
        return self._run(portfolio_config)
