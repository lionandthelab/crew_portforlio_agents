#!/usr/bin/env python3
"""
Step 3: Backtesting
포트폴리오를 가지고 특정 기간에 대해 백테스팅하는 단계

사용법:
python step3_backtest.py --portfolio portfolios/portfolio_01_growth --start 2020-01-01 --end 2025-07-31
"""

import argparse
import os
import logging
import json
import shutil
from datetime import datetime
from src.tools.pipeline_tool_impl import run_backtest
from src.tools.portfolio_tools import generate_equity_curve_plot

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_portfolio_info(portfolio_path):
    """포트폴리오 정보를 로드합니다."""
    metadata_file = os.path.join(portfolio_path, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    else:
        # 기본 정보 생성
        portfolio_name = os.path.basename(portfolio_path)
        return {
            "name": portfolio_name,
            "type": "unknown",
            "objective": "N/A",
            "strategy": "N/A",
        }


def create_backtest_report_directory(reports_dir, portfolio_name, start_date, end_date):
    """백테스트 결과를 저장할 디렉토리를 생성합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{portfolio_name}_{start_date}_{end_date}_{timestamp}"
    report_dir = os.path.join(reports_dir, report_name)
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def copy_portfolio_files(portfolio_path, report_dir):
    """포트폴리오 파일들을 백테스트 결과 디렉토리로 복사합니다."""
    portfolio_files = ["weights.csv", "portfolio_description.md", "metadata.json"]

    for file_name in portfolio_files:
        src_file = os.path.join(portfolio_path, file_name)
        if os.path.exists(src_file):
            dst_file = os.path.join(report_dir, f"portfolio_{file_name}")
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {file_name} to report directory")


def save_backtest_metadata(
    report_dir, portfolio_info, start_date, end_date, backtest_params
):
    """백테스트 메타데이터를 저장합니다."""
    metadata = {
        "backtest_date": datetime.now().isoformat(),
        "portfolio_info": portfolio_info,
        "backtest_period": {"start_date": start_date, "end_date": end_date},
        "backtest_parameters": backtest_params,
    }

    metadata_file = os.path.join(report_dir, "backtest_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Backtest metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Run backtest")
    parser.add_argument(
        "--portfolio", type=str, required=True, help="Path to portfolio directory"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default="2025-07-31", help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/downloaded/",
        help="Directory containing market data",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/",
        help="Directory for backtest reports",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="quarterly",
        help="Rebalancing frequency (daily, weekly, monthly, quarterly)",
    )
    parser.add_argument(
        "--transaction-costs",
        type=float,
        default=0.001,
        help="Transaction costs as percentage",
    )

    args = parser.parse_args()

    logger.info(f"Starting backtest...")
    logger.info(f"Portfolio: {args.portfolio}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Reports directory: {args.reports_dir}")

    # 포트폴리오 정보 로드
    portfolio_info = load_portfolio_info(args.portfolio)
    portfolio_name = portfolio_info["name"]

    # 백테스트 결과 디렉토리 생성
    report_dir = create_backtest_report_directory(
        args.reports_dir, portfolio_name, args.start, args.end
    )

    # 포트폴리오 파일들 복사
    copy_portfolio_files(args.portfolio, report_dir)

    try:
        # 백테스트 파라미터 설정
        backtest_params = {
            "rebalance_frequency": args.rebalance_frequency,
            "transaction_costs": args.transaction_costs,
            "data_directory": args.data_dir,
        }

        # 백테스트 실행
        logger.info("Running backtest...")
        results = run_backtest(
            portfolio_path=args.portfolio,
            start_date=args.start,
            end_date=args.end,
            **backtest_params,
        )

        # 백테스트 결과 저장
        if results:
            # 수익률 곡선 저장
            if "equity_curve" in results:
                equity_file = os.path.join(report_dir, "equity_curve.csv")
                results["equity_curve"].to_csv(equity_file, index=True)
                logger.info(f"Equity curve saved to: {equity_file}")

                # 수익률 곡선 그래프 생성
                plot_file = os.path.join(report_dir, "equity_curve_plot.png")
                generate_equity_curve_plot(
                    equity_curve=results["equity_curve"],
                    portfolio_name=portfolio_name,
                    output_path=plot_file,
                )
                logger.info(f"Equity curve plot saved to: {plot_file}")

            # 거래 내역 저장
            if "trades" in results:
                trades_file = os.path.join(report_dir, "trades.csv")
                results["trades"].to_csv(trades_file, index=False)
                logger.info(f"Trades saved to: {trades_file}")

            # 성과 지표 저장
            if "metrics" in results:
                metrics_file = os.path.join(report_dir, "performance_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(results["metrics"], f, indent=2)
                logger.info(f"Performance metrics saved to: {metrics_file}")

            # 요약 보고서 생성
            summary_file = os.path.join(report_dir, "backtest_summary.md")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"# Backtest Summary: {portfolio_name}\n\n")
                f.write(f"## Portfolio Information\n")
                f.write(f"- **Name**: {portfolio_info['name']}\n")
                f.write(f"- **Type**: {portfolio_info['type']}\n")
                f.write(f"- **Objective**: {portfolio_info['objective']}\n\n")

                f.write(f"## Backtest Period\n")
                f.write(f"- **Start Date**: {args.start}\n")
                f.write(f"- **End Date**: {args.end}\n")
                f.write(f"- **Rebalance Frequency**: {args.rebalance_frequency}\n")
                f.write(f"- **Transaction Costs**: {args.transaction_costs:.3f}%\n\n")

                if "metrics" in results:
                    f.write(f"## Performance Metrics\n")
                    metrics = results["metrics"]
                    f.write(
                        f"- **Total Return**: {metrics.get('total_return', 0) * 100:.2f}%\n"
                    )
                    f.write(
                        f"- **Annualized Return**: {metrics.get('annualized_return', 0) * 100:.2f}%\n"
                    )
                    f.write(
                        f"- **Volatility**: {metrics.get('volatility', 0) * 100:.2f}%\n"
                    )
                    f.write(
                        f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}\n"
                    )
                    f.write(
                        f"- **Max Drawdown**: {metrics.get('max_drawdown', 0) * 100:.2f}%\n"
                    )
                    f.write(
                        f"- **Win Rate**: {metrics.get('win_rate', 0) * 100:.2f}%\n\n"
                    )

                f.write(f"## Files Generated\n")
                f.write(f"- `equity_curve.csv`: Daily portfolio values\n")
                f.write(
                    f"- `equity_curve_plot.png`: Portfolio equity curve visualization\n"
                )
                f.write(f"- `trades.csv`: Trade execution history\n")
                f.write(f"- `performance_metrics.json`: Detailed performance metrics\n")
                f.write(f"- `portfolio_weights.csv`: Original portfolio weights\n")
                f.write(
                    f"- `portfolio_description.md`: Portfolio strategy description\n"
                )

            logger.info(f"Backtest summary saved to: {summary_file}")

        # 백테스트 메타데이터 저장
        save_backtest_metadata(
            report_dir, portfolio_info, args.start, args.end, backtest_params
        )

        logger.info(f"Backtest completed successfully!")
        logger.info(f"Results saved to: {report_dir}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
