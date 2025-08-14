#!/usr/bin/env python3
"""
Portfolio Analysis Workflow
포트폴리오 분석 전체 워크플로우를 실행하는 스크립트

사용법:
python run_workflow.py --start 2020-01-01 --end 2025-07-31 --portfolio-types growth,value,balanced
"""

import argparse
import os
import logging
import json
import subprocess
import sys
from datetime import datetime
from src.tools.portfolio_tools import (
    generate_performance_comparison_table,
    save_performance_comparison_table,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_step(step_name: str, command: list) -> bool:
    """단계별 실행 함수"""
    logger.info(f"Running {step_name}...")
    logger.info(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"{step_name} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} failed with return code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Portfolio Analysis Workflow")
    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default="2025-07-31", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--portfolio-types",
        type=str,
        default="",
        help="Comma-separated list of portfolio types",
    )
    parser.add_argument(
        "--num-portfolios", type=int, default=5, help="Number of portfolios to generate"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of real market data",
    )
    parser.add_argument(
        "--skip-step1", action="store_true", help="Skip data download step"
    )
    parser.add_argument(
        "--skip-step2", action="store_true", help="Skip portfolio generation step"
    )
    parser.add_argument(
        "--backtest-portfolio",
        type=str,
        default="",
        help="Specific portfolio to backtest (if empty, backtest all)",
    )

    args = parser.parse_args()

    logger.info("Starting Portfolio Analysis Workflow")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Portfolio types: {args.portfolio_types}")
    logger.info(f"Number of portfolios: {args.num_portfolios}")
    logger.info(f"Use mock data: {args.use_mock}")

    # 워크플로우 시작 시간 기록
    start_time = datetime.now()

    # Step 1: 데이터 다운로드
    if not args.skip_step1:
        logger.info("=" * 50)
        logger.info("STEP 1: Data Download")
        logger.info("=" * 50)

        step1_cmd = [
            sys.executable,
            "step1_download_data.py",
            "--start",
            args.start,
            "--end",
            args.end,
            "--output",
            "data/downloaded/",
        ]

        if args.use_mock:
            step1_cmd.append("--use-mock")

        if not run_step("Step 1 (Data Download)", step1_cmd):
            logger.error("Step 1 failed. Exiting workflow.")
            return False

    # Step 2: 포트폴리오 생성
    if not args.skip_step2:
        logger.info("=" * 50)
        logger.info("STEP 2: Portfolio Generation")
        logger.info("=" * 50)

        step2_cmd = [
            sys.executable,
            "step2_generate_portfolios.py",
            "--data-dir",
            "data/downloaded/",
            "--output",
            "portfolios/",
            "--num-portfolios",
            str(args.num_portfolios),
        ]

        if args.portfolio_types:
            step2_cmd.extend(["--portfolio-types", args.portfolio_types])

        if not run_step("Step 2 (Portfolio Generation)", step2_cmd):
            logger.error("Step 2 failed. Exiting workflow.")
            return False

    # Step 3: 백테스팅
    logger.info("=" * 50)
    logger.info("STEP 3: Backtesting")
    logger.info("=" * 50)

    # 포트폴리오 목록 확인
    portfolios_dir = "portfolios"
    if not os.path.exists(portfolios_dir):
        logger.error(f"Portfolios directory not found: {portfolios_dir}")
        return False

    # 포트폴리오 목록 로드
    portfolio_list_file = os.path.join(portfolios_dir, "portfolio_list.json")
    if os.path.exists(portfolio_list_file):
        with open(portfolio_list_file, "r") as f:
            portfolio_list = json.load(f)
        portfolios = portfolio_list.get("portfolios", [])
    else:
        # 디렉토리에서 포트폴리오 찾기
        portfolios = [
            d
            for d in os.listdir(portfolios_dir)
            if os.path.isdir(os.path.join(portfolios_dir, d))
        ]

    if not portfolios:
        logger.error("No portfolios found for backtesting")
        return False

    logger.info(f"Found {len(portfolios)} portfolios for backtesting")

    # 백테스팅 실행
    successful_backtests = 0
    portfolio_results = []

    for portfolio in portfolios:
        # 특정 포트폴리오만 백테스팅하는 경우
        if args.backtest_portfolio and portfolio != args.backtest_portfolio:
            continue

        portfolio_path = os.path.join(portfolios_dir, portfolio)
        logger.info(f"Backtesting portfolio: {portfolio}")

        step3_cmd = [
            sys.executable,
            "step3_backtest.py",
            "--portfolio",
            portfolio_path,
            "--start",
            args.start,
            "--end",
            args.end,
            "--data-dir",
            "data/downloaded/",
            "--reports-dir",
            "reports/",
        ]

        if run_step(f"Step 3 (Backtest {portfolio})", step3_cmd):
            successful_backtests += 1

            # 백테스트 결과 수집
            try:
                # 최신 백테스트 리포트 찾기
                reports_dir = "reports"
                portfolio_reports = [
                    d
                    for d in os.listdir(reports_dir)
                    if d.startswith(portfolio)
                    and os.path.isdir(os.path.join(reports_dir, d))
                ]

                if portfolio_reports:
                    latest_report = max(portfolio_reports)
                    report_dir = os.path.join(reports_dir, latest_report)

                    # 포트폴리오 정보 로드
                    portfolio_info_file = os.path.join(portfolio_path, "metadata.json")
                    if os.path.exists(portfolio_info_file):
                        with open(portfolio_info_file, "r") as f:
                            portfolio_info = json.load(f)
                    else:
                        portfolio_info = {"name": portfolio, "type": "unknown"}

                    # 성과 지표 로드
                    metrics_file = os.path.join(report_dir, "performance_metrics.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)
                    else:
                        metrics = {}

                    portfolio_results.append(
                        {
                            "portfolio_info": portfolio_info,
                            "metrics": metrics,
                            "report_dir": report_dir,
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to collect results for {portfolio}: {e}")
        else:
            logger.warning(f"Backtest failed for portfolio: {portfolio}")

    # 워크플로우 완료
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 50)
    logger.info("WORKFLOW COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Total duration: {duration}")
    logger.info(f"Successful backtests: {successful_backtests}/{len(portfolios)}")

    # 결과 요약 생성
    summary = {
        "workflow_start": start_time.isoformat(),
        "workflow_end": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "parameters": {
            "start_date": args.start,
            "end_date": args.end,
            "portfolio_types": args.portfolio_types,
            "num_portfolios": args.num_portfolios,
            "use_mock_data": args.use_mock,
        },
        "results": {
            "total_portfolios": len(portfolios),
            "successful_backtests": successful_backtests,
            "failed_backtests": len(portfolios) - successful_backtests,
        },
    }

    # 성과 비교 표 생성 (성공한 백테스트가 있는 경우)
    if portfolio_results:
        try:
            comparison_df = generate_performance_comparison_table(portfolio_results)

            # 성과 비교 표를 CSV로 저장
            comparison_file = f"reports/performance_comparison_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
            save_performance_comparison_table(comparison_df, comparison_file)

            # 요약에 성과 비교 표 정보 추가
            summary["performance_comparison"] = {
                "file": comparison_file,
                "table_data": comparison_df.to_dict("records"),
            }

            logger.info(f"Performance comparison table saved to: {comparison_file}")

        except Exception as e:
            logger.warning(f"Failed to generate performance comparison table: {e}")

    # 요약 파일 저장
    summary_file = (
        f"reports/workflow_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs("reports", exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Workflow summary saved to: {summary_file}")

    if successful_backtests > 0:
        logger.info("Workflow completed successfully!")
        return True
    else:
        logger.error("No successful backtests. Workflow failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
