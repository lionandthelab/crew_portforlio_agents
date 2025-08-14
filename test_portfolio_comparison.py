#!/usr/bin/env python3
"""
포트폴리오 비교 테스트 스크립트
"""

import sys
import os
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tools.portfolio_generator import run_portfolio_comparison

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """메인 테스트 함수"""
    logger.info("Starting portfolio comparison test...")

    try:
        # 포트폴리오 비교 실행 (3개 포트폴리오, mock 데이터 사용)
        results = run_portfolio_comparison(
            start_date="2023-01-01",
            end_date="2025-07-31",
            num_portfolios=3,
            use_real_data=False,  # mock 데이터 사용
        )

        logger.info(f"Portfolio comparison completed successfully!")
        logger.info(f"Generated {len(results)} portfolio reports")

        # 결과 확인
        for result in results:
            if "error" in result:
                logger.error(f"Portfolio {result['name']} failed: {result['error']}")
            else:
                logger.info(f"Portfolio {result['name']} completed successfully")
                if "metrics" in result:
                    logger.info(
                        f"  - Sharpe Ratio: {result['metrics'].get('Sharpe_Ratio', 'N/A')}"
                    )
                    logger.info(
                        f"  - Total Return: {result['metrics'].get('Total_Return', 'N/A')}"
                    )

        logger.info("Check reports/portfolio_comparison.md for detailed comparison")
        return 0

    except Exception as e:
        logger.error(f"Portfolio comparison test failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


