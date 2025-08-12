#!/usr/bin/env python3
"""
파이프라인 테스트 스크립트
"""

import sys
import os
import logging
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tools.pipeline_tool_impl import run_pipeline
from src.tools.market_data import get_cache_info, clear_cache

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_mock_data():
    """Mock 데이터로 파이프라인 테스트"""
    logger.info("=== Testing with Mock Data ===")

    try:
        result = run_pipeline(start="2020-01-01", end="2024-12-31", use_real_data=False)
        logger.info(f"Mock data test result: {result}")
        return True
    except Exception as e:
        logger.error(f"Mock data test failed: {e}")
        return False


def test_real_data():
    """실제 데이터로 파이프라인 테스트"""
    logger.info("=== Testing with Real Data ===")

    try:
        result = run_pipeline(start="2023-01-01", end="2024-12-31", use_real_data=True)
        logger.info(f"Real data test result: {result}")
        return True
    except Exception as e:
        logger.error(f"Real data test failed: {e}")
        return False


def test_cache():
    """캐시 시스템 테스트"""
    logger.info("=== Testing Cache System ===")

    try:
        # 캐시 정보 확인
        cache_info = get_cache_info()
        logger.info(f"Cache info: {cache_info}")

        # 캐시 클리어 (선택사항)
        # clear_cache()

        return True
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        return False


def main():
    """메인 테스트 함수"""
    logger.info("Starting pipeline tests...")

    # 캐시 테스트
    cache_success = test_cache()

    # Mock 데이터 테스트
    mock_success = test_mock_data()

    # 실제 데이터 테스트 (선택사항)
    real_success = test_real_data()

    # 결과 요약
    logger.info("=== Test Results ===")
    logger.info(f"Cache test: {'PASSED' if cache_success else 'FAILED'}")
    logger.info(f"Mock data test: {'PASSED' if mock_success else 'FAILED'}")
    logger.info(f"Real data test: {'PASSED' if real_success else 'FAILED'}")

    if all([cache_success, mock_success]):
        logger.info("✅ All essential tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
