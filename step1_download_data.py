#!/usr/bin/env python3
"""
Step 1: Data Download
데이터를 다운로드 받는 단계

사용법:
python step1_download_data.py --start 2020-01-01 --end 2025-07-31 --output data/downloaded/
"""

import argparse
import os
import logging
from datetime import datetime
from src.tools.market_data import download_market_data

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_output_directory(output_dir):
    """출력 디렉토리를 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Download market data")
    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default="2025-07-31", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/downloaded/",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols to download",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of real market data",
    )

    args = parser.parse_args()

    logger.info(f"Starting data download...")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Use mock data: {args.use_mock}")

    # 출력 디렉토리 생성
    create_output_directory(args.output)

    try:
        # 데이터 다운로드 실행
        symbols = args.symbols.split(",") if args.symbols else None
        result = download_market_data(
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output,
            symbols=symbols,
            use_mock_data=args.use_mock,
        )

        logger.info("Data download completed successfully!")
        logger.info(f"Downloaded files: {result}")

        # 다운로드 완료 메타데이터 저장
        metadata = {
            "download_date": datetime.now().isoformat(),
            "start_date": args.start,
            "end_date": args.end,
            "output_directory": args.output,
            "symbols": symbols,
            "use_mock_data": args.use_mock,
            "files_downloaded": result,
        }

        metadata_file = os.path.join(args.output, "download_metadata.json")
        import json

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Download metadata saved to: {metadata_file}")

    except Exception as e:
        logger.error(f"Data download failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
