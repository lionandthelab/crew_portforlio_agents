#!/usr/bin/env python3
"""
Step 2: Portfolio Generation
포트폴리오의 목적에 따라 데이터를 분석하여 포트폴리오를 구성하는 단계

사용법:
python step2_generate_portfolios.py --data-dir data/downloaded/ --output portfolios/
"""

import argparse
import os
import logging
import json
from datetime import datetime
from src.tools.portfolio_generator import generate_portfolios

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_portfolio_directory(output_dir, portfolio_name):
    """포트폴리오별 디렉토리를 생성합니다."""
    portfolio_dir = os.path.join(output_dir, portfolio_name)
    os.makedirs(portfolio_dir, exist_ok=True)
    return portfolio_dir


def save_portfolio_description(portfolio_dir, portfolio_info):
    """포트폴리오 설명을 저장합니다."""
    description_file = os.path.join(portfolio_dir, "portfolio_description.md")

    with open(description_file, "w", encoding="utf-8") as f:
        f.write(f"# {portfolio_info['name']}\n\n")
        f.write(f"## 목적\n{portfolio_info['objective']}\n\n")
        f.write(f"## 전략\n{portfolio_info['strategy']}\n\n")
        f.write(f"## 구성\n{portfolio_info['composition']}\n\n")
        f.write(f"## 생성일시\n{portfolio_info['created_at']}\n\n")
        f.write(f"## 파라미터\n")
        for key, value in portfolio_info.get("parameters", {}).items():
            f.write(f"- {key}: {value}\n")

    logger.info(f"Portfolio description saved to: {description_file}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate portfolios")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/downloaded/",
        help="Directory containing downloaded data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="portfolios/",
        help="Output directory for generated portfolios",
    )
    parser.add_argument(
        "--num-portfolios", type=int, default=5, help="Number of portfolios to generate"
    )
    parser.add_argument(
        "--portfolio-types",
        type=str,
        default="",
        help="Comma-separated list of portfolio types (e.g., 'growth,value,balanced')",
    )

    args = parser.parse_args()

    logger.info(f"Starting portfolio generation...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Number of portfolios: {args.num_portfolios}")

    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    try:
        # 포트폴리오 타입 파싱
        portfolio_types = (
            args.portfolio_types.split(",") if args.portfolio_types else None
        )

        # 포트폴리오 생성 실행
        portfolios = generate_portfolios(
            data_dir=args.data_dir,
            num_portfolios=args.num_portfolios,
            portfolio_types=portfolio_types,
        )

        logger.info(f"Generated {len(portfolios)} portfolios")

        # 각 포트폴리오별로 디렉토리 생성 및 파일 저장
        for i, portfolio in enumerate(portfolios):
            portfolio_name = f"portfolio_{i+1:02d}_{portfolio['type']}"
            portfolio_dir = create_portfolio_directory(args.output, portfolio_name)

            # 포트폴리오 설명 저장
            portfolio_info = {
                "name": portfolio_name,
                "type": portfolio["type"],
                "objective": portfolio.get("objective", "N/A"),
                "strategy": portfolio.get("strategy", "N/A"),
                "composition": portfolio.get("composition", "N/A"),
                "created_at": datetime.now().isoformat(),
                "parameters": portfolio.get("parameters", {}),
            }
            save_portfolio_description(portfolio_dir, portfolio_info)

            # 포트폴리오 가중치 저장
            weights_file = os.path.join(portfolio_dir, "weights.csv")
            portfolio["weights"].to_csv(weights_file, index=True)
            logger.info(f"Portfolio weights saved to: {weights_file}")

            # 포트폴리오 메타데이터 저장
            metadata_file = os.path.join(portfolio_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(portfolio_info, f, indent=2, ensure_ascii=False)

        # 전체 포트폴리오 목록 저장
        portfolio_list = {
            "generated_at": datetime.now().isoformat(),
            "total_portfolios": len(portfolios),
            "portfolios": [
                f"portfolio_{i+1:02d}_{p['type']}" for i, p in enumerate(portfolios)
            ],
        }

        list_file = os.path.join(args.output, "portfolio_list.json")
        with open(list_file, "w") as f:
            json.dump(portfolio_list, f, indent=2)

        logger.info(f"Portfolio list saved to: {list_file}")
        logger.info("Portfolio generation completed successfully!")

    except Exception as e:
        logger.error(f"Portfolio generation failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
