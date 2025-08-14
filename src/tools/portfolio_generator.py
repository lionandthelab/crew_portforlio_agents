import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import yaml

from src.utils import (
    _cov_shrinkage,
    _diversified_inv_vol_weights,
    _apply_name_cap,
    _apply_sector_caps,
    _winsorize_zscore,
    _compute_factor_table,
    _choose_price_column,
)

from .pipeline_tool_impl import run_pipeline, load_cfg

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """포트폴리오 설정"""

    name: str
    max_names: int
    max_weight_per_name: float
    factors: Dict[str, Dict[str, float]]
    sector_caps: Dict[str, float]
    description: str = ""


class PortfolioGenerator:
    """포트폴리오 생성기"""

    def __init__(self, base_config_path: str = "config/constraints.yaml"):
        self.base_config = load_cfg()
        self.base_config_path = base_config_path
        self.portfolios = []
        self.results = []

    def create_portfolio_configs(
        self, num_portfolios: int = 5
    ) -> List[PortfolioConfig]:
        """다양한 포트폴리오 설정 생성"""
        configs = []

        # 기본 설정
        base_factors = {
            "momentum_252d": {"weight": 0.4},
            "value_pe_inv": {"weight": 0.2},
            "quality_roa": {"weight": 0.2},
            "low_vol_63d": {"weight": 0.2},
        }

        base_sector_caps = {
            "Technology": 0.4,
            "Financials": 0.35,
            "Energy": 0.35,
            "Health Care": 0.35,
            "Consumer Discretionary": 0.35,
        }

        # 포트폴리오 1: 기본 설정
        configs.append(
            PortfolioConfig(
                name="Balanced Portfolio",
                max_names=12,
                max_weight_per_name=0.15,
                factors=base_factors,
                sector_caps=base_sector_caps,
                description="균형잡힌 포트폴리오 - 모든 팩터를 균등하게 적용",
            )
        )

        # 포트폴리오 2: 모멘텀 중심
        momentum_factors = {
            "momentum_252d": {"weight": 0.6},
            "value_pe_inv": {"weight": 0.1},
            "quality_roa": {"weight": 0.1},
            "low_vol_63d": {"weight": 0.2},
        }
        configs.append(
            PortfolioConfig(
                name="Momentum Portfolio",
                max_names=15,
                max_weight_per_name=0.12,
                factors=momentum_factors,
                sector_caps=base_sector_caps,
                description="모멘텀 중심 포트폴리오 - 성장주에 집중",
            )
        )

        # 포트폴리오 3: 가치 중심
        value_factors = {
            "momentum_252d": {"weight": 0.2},
            "value_pe_inv": {"weight": 0.5},
            "quality_roa": {"weight": 0.2},
            "low_vol_63d": {"weight": 0.1},
        }
        configs.append(
            PortfolioConfig(
                name="Value Portfolio",
                max_names=10,
                max_weight_per_name=0.18,
                factors=value_factors,
                sector_caps=base_sector_caps,
                description="가치 중심 포트폴리오 - 저평가 주식에 집중",
            )
        )

        # 포트폴리오 4: 저변동성 중심
        low_vol_factors = {
            "momentum_252d": {"weight": 0.2},
            "value_pe_inv": {"weight": 0.2},
            "quality_roa": {"weight": 0.2},
            "low_vol_63d": {"weight": 0.4},
        }
        configs.append(
            PortfolioConfig(
                name="Low Volatility Portfolio",
                max_names=8,
                max_weight_per_name=0.20,
                factors=low_vol_factors,
                sector_caps=base_sector_caps,
                description="저변동성 중심 포트폴리오 - 안정성에 집중",
            )
        )

        # 포트폴리오 5: 품질 중심
        quality_factors = {
            "momentum_252d": {"weight": 0.2},
            "value_pe_inv": {"weight": 0.1},
            "quality_roa": {"weight": 0.5},
            "low_vol_63d": {"weight": 0.2},
        }
        configs.append(
            PortfolioConfig(
                name="Quality Portfolio",
                max_names=12,
                max_weight_per_name=0.15,
                factors=quality_factors,
                sector_caps=base_sector_caps,
                description="품질 중심 포트폴리오 - 고ROA 기업에 집중",
            )
        )

        return configs[:num_portfolios]

    def save_portfolio_config(self, config: PortfolioConfig, portfolio_dir: str):
        """포트폴리오 설정을 파일로 저장"""
        config_dict = asdict(config)

        # YAML 파일로 저장
        config_path = os.path.join(portfolio_dir, "portfolio_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # JSON 파일로도 저장 (읽기 쉬움)
        json_path = os.path.join(portfolio_dir, "portfolio_config.json")
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def run_portfolio_test(
        self,
        config: PortfolioConfig,
        start_date: str,
        end_date: str,
        use_real_data: bool = True,
    ) -> Dict[str, Any]:
        """개별 포트폴리오 테스트 실행"""
        try:
            logger.info(f"Testing portfolio: {config.name}")

            # 포트폴리오별 디렉토리 생성
            portfolio_dir = f"reports/portfolio_{config.name.replace(' ', '_')}"
            os.makedirs(portfolio_dir, exist_ok=True)

            # 설정 파일 저장
            self.save_portfolio_config(config, portfolio_dir)

            # 임시 설정 파일 생성
            temp_config = self.base_config.copy()
            temp_config.update(
                {
                    "max_names": config.max_names,
                    "max_weight_per_name": config.max_weight_per_name,
                    "factors": config.factors,
                    "position_limits": {"sector_caps": config.sector_caps},
                }
            )

            # 임시 설정 파일 저장
            temp_config_path = os.path.join(portfolio_dir, "temp_constraints.yaml")
            with open(temp_config_path, "w") as f:
                yaml.dump(temp_config, f, default_flow_style=False)

            # 원본 설정 백업
            original_config_path = "config/constraints.yaml"
            backup_config_path = "config/constraints_backup.yaml"
            if os.path.exists(original_config_path):
                import shutil

                shutil.copy2(original_config_path, backup_config_path)

            # 임시 설정으로 교체
            shutil.copy2(temp_config_path, original_config_path)

            try:
                # 파이프라인 실행
                result = run_pipeline(start_date, end_date, use_real_data)

                # 결과 수집
                portfolio_result = {
                    "name": config.name,
                    "description": config.description,
                    "config": asdict(config),
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_dir": portfolio_dir,
                }

                # 성과 지표 수집
                try:
                    summary_path = os.path.join(portfolio_dir, "summary.md")
                    if os.path.exists(summary_path):
                        with open(summary_path, "r") as f:
                            summary_content = f.read()
                        portfolio_result["summary"] = summary_content

                        # 성과 지표 파싱
                        metrics = self.parse_performance_metrics(summary_content)
                        portfolio_result["metrics"] = metrics
                except Exception as e:
                    logger.warning(f"Failed to parse summary for {config.name}: {e}")

                logger.info(f"Portfolio {config.name} test completed successfully")
                return portfolio_result

            finally:
                # 원본 설정 복원
                if os.path.exists(backup_config_path):
                    shutil.copy2(backup_config_path, original_config_path)
                    os.remove(backup_config_path)
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

        except Exception as e:
            logger.error(f"Portfolio {config.name} test failed: {e}")
            return {
                "name": config.name,
                "description": config.description,
                "config": asdict(config),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def parse_performance_metrics(self, summary_content: str) -> Dict[str, float]:
        """성과 지표 파싱"""
        metrics = {}

        try:
            # 성과 지표 섹션 찾기
            lines = summary_content.split("\n")
            in_metrics_section = False

            for line in lines:
                if "## Performance Metrics" in line:
                    in_metrics_section = True
                    continue
                elif line.startswith("##") and in_metrics_section:
                    break

                if in_metrics_section and "**" in line and ":" in line:
                    # 지표 파싱
                    parts = line.split("**")
                    if len(parts) >= 3:
                        metric_name = parts[1].strip()
                        value_part = parts[2].split(":")[1].strip()

                        # 퍼센트 기호 제거하고 숫자만 추출
                        value_str = value_part.replace("%", "").replace(",", "")

                        try:
                            if "%" in value_part:
                                # 퍼센트 값은 소수로 변환
                                metrics[metric_name] = float(value_str) / 100
                            else:
                                metrics[metric_name] = float(value_str)
                        except ValueError:
                            continue

        except Exception as e:
            logger.warning(f"Failed to parse metrics: {e}")

        return metrics

    def run_multiple_portfolio_tests(
        self,
        start_date: str,
        end_date: str,
        num_portfolios: int = 5,
        use_real_data: bool = True,
    ) -> List[Dict[str, Any]]:
        """여러 포트폴리오 테스트 실행"""
        logger.info(f"Starting multiple portfolio tests: {num_portfolios} portfolios")

        # 포트폴리오 설정 생성
        configs = self.create_portfolio_configs(num_portfolios)

        results = []
        for config in configs:
            result = self.run_portfolio_test(
                config, start_date, end_date, use_real_data
            )
            results.append(result)

        self.results = results
        return results

    def generate_comparison_report(self, output_dir: str = "reports"):
        """포트폴리오 비교 리포트 생성"""
        if not self.results:
            logger.warning("No results to compare")
            return

        # 성과 지표 비교표 생성
        comparison_data = []

        for result in self.results:
            if "metrics" in result:
                row = {
                    "Portfolio": result["name"],
                    "Description": result["description"],
                }
                row.update(result["metrics"])
                comparison_data.append(row)

        if comparison_data:
            # DataFrame으로 변환
            df = pd.DataFrame(comparison_data)

            # CSV 파일로 저장
            csv_path = os.path.join(output_dir, "portfolio_comparison.csv")
            df.to_csv(csv_path, index=False)

            # 마크다운 테이블 생성
            md_path = os.path.join(output_dir, "portfolio_comparison.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# Portfolio Performance Comparison\n\n")
                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                # 성과 지표 테이블
                f.write("## Performance Metrics Comparison\n\n")

                # 마크다운 테이블 헤더
                if comparison_data:
                    headers = list(comparison_data[0].keys())
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

                    # 데이터 행
                    for row in comparison_data:
                        formatted_row = []
                        for key in headers:
                            value = row[key]
                            if isinstance(value, float):
                                if "Return" in key or "Drawdown" in key:
                                    formatted_row.append(f"{value:.2%}")
                                else:
                                    formatted_row.append(f"{value:.3f}")
                            else:
                                formatted_row.append(str(value))
                        f.write("| " + " | ".join(formatted_row) + " |\n")

                f.write("\n\n")

                # 포트폴리오별 상세 정보
                f.write("## Portfolio Details\n\n")
                for result in self.results:
                    f.write(f"### {result['name']}\n\n")
                    f.write(f"**Description**: {result['description']}\n\n")

                    if "metrics" in result:
                        f.write("**Key Metrics**:\n")
                        for metric, value in result["metrics"].items():
                            if "Return" in metric or "Drawdown" in metric:
                                f.write(f"- {metric}: {value:.2%}\n")
                            else:
                                f.write(f"- {metric}: {value:.3f}\n")

                    if "portfolio_dir" in result:
                        f.write(
                            f"\n**Detailed Report**: [View Report]({result['portfolio_dir']}/summary.md)\n"
                        )

                    f.write("\n---\n\n")

            logger.info(f"Comparison report generated: {md_path}")
            logger.info(f"CSV data exported: {csv_path}")

        # 전체 결과 요약 JSON
        summary_path = os.path.join(output_dir, "portfolio_test_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Test summary saved: {summary_path}")


def run_portfolio_comparison(
    start_date: str = "2023-01-01",
    end_date: str = "2025-07-31",
    num_portfolios: int = 5,
    use_real_data: bool = True,
):
    """포트폴리오 비교 실행 함수"""
    generator = PortfolioGenerator()

    # 여러 포트폴리오 테스트 실행
    results = generator.run_multiple_portfolio_tests(
        start_date, end_date, num_portfolios, use_real_data
    )

    # 비교 리포트 생성
    generator.generate_comparison_report()

    return results


def generate_portfolios(
    data_dir: str, num_portfolios: int = 5, portfolio_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    다양한 포트폴리오를 생성합니다. (전문화 버전)
    - prices.csv: columns ⊇ [date, ticker, adj_close|close|price, (optional) volume]
    - fundamentals.csv (optional): columns ⊇ [ticker, pe, pb, roa, roe, sector]
    """
    logger.info(f"Generating {num_portfolios} portfolios...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Portfolio types: {portfolio_types}")

    portfolio_configs = {
        "growth": {
            "name": "Growth Portfolio",
            "type": "growth",
            "objective": "장기 성장을 추구하는 포트폴리오",
            "strategy": "모멘텀과 성장성을 중심으로 구성",
            "composition": "고성장 기업 위주로 구성",
            "factors": {
                "momentum_252d": {"weight": 0.6},
                "value_pe_inv": {"weight": 0.1},
                "quality_roa": {"weight": 0.2},
                "low_vol_63d": {"weight": 0.1},
            },
            "max_names": 15,
            "max_weight_per_name": 0.12,
        },
        "value": {
            "name": "Value Portfolio",
            "type": "value",
            "objective": "저평가 주식에 투자하여 가치 발견",
            "strategy": "P/E, P/B 등 가치 지표 중심",
            "composition": "저평가 기업 위주로 구성",
            "factors": {
                "momentum_252d": {"weight": 0.2},
                "value_pe_inv": {"weight": 0.5},
                "quality_roa": {"weight": 0.2},
                "low_vol_63d": {"weight": 0.1},
            },
            "max_names": 10,
            "max_weight_per_name": 0.18,
        },
        "balanced": {
            "name": "Balanced Portfolio",
            "type": "balanced",
            "objective": "위험과 수익의 균형을 추구",
            "strategy": "모든 팩터를 균등하게 적용",
            "composition": "다양한 스타일의 기업으로 구성",
            "factors": {
                "momentum_252d": {"weight": 0.4},
                "value_pe_inv": {"weight": 0.2},
                "quality_roa": {"weight": 0.2},
                "low_vol_63d": {"weight": 0.2},
            },
            "max_names": 12,
            "max_weight_per_name": 0.15,
        },
        "low_vol": {
            "name": "Low Volatility Portfolio",
            "type": "low_vol",
            "objective": "안정적인 수익을 추구",
            "strategy": "저변동성 주식 중심으로 구성",
            "composition": "안정적인 기업 위주로 구성",
            "factors": {
                "momentum_252d": {"weight": 0.2},
                "value_pe_inv": {"weight": 0.2},
                "quality_roa": {"weight": 0.2},
                "low_vol_63d": {"weight": 0.4},
            },
            "max_names": 8,
            "max_weight_per_name": 0.20,
        },
        "quality": {
            "name": "Quality Portfolio",
            "type": "quality",
            "objective": "고품질 기업에 투자",
            "strategy": "ROA, ROE 등 품질 지표 중심",
            "composition": "고ROA 기업 위주로 구성",
            "factors": {
                "momentum_252d": {"weight": 0.2},
                "value_pe_inv": {"weight": 0.1},
                "quality_roa": {"weight": 0.5},
                "low_vol_63d": {"weight": 0.2},
            },
            "max_names": 12,
            "max_weight_per_name": 0.15,
        },
    }

    if portfolio_types is None:
        available_types = list(portfolio_configs.keys())
    else:
        available_types = [t for t in portfolio_types if t in portfolio_configs]
    if not available_types:
        logger.warning("No valid portfolio types specified, using all types")
        available_types = list(portfolio_configs.keys())

    try:
        prices_file = os.path.join(data_dir, "prices.csv")
        fundamentals_file = os.path.join(data_dir, "fundamentals.csv")
        if not os.path.exists(prices_file):
            raise FileNotFoundError(f"Price data not found: {prices_file}")

        prices = pd.read_csv(prices_file)
        fundamentals = (
            pd.read_csv(fundamentals_file)
            if os.path.exists(fundamentals_file)
            else None
        )

        logger.info(f"Loaded price data: {prices.shape}")
        if fundamentals is not None:
            logger.info(f"Loaded fundamental data: {fundamentals.shape}")

        available_symbols = prices["ticker"].unique().tolist()
        logger.info(f"Available symbols: {len(available_symbols)}")

        portfolios = []
        for i, portfolio_type in enumerate(available_types[:num_portfolios]):
            config = portfolio_configs[portfolio_type]

            # 가중치 생성 (전문화 버전)
            weights_df = generate_portfolio_weights(
                symbols=available_symbols,
                config=config,
                prices=prices,
                fundamentals=fundamentals,
            )

            portfolio = {
                "name": config["name"],
                "type": config["type"],
                "objective": config["objective"],
                "strategy": config["strategy"],
                "composition": config["composition"],
                "weights": weights_df,
                "parameters": {
                    "max_names": config["max_names"],
                    "max_weight_per_name": config["max_weight_per_name"],
                    "factors": config["factors"],
                },
            }
            portfolios.append(portfolio)
            logger.info(f"Generated portfolio: {portfolio['name']}")

        logger.info(f"Successfully generated {len(portfolios)} portfolios")
        return portfolios

    except Exception as e:
        logger.error(f"Portfolio generation failed: {e}")
        raise


def generate_portfolio_weights(
    symbols: List[str],
    config: Dict[str, Any],
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    팩터 기반 선정 + 수축 공분산 + 리스크 기반 가중치 + 캡 제약 적용
    Returns: DataFrame[symbol, weight] (합=1)
    """
    # 1) 팩터 테이블 생성 & 멀티팩터 점수
    factors_cfg = config["factors"]
    fac_tbl = _compute_factor_table(prices, fundamentals, factors_cfg)
    feature_cols = [c for c in factors_cfg.keys() if c in fac_tbl.columns]
    if not feature_cols:
        raise ValueError("사용 가능한 팩터가 없습니다. (prices/fundamentals 확인)")

    weights_dict = {f: factors_cfg[f]["weight"] for f in feature_cols}
    # 가중합 점수
    score = sum(fac_tbl[f] * weights_dict[f] for f in feature_cols)
    fac_tbl["__score__"] = score

    # 심볼 필터(실제로 존재하는 가격 데이터)
    universe = pd.Index(symbols)
    fac_tbl = fac_tbl.loc[fac_tbl.index.intersection(universe)]

    # 2) 상위 종목 선택
    max_names = int(config["max_names"])
    fac_tbl = fac_tbl.sort_values("__score__", ascending=False)
    sel = fac_tbl.head(max_names).index.tolist()

    # 3) 팩터 점수 기반 초기 가중치 생성
    # 팩터 점수를 양수로 변환하고 정규화
    factor_scores = fac_tbl.loc[sel, "__score__"]
    factor_scores = factor_scores - factor_scores.min() + 1e-6  # 양수로 변환
    w = factor_scores / factor_scores.sum()

    # 4) 리스크 조정 (선택적)
    # 공분산 추정 및 리스크 조정을 적용할지 결정
    apply_risk_adjustment = True  # 이 값을 False로 하면 순수 팩터 기반 가중치 사용

    if apply_risk_adjustment:
        price_col = _choose_price_column(prices)
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])

        # 중복된 데이터 제거 (같은 날짜, 같은 ticker에 대해 마지막 값만 유지)
        prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")

        px = prices.pivot(index="date", columns="ticker", values=price_col).sort_index()
        rets = px[sel].pct_change().dropna(how="all").dropna(axis=0)
        # 결측 많은 종목 제거
        valid_cols = [
            c for c in sel if c in rets.columns and rets[c].notna().sum() >= 40
        ]
        rets = rets[valid_cols].dropna()

        if rets.shape[1] > 1:  # 충분한 데이터가 있을 때만 리스크 조정
            cov = np.cov(rets.values, rowvar=False)
            cov = _cov_shrinkage(cov, lam=None)
            risk_weights = _diversified_inv_vol_weights(cov)

            # 팩터 가중치와 리스크 가중치를 결합 (팩터 70%, 리스크 30%)
            risk_w = pd.Series(0.0, index=sel)
            risk_w.loc[rets.columns] = risk_weights
            risk_w = (risk_w / risk_w.sum()).fillna(0.0)

            # 팩터와 리스크 가중치 결합
            w = 0.7 * w + 0.3 * risk_w
            w = w / w.sum()

    # 5) 종목 상한(cap) 적용
    cap = float(config.get("max_weight_per_name", 1.0))
    raw_pref = fac_tbl.loc[w.index, "__score__"].clip(lower=0.0)
    w = _apply_name_cap(w, cap=cap, raw_pref=raw_pref)

    # 6) 섹터 캡 적용(가능할 때만)
    sectors = None
    if "__sector__" in fac_tbl.columns:
        sectors = fac_tbl.loc[w.index, "__sector__"]
    sector_caps = config.get("sector_caps", None)
    w = _apply_sector_caps(
        w, sectors=sectors, sector_caps=sector_caps, raw_pref=raw_pref
    )

    # 7) DataFrame 반환 (정렬)
    w = w.clip(lower=0.0)
    if w.sum() == 0:
        w[:] = 1.0 / len(w)
    w = w / w.sum()
    weights_df = pd.DataFrame({"symbol": w.index, "weight": w.values}).sort_values(
        "weight", ascending=False
    )
    return weights_df
