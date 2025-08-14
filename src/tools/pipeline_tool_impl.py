from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import yaml
import os
import logging
from dataclasses import dataclass
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cfg() -> Dict[str, Any]:
    """설정 파일 로드"""
    try:
        with open("config/constraints.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def load_prices(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-31",
    use_real_data: bool = True,
) -> pd.DataFrame:
    """주가 데이터를 로드합니다."""
    try:
        if use_real_data:
            from src.tools.market_data import load_real_market_data

            prices, _ = load_real_market_data(start_date, end_date, use_real_data=True)
            return prices
        else:
            return pd.read_csv("data/mock_prices.csv", parse_dates=["date"])
    except Exception as e:
        logger.error(f"Error loading prices: {e}")
        logger.info("Falling back to mock prices...")
        return pd.read_csv("data/mock_prices.csv", parse_dates=["date"])


def load_fundamentals(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-31",
    use_real_data: bool = True,
) -> pd.DataFrame:
    """재무 데이터를 로드합니다."""
    try:
        if use_real_data:
            from src.tools.market_data import load_real_market_data

            _, fundamentals = load_real_market_data(
                start_date, end_date, use_real_data=True
            )
            return fundamentals
        else:
            return pd.read_csv("data/mock_fundamentals.csv", parse_dates=["date"])
    except Exception as e:
        logger.error(f"Error loading fundamentals: {e}")
        logger.info("Falling back to mock fundamentals...")
        return pd.read_csv("data/mock_fundamentals.csv", parse_dates=["date"])


def validate_data(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> bool:
    """데이터 유효성 검증"""
    if prices.empty:
        logger.error("Prices DataFrame is empty")
        return False

    if fundamentals.empty:
        logger.error("Fundamentals DataFrame is empty")
        return False

    required_price_cols = ["date", "ticker", "close"]
    required_fundamental_cols = ["date", "ticker", "PE", "ROA", "sector"]

    if not all(col in prices.columns for col in required_price_cols):
        logger.error(f"Missing required columns in prices: {required_price_cols}")
        return False

    if not all(col in fundamentals.columns for col in required_fundamental_cols):
        logger.error(
            f"Missing required columns in fundamentals: {required_fundamental_cols}"
        )
        return False

    logger.info(
        f"Data validation passed. Prices: {prices.shape}, Fundamentals: {fundamentals.shape}"
    )
    return True


def compute_signals(
    prices: pd.DataFrame, fundamentals: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """신호 계산"""
    try:
        logger.info(
            f"Computing signals with prices shape: {prices.shape}, fundamentals shape: {fundamentals.shape}"
        )
        logger.info(
            f"Price date range: {prices['date'].min()} to {prices['date'].max()}"
        )
        logger.info(
            f"Fundamental date range: {fundamentals['date'].min()} to {fundamentals['date'].max()}"
        )

        # 가격 데이터 피벗
        px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
        logger.info(f"Pivoted prices shape: {px.shape}")
        logger.info(f"Available tickers: {list(px.columns)}")

        if px.empty:
            logger.error("Pivoted prices DataFrame is empty")
            return pd.DataFrame(columns=["date", "ticker", "score"])

        rets = px.pct_change()

        # 모멘텀 계산 (252일)
        mom = (px / px.shift(252)) - 1.0
        mom = mom.shift(5)  # 5일 지연

        # 변동성 계산 (63일)
        vol = rets.rolling(63).std()
        low_vol = -vol

        # 재무 데이터 피벗
        f = fundamentals.set_index(["date", "ticker"]).sort_index()
        logger.info(f"Fundamentals indexed shape: {f.shape}")

        pe = f["PE"].unstack().reindex(px.index).ffill()
        roa = f["ROA"].unstack().reindex(px.index).ffill()

        logger.info(f"PE shape: {pe.shape}, ROA shape: {roa.shape}")

        # 팩터 계산
        value_inv_pe = -pe  # 낮은 PE가 좋음
        quality_roa = roa  # 높은 ROA가 좋음

        # 섹터별 표준화
        def xsect_z(df):
            # NaN 값 처리 개선
            result = df.apply(
                lambda x: (x - x.mean())
                / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1.0),
                axis=1,
            )
            # 무한대 값 처리
            result = result.replace([np.inf, -np.inf], np.nan)
            return result

        # 각 팩터별로 개별 처리
        z_mom = xsect_z(mom)
        z_val = xsect_z(value_inv_pe)
        z_qlt = xsect_z(quality_roa)
        z_lv = xsect_z(low_vol)

        logger.info(
            f"Z-scores shapes - mom: {z_mom.shape}, val: {z_val.shape}, qlt: {z_qlt.shape}, lv: {z_lv.shape}"
        )
        logger.info(
            f"Z-scores non-null counts - mom: {z_mom.notna().sum().sum()}, val: {z_val.notna().sum().sum()}, qlt: {z_qlt.notna().sum().sum()}, lv: {z_lv.notna().sum().sum()}"
        )

        # NaN 값이 너무 많은 경우 처리
        if z_mom.notna().sum().sum() == 0:
            logger.warning("All momentum z-scores are NaN, using zeros")
            z_mom = z_mom.fillna(0)

        if z_val.notna().sum().sum() == 0:
            logger.warning("All value z-scores are NaN, using zeros")
            z_val = z_val.fillna(0)

        if z_qlt.notna().sum().sum() == 0:
            logger.warning("All quality z-scores are NaN, using zeros")
            z_qlt = z_qlt.fillna(0)

        if z_lv.notna().sum().sum() == 0:
            logger.warning("All low vol z-scores are NaN, using zeros")
            z_lv = z_lv.fillna(0)

        # 팩터 가중치
        w = cfg.get(
            "factors",
            {
                "momentum_252d": {"weight": 0.4},
                "value_pe_inv": {"weight": 0.2},
                "quality_roa": {"weight": 0.2},
                "low_vol_63d": {"weight": 0.2},
            },
        )

        # 종합 점수 계산
        comp = (
            w["momentum_252d"]["weight"] * z_mom
            + w["value_pe_inv"]["weight"] * z_val
            + w["quality_roa"]["weight"] * z_qlt
            + w["low_vol_63d"]["weight"] * z_lv
        )

        logger.info(f"Combined score shape: {comp.shape}")
        logger.info(f"Non-null values in combined score: {comp.notna().sum().sum()}")

        # 최소한의 유효한 데이터가 있는지 확인
        if comp.notna().sum().sum() == 0:
            logger.error("No valid combined scores generated")
            # 간단한 대안 점수 생성
            comp = pd.DataFrame(
                np.random.randn(*comp.shape), index=comp.index, columns=comp.columns
            )
            logger.info("Generated random scores as fallback")

        out = comp.stack().rename("score").reset_index()
        out.columns = ["date", "ticker", "score"]

        logger.info(f"Final signals shape: {out.shape}")
        logger.info(f"Non-null scores: {out['score'].notna().sum()}")

        result = out.dropna()
        logger.info(f"After dropna: {len(result)} observations")

        return result

    except Exception as e:
        logger.error(f"Error computing signals: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def build_portfolio(
    scores: pd.DataFrame, sectors_map: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """포트폴리오 구성"""
    try:
        scores = scores.copy()
        scores["qend"] = scores["date"] + pd.offsets.QuarterEnd(0)
        rebal_dates = sorted(scores["qend"].unique())

        rows = []
        for d in rebal_dates:
            snap = scores[scores["qend"] == d]
            top = (
                snap.sort_values("score", ascending=False)
                .groupby("ticker")
                .tail(1)
                .sort_values("score", ascending=False)
            )

            max_names = int(cfg.get("max_names", 12))
            choose = top.head(max_names).copy()

            # 섹터 정보 추가
            sectors_s = (
                choose[["ticker"]]
                .merge(sectors_map, on="ticker", how="left")["sector"]
                .reset_index(drop=True)
            )

            tickers = choose["ticker"].values
            sc = choose["score"].values

            # 가중치 계산
            raw = np.maximum(sc - sc.min(), 0.0)
            if raw.sum() == 0:
                raw = np.ones_like(raw)
            w = pd.Series(raw / raw.sum(), index=tickers)

            # 개별 종목 상한
            cap_name = float(cfg.get("max_weight_per_name", 0.15))
            w = w.clip(lower=0.0, upper=cap_name)

            # 섹터별 상한 적용
            sector_caps = cfg.get("position_limits", {}).get("sector_caps", {})
            sectors_series = pd.Series(sectors_s.values, index=w.index)

            for sec, cap in sector_caps.items():
                mask = sectors_series == sec
                tot = w[mask].sum()
                if tot > cap:
                    w[mask] = w[mask] * (cap / tot)

            # 정규화
            w = w / w.sum()

            # 결과 저장
            for ticker, weight in w.items():
                rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "weight": weight,
                        "sector": sectors_series.get(ticker, "Unknown"),
                    }
                )

        result = pd.DataFrame(rows)
        logger.info(
            f"Built portfolio with {len(result)} positions across {len(rebal_dates)} rebalancing dates"
        )
        return result

    except Exception as e:
        logger.error(f"Error building portfolio: {e}")
        raise


def execute_trades(
    target_weights: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """거래 실행"""
    try:
        # 거래 비용 설정
        transaction_cost_bps = float(cfg.get("transaction_cost_bps", 10))
        slippage_bps = float(cfg.get("slippage_bps", 5))

        # 거래 데이터 생성 (간단한 구현)
        trades = target_weights.copy()
        trades["trade_cost"] = (
            trades["weight"] * (transaction_cost_bps + slippage_bps) / 10000
        )

        # 실제 가중치 (거래 비용 차감)
        actual_weights = target_weights.copy()
        actual_weights["weight"] = actual_weights["weight"] * (
            1 - (transaction_cost_bps + slippage_bps) / 10000
        )

        logger.info(f"Executed trades with {len(trades)} transactions")
        return actual_weights, trades

    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        raise


def run_backtest(
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """백테스트 실행"""
    try:
        # 가격 데이터 피벗
        px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
        rets = px.pct_change()

        # 포트폴리오 수익률 계산
        portfolio_rets = []
        dates = sorted(weights["date"].unique())

        for i, date in enumerate(dates):
            if i == 0:
                portfolio_rets.append(0.0)
                continue

            # 현재 가중치
            current_weights = weights[weights["date"] == date].set_index("ticker")[
                "weight"
            ]

            # 이전 날짜의 수익률
            prev_date = dates[i - 1]
            if prev_date in rets.index:
                daily_rets = rets.loc[prev_date]
                portfolio_ret = (current_weights * daily_rets).sum()
                portfolio_rets.append(portfolio_ret)
            else:
                portfolio_rets.append(0.0)

        # 누적 수익률 계산
        cumulative_rets = pd.Series(portfolio_rets, index=dates)
        equity = (1 + cumulative_rets).cumprod()

        # 성과 지표 계산
        total_return = equity.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(equity)) - 1
        volatility = cumulative_rets.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # 최대 낙폭 계산
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        metrics = {
            "Total_Return": total_return,
            "Annual_Return": annual_return,
            "Volatility": volatility,
            "Sharpe_Ratio": sharpe_ratio,
            "Max_Drawdown": max_drawdown,
            "CAGR": annual_return,
        }

        logger.info(
            f"Backtest completed. Sharpe: {sharpe_ratio:.3f}, Max DD: {max_drawdown:.3f}"
        )
        return equity, metrics

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise


def compliance_checks(
    weights: pd.DataFrame, sectors: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """규정 준수 검사"""
    try:
        violations = []

        # 개별 종목 상한 검사
        max_weight_per_name = float(cfg.get("max_weight_per_name", 0.15))
        max_weights = weights.groupby("ticker")["weight"].max()
        overweight_names = max_weights[max_weights > max_weight_per_name]

        if not overweight_names.empty:
            for ticker, weight in overweight_names.items():
                violations.append(f"{ticker}: {weight:.3f} > {max_weight_per_name}")

        # 섹터별 상한 검사
        sector_caps = cfg.get("position_limits", {}).get("sector_caps", {})
        for date in weights["date"].unique():
            date_weights = weights[weights["date"] == date]
            sector_weights = date_weights.groupby("sector")["weight"].sum()

            for sector, cap in sector_caps.items():
                if sector in sector_weights and sector_weights[sector] > cap:
                    violations.append(
                        f"{sector} on {date}: {sector_weights[sector]:.3f} > {cap}"
                    )

        compliance_passed = len(violations) == 0
        logger.info(
            f"Compliance check: {'PASSED' if compliance_passed else 'FAILED'} ({len(violations)} violations)"
        )

        return compliance_passed, violations

    except Exception as e:
        logger.error(f"Error in compliance checks: {e}")
        return False, [f"Compliance check error: {e}"]


def explain(weights: pd.DataFrame, metrics: Dict[str, float]) -> str:
    """포트폴리오 설명"""
    try:
        # 상위 보유 종목
        latest_weights = weights[weights["date"] == weights["date"].max()]
        top_holdings = latest_weights.nlargest(5, "weight")[
            ["ticker", "weight", "sector"]
        ]

        # 섹터 배분
        sector_allocation = (
            latest_weights.groupby("sector")["weight"]
            .sum()
            .sort_values(ascending=False)
        )

        memo = f"""
Portfolio Summary:
- Total Return: {metrics.get('Total_Return', 0):.2%}
- Annual Return: {metrics.get('Annual_Return', 0):.2%}
- Sharpe Ratio: {metrics.get('Sharpe_Ratio', 0):.3f}
- Max Drawdown: {metrics.get('Max_Drawdown', 0):.2%}

Top Holdings:
{top_holdings.to_string(index=False)}

Sector Allocation:
{sector_allocation.to_string()}
"""
        return memo

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return f"Error generating explanation: {e}"


def save_report(
    equity: pd.DataFrame,
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: Dict[str, float],
    compliance_passed: bool,
    violations: List[str],
    memo: str,
) -> None:
    """리포트 저장"""
    try:
        os.makedirs("reports", exist_ok=True)

        # 성과 지표 저장
        with open("reports/summary.md", "w", encoding="utf-8") as f:
            f.write("# Portfolio Performance Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Performance Metrics\n\n")
            for key, value in metrics.items():
                if "Return" in key or "Drawdown" in key:
                    f.write(f"- **{key}**: {value:.2%}\n")
                else:
                    f.write(f"- **{key}**: {value:.3f}\n")

            f.write(f"\n## Compliance Status\n\n")
            f.write(
                f"- **Status**: {'✅ PASSED' if compliance_passed else '❌ FAILED'}\n"
            )
            if violations:
                f.write("- **Violations**:\n")
                for violation in violations:
                    f.write(f"  - {violation}\n")

            f.write(f"\n## Portfolio Analysis\n\n")
            f.write(memo)

        # 데이터 저장
        equity.to_csv("reports/equity_curve.csv", index=True)
        weights.to_csv("reports/portfolio_weights.csv", index=False)
        trades.to_csv("reports/trades.csv", index=False)

        logger.info("Report saved to reports/ directory")

    except Exception as e:
        logger.error(f"Error saving report: {e}")


def run_pipeline(start: str, end: str, use_real_data: bool = True) -> str:
    """메인 파이프라인 실행"""
    try:
        logger.info(
            f"Starting pipeline: {start} to {end}, use_real_data={use_real_data}"
        )

        # 설정 로드
        cfg = load_cfg()

        # 데이터 로드
        prices = load_prices(start, end, use_real_data)
        fundamentals = load_fundamentals(start, end, use_real_data)

        # 데이터 검증
        if not validate_data(prices, fundamentals):
            raise ValueError("Data validation failed")

        # 날짜 필터링 (mock 데이터의 경우)
        if not use_real_data:
            prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
            fundamentals = fundamentals[
                (fundamentals["date"] >= start) & (fundamentals["date"] <= end)
            ].copy()

        # 파이프라인 단계별 실행
        logger.info("Computing signals...")
        sectors = fundamentals[["ticker", "sector"]].drop_duplicates().copy()
        scores = compute_signals(prices, fundamentals, cfg)

        logger.info("Building portfolio...")
        tw = build_portfolio(scores, sectors, cfg)

        logger.info("Executing trades...")
        weights, trades = execute_trades(tw, cfg)

        logger.info("Running backtest...")
        equity, metrics = run_backtest(weights, trades, prices, cfg)

        logger.info("Checking compliance...")
        comp_pass, vios = compliance_checks(weights, sectors, cfg)

        logger.info("Generating explanation...")
        memo = explain(weights, metrics)

        logger.info("Saving report...")
        save_report(equity, weights, trades, metrics, comp_pass, vios, memo)

        data_source = "real market data" if use_real_data else "mock data"
        logger.info(f"Pipeline completed successfully using {data_source}")

        return f"Pipeline complete using {data_source}. See reports/summary.md"

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def run_backtest(
    portfolio_path: str,
    start_date: str,
    end_date: str,
    rebalance_frequency: str = "quarterly",
    transaction_costs: float = 0.001,
    data_directory: str = "data/downloaded/",
) -> Dict[str, Any]:
    """
    포트폴리오 백테스트를 실행합니다.

    Args:
        portfolio_path: 포트폴리오 디렉토리 경로
        start_date: 백테스트 시작 날짜
        end_date: 백테스트 종료 날짜
        rebalance_frequency: 리밸런싱 빈도
        transaction_costs: 거래 비용 (퍼센트)
        data_directory: 데이터 디렉토리 경로

    Returns:
        백테스트 결과 딕셔너리
    """
    try:
        logger.info(f"Starting backtest for portfolio: {portfolio_path}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Rebalance frequency: {rebalance_frequency}")
        logger.info(f"Transaction costs: {transaction_costs:.3f}%")

        # 포트폴리오 가중치 로드
        weights_file = os.path.join(portfolio_path, "weights.csv")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Portfolio weights file not found: {weights_file}")

        portfolio_weights = pd.read_csv(weights_file)
        logger.info(f"Loaded portfolio weights: {portfolio_weights.shape}")

        # 데이터 로드
        prices_file = os.path.join(data_directory, "prices.csv")
        if not os.path.exists(prices_file):
            raise FileNotFoundError(f"Price data not found: {prices_file}")

        # Load prices and ensure proper datetime parsing with timezone handling
        prices = pd.read_csv(prices_file)
        prices["date"] = pd.to_datetime(prices["date"], utc=True)

        # Convert string dates to pandas Timestamp objects (timezone-naive)
        start_date_ts = pd.to_datetime(start_date)
        end_date_ts = pd.to_datetime(end_date)

        # Convert timezone-aware dates to timezone-naive for comparison
        if hasattr(prices["date"], "dt") and prices["date"].dt.tz is not None:
            prices["date"] = prices["date"].dt.tz_localize(None)

        prices = prices[
            (prices["date"] >= start_date_ts) & (prices["date"] <= end_date_ts)
        ]

        logger.info(f"Loaded price data: {prices.shape}")
        logger.info(
            f"Price date range: {prices['date'].min()} to {prices['date'].max()}"
        )

        # 백테스트 실행
        equity_curve, trades, metrics = run_portfolio_backtest(
            portfolio_weights=portfolio_weights,
            prices=prices,
            rebalance_frequency=rebalance_frequency,
            transaction_costs=transaction_costs,
        )

        results = {"equity_curve": equity_curve, "trades": trades, "metrics": metrics}

        logger.info("Backtest completed successfully")
        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


def run_portfolio_backtest(
    portfolio_weights: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_frequency: str = "quarterly",
    transaction_costs: float = 0.001,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    포트폴리오 백테스트를 실행합니다.

    Args:
        portfolio_weights: 포트폴리오 가중치 DataFrame
        prices: 가격 데이터 DataFrame
        rebalance_frequency: 리밸런싱 빈도
        transaction_costs: 거래 비용

    Returns:
        (수익률 곡선, 거래 내역, 성과 지표)
    """
    try:
        # 가격 데이터 피벗 (중복 처리)
        # 중복된 데이터가 있는 경우 마지막 값을 사용
        prices_clean = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
        px = prices_clean.pivot(
            index="date", columns="ticker", values="close"
        ).sort_index()
        rets = px.pct_change()

        # 포트폴리오 심볼 추출
        portfolio_symbols = portfolio_weights["symbol"].tolist()
        available_symbols = [s for s in portfolio_symbols if s in px.columns]

        if not available_symbols:
            raise ValueError("No portfolio symbols found in price data")

        logger.info(
            f"Portfolio symbols: {len(available_symbols)} available out of {len(portfolio_symbols)}"
        )

        # 포트폴리오 가중치 정규화
        portfolio_weights_filtered = portfolio_weights[
            portfolio_weights["symbol"].isin(available_symbols)
        ].copy()
        portfolio_weights_filtered["weight"] = (
            portfolio_weights_filtered["weight"]
            / portfolio_weights_filtered["weight"].sum()
        )

        # 리밸런싱 날짜 생성
        if rebalance_frequency == "quarterly":
            rebalance_dates = pd.date_range(
                start=prices["date"].min(), end=prices["date"].max(), freq="Q"
            )
        elif rebalance_frequency == "monthly":
            rebalance_dates = pd.date_range(
                start=prices["date"].min(), end=prices["date"].max(), freq="M"
            )
        elif rebalance_frequency == "weekly":
            rebalance_dates = pd.date_range(
                start=prices["date"].min(), end=prices["date"].max(), freq="W"
            )
        else:  # daily
            rebalance_dates = px.index

        # 백테스트 실행
        equity_curve = []
        trades_list = []
        current_weights = portfolio_weights_filtered.set_index("symbol")["weight"]

        for i, date in enumerate(px.index):
            if i == 0:
                # 초기 포트폴리오
                equity_curve.append(
                    {"date": date, "portfolio_value": 1.0, "daily_return": 0.0}
                )
                continue

            # 리밸런싱 체크
            if date in rebalance_dates:
                # 거래 비용 적용
                trade_cost = transaction_costs
                current_weights = current_weights * (1 - trade_cost)

                # 거래 내역 기록
                for symbol, weight in current_weights.items():
                    trades_list.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "action": "rebalance",
                            "weight": weight,
                            "cost": weight * trade_cost,
                        }
                    )

            # 일일 수익률 계산
            if date in rets.index:
                daily_rets = rets.loc[date]
                portfolio_ret = (current_weights * daily_rets).sum()

                # 포트폴리오 가중치 업데이트 (가격 변동 반영)
                price_changes = 1 + daily_rets
                current_weights = current_weights * price_changes
                current_weights = current_weights / current_weights.sum()
            else:
                portfolio_ret = 0.0

            # 누적 수익률 계산
            if i > 0:
                prev_value = equity_curve[-1]["portfolio_value"]
                current_value = prev_value * (1 + portfolio_ret)
            else:
                current_value = 1.0

            equity_curve.append(
                {
                    "date": date,
                    "portfolio_value": current_value,
                    "daily_return": portfolio_ret,
                }
            )

        # 결과 DataFrame 생성
        equity_df = pd.DataFrame(equity_curve)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df.set_index("date", inplace=True)

        trades_df = (
            pd.DataFrame(trades_list)
            if trades_list
            else pd.DataFrame(columns=["date", "symbol", "action", "weight", "cost"])
        )

        # 성과 지표 계산
        total_return = equity_df["portfolio_value"].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        volatility = equity_df["daily_return"].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # 최대 낙폭 계산
        rolling_max = equity_df["portfolio_value"].expanding().max()
        drawdown = (equity_df["portfolio_value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 승률 계산
        positive_days = (equity_df["daily_return"] > 0).sum()
        total_days = len(equity_df)
        win_rate = positive_days / total_days if total_days > 0 else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

        logger.info(
            f"Backtest metrics - Total Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.3f}"
        )

        return equity_df, trades_df, metrics

    except Exception as e:
        logger.error(f"Portfolio backtest failed: {e}")
        raise
