import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _choose_price_column(prices: pd.DataFrame) -> str:
    for c in ["adj_close", "close", "price"]:
        if c in prices.columns:
            return c
    raise ValueError("prices.csv에 adj_close/close/price 중 하나가 필요합니다.")


def _winsorize_zscore(s: pd.Series, lo_q=5, hi_q=95) -> pd.Series:
    s = s.astype(float)
    lo = np.nanpercentile(s, lo_q)
    hi = np.nanpercentile(s, hi_q)
    s = s.clip(lo, hi)
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _compute_factor_table(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    factors_cfg: dict,
    momentum_lb: int = 252,
    vol_lb: int = 63,
) -> pd.DataFrame:
    """팩터별 원천 값 → 윈저+zscore → 단일 테이블 리턴(index=ticker)."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    price_col = _choose_price_column(prices)

    # 중복된 데이터 제거 (같은 날짜, 같은 ticker에 대해 마지막 값만 유지)
    prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")

    # 피벗 및 수익률 (중복된 ticker 제거)
    px = prices.pivot(index="date", columns="ticker", values=price_col).sort_index()

    # 중복된 컬럼(동일한 ticker)이 있으면 첫 번째만 유지
    if px.columns.duplicated().any():
        duplicated_cols = px.columns[px.columns.duplicated()].unique()
        logger.warning(
            f"Duplicate tickers found and removed: {duplicated_cols.tolist()}"
        )
        px = px.loc[:, ~px.columns.duplicated()]

    ret = px.pct_change()

    # 모멘텀(252d 누적수익)
    if len(px) < momentum_lb + 2:
        raise ValueError(f"가격 히스토리가 부족합니다(>= {momentum_lb+2} 일 필요).")
    mom = (px / px.shift(momentum_lb) - 1.0).iloc[-1]

    # 저변동성(63d): 변동성 낮을수록 점수↑ → score = -std
    if len(ret) < vol_lb + 2:
        raise ValueError(f"수익률 히스토리가 부족합니다(>= {vol_lb+2} 일 필요).")
    vol = ret.rolling(vol_lb).std().iloc[-1]
    low_vol_score = -vol

    # 펀더멘탈: 역PER, ROA (존재 시)
    pe_inv = None
    roa = None
    sector_map = None
    if fundamentals is not None:
        f = fundamentals.copy()
        # ticker, pe, pb, roa, roe, sector 등의 컬럼이 있다고 가정하되, 없으면 넘어감
        if "ticker" not in f.columns:
            # 가격과 조인 위해 컬럼명 표준화 요구
            pass
        else:
            # 중복된 ticker 제거 (첫 번째만 유지)
            if f["ticker"].duplicated().any():
                duplicated_tickers = f[f["ticker"].duplicated()]["ticker"].unique()
                logger.warning(
                    f"Duplicate tickers in fundamentals found and removed: {duplicated_tickers.tolist()}"
                )
                f = f.drop_duplicates(subset=["ticker"], keep="first")

        f = f.set_index("ticker") if "ticker" in f.columns else f

        if "pe" in f.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                pe_inv = 1.0 / f["pe"].replace({0: np.nan})
        elif "pb" in f.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                pe_inv = 1.0 / (f["pb"] * 15.0)  # 매우 러프한 대체(PE~15*PB 가정)

        if "roa" in f.columns:
            roa = f["roa"]
        elif "roe" in f.columns:
            # roe로 대체 가능(스케일만 다름)
            roa = 0.6 * f["roe"]

        if "sector" in f.columns:
            sector_map = f["sector"]

    # 팩터 dict 구성 (존재 여부 체크)
    raw = pd.DataFrame(index=px.columns)
    if "momentum_252d" in factors_cfg:
        raw["momentum_252d"] = mom.reindex(raw.index)
    if "low_vol_63d" in factors_cfg:
        raw["low_vol_63d"] = low_vol_score.reindex(raw.index)
    if "value_pe_inv" in factors_cfg and pe_inv is not None:
        raw["value_pe_inv"] = pe_inv.reindex(raw.index)
    if "quality_roa" in factors_cfg and roa is not None:
        raw["quality_roa"] = roa.reindex(raw.index)

    # 팩터별 윈저+zscore
    for c in raw.columns:
        raw[c] = _winsorize_zscore(raw[c])

    # 섹터
    if sector_map is not None:
        raw["__sector__"] = sector_map.reindex(raw.index)

    return raw


def _cov_shrinkage(
    cov: np.ndarray, shrink_to: str = "diag", lam: float | None = None
) -> np.ndarray:
    """간단한 수축 공분산. 표본 공분산을 대각(diagonal)로 수축."""
    n = cov.shape[0]
    if shrink_to != "diag":
        raise NotImplementedError("현재는 diag 수축만 지원합니다.")
    diag = np.diag(np.diag(cov))
    if lam is None:
        # 샘플 개수(T)가 작을수록 수축을 키움
        # 대략적 휴리스틱: lam = min(0.5, 0.1 + 0.9 * (n / T)) 형태를 쓰고,
        # T(표본 수)는 추정이 어려우니 보수적으로 0.15~0.35 사이에서 조정
        lam = 0.25
    return (1 - lam) * cov + lam * diag


def _diversified_inv_vol_weights(cov: np.ndarray) -> np.ndarray:
    """역변동성 기반 + 상관분산화 보정(avg corr ↓ ⇒ 가중치 ↑)."""
    std = np.sqrt(np.diag(cov))
    std[std <= 1e-12] = 1e-12
    base = 1.0 / std

    # 상관행렬
    inv_std = np.diag(1 / std)
    corr = inv_std @ cov @ inv_std
    # 평균 상관
    n = len(std)
    with np.errstate(invalid="ignore"):
        avg_corr = (corr.sum(axis=1) - 1.0) / (n - 1.0)
    div_adj = np.clip(1.0 - avg_corr, 0.5, 1.5)
    w = base * div_adj
    w = np.maximum(w, 0)
    return w / w.sum()


def _apply_name_cap(
    weights: pd.Series, cap: float, raw_pref: pd.Series | None = None
) -> pd.Series:
    """종목 상한(cap) 적용: 초과분을 uncapped 비중에 비례 재분배(반복)."""
    if raw_pref is None:
        raw_pref = weights.copy()
    w = weights.copy()
    raw = raw_pref.copy()
    cap = float(cap)
    for _ in range(20):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap

        under = ~over
        if not under.any():
            # 모두 캡이면 그냥 정규화
            w = w / w.sum()
            break
        alloc_base = raw[under].clip(lower=0)
        s = alloc_base.sum()
        if s <= 1e-12:
            # 선호가 0이면 현재 비중 비례
            alloc_base = w[under]
            s = alloc_base.sum()
        if s <= 1e-12:
            # 그래도 0이면 균등
            alloc_base = pd.Series(1.0, index=w[under].index)
            s = alloc_base.sum()
        w[under] += excess * (alloc_base / s)
    # 수치 안정화
    w = w.clip(lower=0)
    return w / w.sum()


def _apply_sector_caps(
    weights: pd.Series,
    sectors: pd.Series | None,
    sector_caps: dict[str, float] | None,
    raw_pref: pd.Series | None = None,
) -> pd.Series:
    """섹터 캡 적용(반복 축소/재분배). 섹터 정보/캡 없으면 그대로 반환."""
    if sectors is None or sector_caps is None or len(sector_caps) == 0:
        return weights
    w = weights.copy()
    if raw_pref is None:
        raw_pref = w.copy()

    for _ in range(30):
        # 섹터 합
        sec_w = w.groupby(sectors.reindex(w.index)).sum()
        over_secs = [
            s
            for s, val in sec_w.items()
            if s in sector_caps and val > sector_caps[s] + 1e-12
        ]
        if not over_secs:
            break

        # 초과 섹터 비중 줄이고 초과분을 타 섹터로 재분배
        total_excess = 0.0
        for s in over_secs:
            cur = sec_w[s]
            cap = sector_caps[s]
            if cur <= cap:
                continue
            # 해당 섹터 가중치들을 비례 축소
            idx = w.index[sectors.reindex(w.index) == s]
            if len(idx) == 0:
                continue
            factor = cap / cur
            reduced = w.loc[idx] * (1 - factor)
            w.loc[idx] *= factor
            total_excess += reduced.sum()

        # 재분배: 언더캡 섹터들(또는 캡 미지정 섹터) 대상으로 raw_pref 비례
        sec_w = w.groupby(sectors.reindex(w.index)).sum()
        under_idx = []
        alloc_pref = []
        for sym in w.index:
            sec = sectors.get(sym, None)
            cap = sector_caps.get(sec, 1.0) if sec in sector_caps else 1.0
            if sec_w.get(sec, 0.0) < cap - 1e-12:
                under_idx.append(sym)
                alloc_pref.append(max(raw_pref.get(sym, 0.0), 0.0))
        if len(under_idx) == 0 or total_excess <= 1e-12:
            # 재분배 대상 없으면 균등
            under_idx = list(w.index)
            alloc_pref = [1.0] * len(under_idx)

        alloc_pref = np.array(alloc_pref, dtype=float)
        s = alloc_pref.sum()
        if s <= 1e-12:
            alloc_pref = np.ones_like(alloc_pref)
            s = alloc_pref.sum()

        w.loc[under_idx] += total_excess * (alloc_pref / s)

    return (w / w.sum()).clip(lower=0)
