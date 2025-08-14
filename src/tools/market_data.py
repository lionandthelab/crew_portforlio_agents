import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import time
import warnings
import os
import hashlib
import json
import logging

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 캐시 디렉토리
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# S&P 500 대표 주식들 (실제 거래량이 많고 안정적인 주식들) - 100개로 확장
SP500_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "UNH",
    "JNJ",
    "JPM",
    "V",
    "PG",
    "HD",
    "MA",
    "PFE",
    "ABBV",
    "AVGO",
    "KO",
    "PEP",
    "COST",
    "TMO",
    "DHR",
    "ACN",
    "WMT",
    "MRK",
    "ABT",
    "VZ",
    "NKE",
    "ADBE",
    "CRM",
    "PM",
    "TXN",
    "NEE",
    "RTX",
    "HON",
    "QCOM",
    "LOW",
    "UPS",
    "IBM",
    "MS",
    "GS",
    "CAT",
    "BA",
    "UNP",
    "DE",
    "LMT",
    "SPGI",
    "INTU",
    "GILD",
    "CVX",
    "XOM",
    "LLY",
    "BKNG",
    "SCHW",
    "T",
    "CMCSA",
    "AMGN",
    "ADI",
    "ISRG",
    "VRTX",
    "REGN",
    "PANW",
    "KLAC",
    "SNPS",
    "CDNS",
    "MELI",
    "MU",
    "ASML",
    "ORLY",
    "MNST",
    "CHTR",
    "MAR",
    "MCHP",
    "ADP",
    "CTAS",
    "PAYX",
    "ROST",
    "BIIB",
    "ALGN",
    "DXCM",
    "IDXX",
    "CPRT",
    "FAST",
    "ODFL",
    "EXC",
    "AEP",
    "SO",
    "DUK",
    "D",
    "SRE",
    "NEE",
    "XEL",
    "WEC",
    "DTE",
    "AEE",
    "PEG",
    "EIX",
    "ED",
    "PCG",
    "ETR",
    "AES",
    "NRG",
    "VST",
    "CEG",
    "CNP",
    "NI",
    "ALE",
    "LNT",
    "ATO",
]

# 섹터 매핑 (실제 GICS 섹터 기준) - 100개 주식에 대한 확장
SECTOR_MAPPING = {
    # 기존 50개
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Technology",
    "TSLA": "Consumer Discretionary",
    "BRK-B": "Financials",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "JPM": "Financials",
    "V": "Financials",
    "PG": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "MA": "Financials",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "AVGO": "Technology",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "TMO": "Healthcare",
    "DHR": "Industrials",
    "ACN": "Technology",
    "WMT": "Consumer Staples",
    "MRK": "Healthcare",
    "ABT": "Healthcare",
    "VZ": "Communication Services",
    "NKE": "Consumer Discretionary",
    "ADBE": "Technology",
    "CRM": "Technology",
    "PM": "Consumer Staples",
    "TXN": "Technology",
    "NEE": "Utilities",
    "RTX": "Industrials",
    "HON": "Industrials",
    "QCOM": "Technology",
    "LOW": "Consumer Discretionary",
    "UPS": "Industrials",
    "IBM": "Technology",
    "MS": "Financials",
    "GS": "Financials",
    "CAT": "Industrials",
    "BA": "Industrials",
    "UNP": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",
    "SPGI": "Financials",
    "INTU": "Technology",
    "GILD": "Healthcare",
    # 추가 50개
    "CVX": "Energy",
    "XOM": "Energy",
    "LLY": "Healthcare",
    "BKNG": "Consumer Discretionary",
    "SCHW": "Financials",
    "T": "Communication Services",
    "CMCSA": "Communication Services",
    "AMGN": "Healthcare",
    "ADI": "Technology",
    "ISRG": "Healthcare",
    "VRTX": "Healthcare",
    "REGN": "Healthcare",
    "PANW": "Technology",
    "KLAC": "Technology",
    "SNPS": "Technology",
    "CDNS": "Technology",
    "MELI": "Consumer Discretionary",
    "MU": "Technology",
    "ASML": "Technology",
    "ORLY": "Consumer Discretionary",
    "MNST": "Consumer Staples",
    "CHTR": "Communication Services",
    "MAR": "Consumer Discretionary",
    "MCHP": "Technology",
    "ADP": "Technology",
    "CTAS": "Industrials",
    "PAYX": "Technology",
    "ROST": "Consumer Discretionary",
    "BIIB": "Healthcare",
    "ALGN": "Healthcare",
    "DXCM": "Healthcare",
    "IDXX": "Healthcare",
    "CPRT": "Industrials",
    "FAST": "Industrials",
    "ODFL": "Industrials",
    "EXC": "Utilities",
    "AEP": "Utilities",
    "SO": "Utilities",
    "DUK": "Utilities",
    "D": "Utilities",
    "SRE": "Utilities",
    "NEE": "Utilities",
    "XEL": "Utilities",
    "WEC": "Utilities",
    "DTE": "Utilities",
    "AEE": "Utilities",
    "PEG": "Utilities",
    "EIX": "Utilities",
    "ED": "Utilities",
    "PCG": "Utilities",
    "ETR": "Utilities",
    "AES": "Utilities",
    "NRG": "Utilities",
    "VST": "Utilities",
    "CEG": "Utilities",
    "CNP": "Utilities",
    "NI": "Utilities",
    "ALE": "Utilities",
    "LNT": "Utilities",
    "ATO": "Utilities",
}


class DataCache:
    """데이터 캐싱을 위한 클래스"""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(
        self, data_type: str, start_date: str, end_date: str, tickers: List[str]
    ) -> str:
        """캐시 키 생성"""
        content = f"{data_type}_{start_date}_{end_date}_{'_'.join(sorted(tickers))}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, data_type: str) -> str:
        """캐시 파일 경로 반환"""
        return os.path.join(self.cache_dir, f"{data_type}_{cache_key}.csv")

    def exists(self, cache_key: str, data_type: str) -> bool:
        """캐시 파일 존재 여부 확인"""
        cache_path = self._get_cache_path(cache_key, data_type)
        return os.path.exists(cache_path)

    def load(self, cache_key: str, data_type: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 로드"""
        try:
            cache_path = self._get_cache_path(cache_key, data_type)
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, parse_dates=["date"])
                logger.info(f"Loaded {data_type} data from cache: {cache_path}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return None

    def save(self, data: pd.DataFrame, cache_key: str, data_type: str) -> None:
        """데이터를 캐시에 저장"""
        try:
            cache_path = self._get_cache_path(cache_key, data_type)
            data.to_csv(cache_path, index=False)
            logger.info(f"Saved {data_type} data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class MarketDataLoader:
    """시장 데이터 로더 클래스"""

    def __init__(self, cache: DataCache = None):
        self.cache = cache or DataCache()

    def get_available_tickers(self, max_tickers: int = 100) -> List[str]:
        """사용 가능한 티커 목록 반환"""
        return SP500_TICKERS[:max_tickers]

    def download_price_data(
        self, tickers: List[str], start_date: str, end_date: str, use_cache: bool = True
    ) -> pd.DataFrame:
        """가격 데이터 다운로드"""
        cache_key = self.cache._get_cache_key("prices", start_date, end_date, tickers)

        # 캐시에서 로드 시도
        if use_cache:
            cached_data = self.cache.load(cache_key, "prices")
            if cached_data is not None:
                return cached_data

        logger.info(f"Downloading price data for {len(tickers)} tickers...")

        all_data = []
        successful_tickers = []

        for ticker in tickers:
            try:
                logger.info(f"Downloading {ticker}...")
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)

                if not data.empty:
                    data = data.reset_index()
                    data["ticker"] = ticker
                    data = data[["Date", "ticker", "Close", "Volume"]]
                    data.columns = ["date", "ticker", "close", "volume"]
                    all_data.append(data)
                    successful_tickers.append(ticker)
                    logger.info(f"Successfully downloaded {ticker}")
                else:
                    logger.warning(f"No data for {ticker}")

                # API 제한 방지
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")
                continue

        if not all_data:
            logger.error("No price data downloaded")
            return pd.DataFrame()

        # 데이터 결합
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(["date", "ticker"]).reset_index(
            drop=True
        )

        # 캐시에 저장
        if use_cache and not combined_data.empty:
            self.cache.save(combined_data, cache_key, "prices")

        logger.info(f"Downloaded price data for {len(successful_tickers)} tickers")
        return combined_data

    def download_fundamental_data(
        self, tickers: List[str], start_date: str, end_date: str, use_cache: bool = True
    ) -> pd.DataFrame:
        """재무 데이터 다운로드 (yfinance API 사용)"""
        cache_key = self.cache._get_cache_key(
            "fundamentals", start_date, end_date, tickers
        )

        # 캐시에서 로드 시도
        if use_cache:
            cached_data = self.cache.load(cache_key, "fundamentals")
            if cached_data is not None:
                return cached_data

        logger.info(f"Downloading fundamental data for {len(tickers)} tickers...")

        # 월간 날짜 생성 (Quarter 대신 Monthly)
        dates = pd.date_range(start=start_date, end=end_date, freq="M")

        fundamental_data = []
        successful_tickers = []

        for ticker in tickers:
            try:
                logger.info(f"Downloading fundamental data for {ticker}...")
                stock = yf.Ticker(ticker)

                # 재무 데이터 가져오기
                info = stock.info

                # 기본 정보 추출
                pe_ratio = info.get("trailingPE", np.nan)
                pb_ratio = info.get("priceToBook", np.nan)
                roe = info.get("returnOnEquity", np.nan)
                roa = info.get("returnOnAssets", np.nan)
                sector = info.get("sector", SECTOR_MAPPING.get(ticker, "Technology"))

                # 각 월별로 데이터 생성
                for date in dates:
                    fundamental_data.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "PE": pe_ratio,
                            "PB": pb_ratio,
                            "ROE": roe,
                            "ROA": roa,
                            "sector": sector,
                        }
                    )

                successful_tickers.append(ticker)
                logger.info(f"Successfully downloaded fundamental data for {ticker}")

                # API 제한 방지
                time.sleep(0.2)

            except Exception as e:
                logger.warning(f"Failed to download fundamental data for {ticker}: {e}")
                # 실패한 경우 기본값 사용
                sector = SECTOR_MAPPING.get(ticker, "Technology")
                for date in dates:
                    fundamental_data.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "PE": np.random.uniform(10, 30),
                            "PB": np.random.uniform(1, 5),
                            "ROE": np.random.uniform(0.05, 0.25),
                            "ROA": np.random.uniform(0.05, 0.20),
                            "sector": sector,
                        }
                    )
                continue

        df = pd.DataFrame(fundamental_data)

        # NaN 값 처리
        df = df.fillna(
            {
                "PE": df["PE"].median(),
                "PB": df["PB"].median(),
                "ROE": df["ROE"].median(),
                "ROA": df["ROA"].median(),
            }
        )

        # 캐시에 저장
        if use_cache and not df.empty:
            self.cache.save(df, cache_key, "fundamentals")

        logger.info(
            f"Downloaded fundamental data for {len(successful_tickers)} tickers"
        )
        return df

    def create_mock_data_fallback(
        self, start_date: str, end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Mock 데이터 생성 (fallback용)"""
        logger.info("Creating fallback mock data...")

        tickers = self.get_available_tickers(10)  # 처음 10개만 사용
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # 가격 데이터
        price_data = []
        for ticker in tickers:
            base_price = 100 + np.random.randint(0, 900)
            for date in dates:
                change = np.random.normal(0, 0.02)
                price = base_price * (1 + change)
                base_price = price

                price_data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "close": max(price, 1),
                        "volume": np.random.randint(1000000, 10000000),
                    }
                )

        # 재무 데이터
        fundamental_data = []
        for ticker in tickers:
            sector = SECTOR_MAPPING.get(ticker, "Technology")
            for date in pd.date_range(start=start_date, end=end_date, freq="M"):
                fundamental_data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "PE": np.random.uniform(10, 30),
                        "PB": np.random.uniform(1, 5),
                        "ROE": np.random.uniform(0.05, 0.25),
                        "ROA": np.random.uniform(0.05, 0.20),
                        "sector": sector,
                    }
                )

        return pd.DataFrame(price_data), pd.DataFrame(fundamental_data)

    def load_market_data(
        self,
        start_date: str,
        end_date: str,
        use_real_data: bool = True,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """시장 데이터 로드 (메인 함수)"""
        if use_real_data:
            try:
                tickers = self.get_available_tickers()
                prices = self.download_price_data(
                    tickers, start_date, end_date, use_cache
                )
                fundamentals = self.download_fundamental_data(
                    tickers, start_date, end_date, use_cache
                )

                if prices.empty or fundamentals.empty:
                    logger.warning("Real data download failed, using fallback data...")
                    return self.create_mock_data_fallback(start_date, end_date)

                return prices, fundamentals

            except Exception as e:
                logger.error(f"Error downloading real data: {e}")
                logger.info("Using fallback mock data...")
                return self.create_mock_data_fallback(start_date, end_date)
        else:
            logger.info("Using mock data as requested...")
            return self.create_mock_data_fallback(start_date, end_date)


# 전역 인스턴스
_cache = DataCache()
_market_loader = MarketDataLoader(_cache)


def load_real_market_data(
    start_date: str, end_date: str, use_real_data: bool = True, use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """실제 시장 데이터를 로드하거나 fallback 데이터를 생성합니다."""
    return _market_loader.load_market_data(
        start_date, end_date, use_real_data, use_cache
    )


def clear_cache():
    """캐시 디렉토리의 모든 파일을 삭제합니다."""
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            filepath = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        logger.info(f"Cache cleared: {CACHE_DIR}")


def get_cache_info():
    """캐시 정보를 반환합니다."""
    if not os.path.exists(CACHE_DIR):
        return {"cache_dir": CACHE_DIR, "files": [], "total_size": 0}

    files = []
    total_size = 0

    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            files.append(
                {
                    "filename": filename,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
            total_size += size

    return {
        "cache_dir": CACHE_DIR,
        "files": files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


def download_market_data(
    start_date: str,
    end_date: str,
    output_dir: str,
    symbols: List[str] = None,
    use_mock_data: bool = False,
) -> Dict[str, str]:
    """
    시장 데이터를 다운로드하고 지정된 디렉토리에 저장합니다.

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        output_dir: 출력 디렉토리
        symbols: 다운로드할 심볼 리스트 (None이면 기본 심볼 사용)
        use_mock_data: Mock 데이터 사용 여부

    Returns:
        다운로드된 파일들의 경로 딕셔너리
    """
    logger.info(f"Starting market data download...")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Use mock data: {use_mock_data}")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 심볼 설정
    if symbols is None:
        symbols = _market_loader.get_available_tickers()

    logger.info(f"Downloading data for {len(symbols)} symbols: {symbols}")

    try:
        # 데이터 다운로드
        prices, fundamentals = _market_loader.load_market_data(
            start_date=start_date,
            end_date=end_date,
            use_real_data=not use_mock_data,
            use_cache=True,
        )

        # 파일 저장
        downloaded_files = {}

        # 가격 데이터 저장
        if not prices.empty:
            prices_file = os.path.join(output_dir, "prices.csv")
            prices.to_csv(prices_file, index=False)
            downloaded_files["prices"] = prices_file
            logger.info(f"Price data saved to: {prices_file}")
            logger.info(f"Price data shape: {prices.shape}")

        # 재무 데이터 저장
        if not fundamentals.empty:
            fundamentals_file = os.path.join(output_dir, "fundamentals.csv")
            fundamentals.to_csv(fundamentals_file, index=False)
            downloaded_files["fundamentals"] = fundamentals_file
            logger.info(f"Fundamental data saved to: {fundamentals_file}")
            logger.info(f"Fundamental data shape: {fundamentals.shape}")

        # 다운로드 요약 정보 저장
        summary = {
            "download_date": datetime.now().isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "use_mock_data": use_mock_data,
            "prices_shape": prices.shape if not prices.empty else None,
            "fundamentals_shape": (
                fundamentals.shape if not fundamentals.empty else None
            ),
            "files": downloaded_files,
        }

        summary_file = os.path.join(output_dir, "download_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        downloaded_files["summary"] = summary_file
        logger.info(f"Download summary saved to: {summary_file}")

        logger.info("Market data download completed successfully!")
        return downloaded_files

    except Exception as e:
        logger.error(f"Market data download failed: {e}")
        raise
