# Portfolio Analysis Workflow

포트폴리오 분석을 위한 단계별 워크플로우 시스템입니다. 데이터 다운로드, 포트폴리오 생성, 백테스팅을 순차적으로 실행할 수 있습니다.

## 프로젝트 구조

```
crew_portfolio_agents/
├── step1_download_data.py      # 1단계: 시장 데이터 다운로드
├── step2_generate_portfolios.py # 2단계: 포트폴리오 생성
├── step3_backtest.py           # 3단계: 포트폴리오 백테스팅
├── run_workflow.py             # 전체 워크플로우 실행
├── src/
│   └── tools/
│       ├── market_data.py      # 시장 데이터 도구
│       ├── portfolio_generator.py # 포트폴리오 생성 도구
│       ├── pipeline_tool_impl.py # 백테스팅 도구
│       └── portfolio_tools.py  # 그래프 및 비교 도구
├── data/
│   └── downloaded/             # 다운로드된 데이터
├── portfolios/                 # 생성된 포트폴리오
└── reports/                    # 백테스트 리포트
```

## 설치 및 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (선택사항)
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 추가 (실제 데이터 사용 시)
```

## 사용법

### 전체 워크플로우 실행

```bash
# 기본 실행 (2020-2024년, 모든 포트폴리오 타입)
python run_workflow.py

# 특정 기간과 포트폴리오 타입 지정
python run_workflow.py --start 2020-01-01 --end 2025-07-31 --portfolio-types growth,value,balanced

# Mock 데이터 사용
python run_workflow.py --use-mock

# 특정 단계 건너뛰기
python run_workflow.py --skip-step1 --skip-step2

# 특정 포트폴리오만 백테스팅
python run_workflow.py --backtest-portfolio portfolio_01_growth
```

### 개별 단계 실행

#### 1단계: 데이터 다운로드
```bash
python step1_download_data.py --start 2020-01-01 --end 2025-07-31 --output data/downloaded/
```

#### 2단계: 포트폴리오 생성
```bash
python step2_generate_portfolios.py --data-dir data/downloaded/ --num-portfolios 1
```

#### 3단계: 백테스팅
```bash
python step3_backtest.py --portfolio portfolios/portfolio_01_growth --start 2020-01-01 --end 2025-07-31
```

## 포트폴리오 타입

### 1. Growth Portfolio (성장 포트폴리오)
- **목적**: 장기 성장을 추구
- **전략**: 모멘텀과 성장성을 중심으로 구성
- **구성**: 고성장 기업 위주
- **팩터 가중치**: 모멘텀 60%, 가치 10%, 품질 20%, 저변동성 10%

### 2. Value Portfolio (가치 포트폴리오)
- **목적**: 저평가 주식에 투자하여 가치 발견
- **전략**: P/E, P/B 등 가치 지표 중심
- **구성**: 저평가 기업 위주
- **팩터 가중치**: 모멘텀 20%, 가치 50%, 품질 20%, 저변동성 10%

### 3. Balanced Portfolio (균형 포트폴리오)
- **목적**: 위험과 수익의 균형을 추구
- **전략**: 모든 팩터를 균등하게 적용
- **구성**: 다양한 스타일의 기업으로 구성
- **팩터 가중치**: 모멘텀 40%, 가치 20%, 품질 20%, 저변동성 20%

### 4. Low Volatility Portfolio (저변동성 포트폴리오)
- **목적**: 안정적인 수익을 추구
- **전략**: 저변동성 주식 중심으로 구성
- **구성**: 안정적인 기업 위주
- **팩터 가중치**: 모멘텀 20%, 가치 20%, 품질 20%, 저변동성 40%

### 5. Quality Portfolio (품질 포트폴리오)
- **목적**: 고품질 기업에 투자
- **전략**: ROA, ROE 등 품질 지표 중심
- **구성**: 고ROA 기업 위주
- **팩터 가중치**: 모멘텀 20%, 가치 10%, 품질 50%, 저변동성 20%

## 출력 파일 구조

### 1. 데이터 다운로드 (`data/downloaded/`)
- `prices.csv`: 주가 데이터
- `fundamentals.csv`: 재무 데이터
- `download_summary.json`: 다운로드 요약 정보

### 2. 포트폴리오 생성 (`portfolios/`)
각 포트폴리오별로 별도 폴더 생성:
- `portfolio_01_growth/`
- `portfolio_02_value/`
- `portfolio_03_balanced/`
- `portfolio_04_low_vol/`
- `portfolio_05_quality/`

각 폴더 내부:
- `weights.csv`: 포트폴리오 가중치
- `portfolio_description.md`: 포트폴리오 설명
- `metadata.json`: 포트폴리오 메타데이터

### 3. 백테스트 리포트 (`reports/`)
각 백테스트별로 타임스탬프가 포함된 폴더 생성:
- `Growth_Portfolio_2020-01-01_2025-07-31_20241201_143022/`
- `Value_Portfolio_2020-01-01_2025-07-31_20241201_143045/`

각 리포트 폴더 내부:
- `equity_curve.csv`: 일별 포트폴리오 가치
- `equity_curve_plot.png`: 포트폴리오 수익률 곡선 그래프
- `trades.csv`: 거래 내역
- `performance_metrics.json`: 성과 지표
- `portfolio_weights.csv`: 원본 포트폴리오 가중치
- `portfolio_description.md`: 포트폴리오 전략 설명
- `backtest_metadata.json`: 백테스트 메타데이터
- `backtest_summary.md`: 백테스트 요약 보고서

### 4. 워크플로우 요약 (`reports/`)
- `workflow_summary_YYYYMMDD_HHMMSS.json`: 전체 워크플로우 요약
- `performance_comparison_YYYYMMDD_HHMMSS.csv`: 포트폴리오 성과 비교 표

## 성과 지표

각 백테스트에서 계산되는 성과 지표:

- **Total Return**: 전체 수익률
- **Annualized Return**: 연간 수익률
- **Volatility**: 변동성 (연간화)
- **Sharpe Ratio**: 샤프 비율 (위험조정수익률)
- **Max Drawdown**: 최대 낙폭
- **Win Rate**: 승률 (양의 수익률을 기록한 일수의 비율)

## 의존성

- `pandas>=2.0`: 데이터 처리
- `numpy>=1.24`: 수치 계산
- `yfinance>=0.2.0`: 시장 데이터 다운로드
- `matplotlib>=3.7.0`: 그래프 생성
- `seaborn>=0.12.0`: 그래프 스타일링
- `pyyaml>=6.0`: 설정 파일 처리
- `python-dotenv>=1.0.1`: 환경 변수 관리

## 주의사항

1. **Mock 데이터**: 기본적으로 Mock 데이터를 사용합니다. 실제 데이터를 사용하려면 `.env` 파일에 API 키를 설정하세요.
2. **데이터 캐싱**: 다운로드한 데이터는 `data/cache/` 폴더에 캐시됩니다.
3. **리밸런싱**: 기본적으로 분기별 리밸런싱을 수행합니다.
4. **거래 비용**: 기본 거래 비용은 0.1%로 설정되어 있습니다.

## 라이선스

MIT License