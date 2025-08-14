# Portfolio Analysis Workflow

포트폴리오 분석을 위한 단계별 워크플로우 시스템입니다.

## 프로젝트 구조

```
crew_portfolio_agents/
├── step1_download_data.py      # Step 1: 데이터 다운로드
├── step2_generate_portfolios.py # Step 2: 포트폴리오 생성
├── step3_backtest.py           # Step 3: 백테스팅
├── run_workflow.py             # 전체 워크플로우 실행
├── data/
│   ├── downloaded/             # 다운로드된 데이터
│   └── cache/                  # 캐시 데이터
├── portfolios/                 # 생성된 포트폴리오들
│   ├── portfolio_01_growth/
│   ├── portfolio_02_value/
│   └── ...
└── reports/                    # 백테스트 결과
    ├── portfolio_01_growth_2020-01-01_2025-07-31_20241201_143022/
    ├── portfolio_02_value_2020-01-01_2025-07-31_20241201_143022/
    └── ...
```

## 단계별 설명

### Step 1: 데이터 다운로드
- 시장 데이터(가격, 재무)를 다운로드
- 캐시 시스템으로 중복 다운로드 방지
- Mock 데이터 옵션 제공

### Step 2: 포트폴리오 생성
- 다양한 투자 목적에 따른 포트폴리오 생성
- 각 포트폴리오별 설명 및 전략 포함
- 포트폴리오 타입: Growth, Value, Balanced, Low Volatility, Quality

### Step 3: 백테스팅
- 생성된 포트폴리오에 대한 백테스트 실행
- 성과 지표 계산 (수익률, 샤프 비율, 최대 낙폭 등)
- 결과를 reports 폴더에 체계적으로 저장

## 사용법

### 전체 워크플로우 실행
```bash
# 기본 실행 (Mock 데이터 사용)
python run_workflow.py --use-mock

# 실제 데이터 사용
python run_workflow.py --start 2020-01-01 --end 2025-07-31

# 특정 포트폴리오 타입만 생성
python run_workflow.py --portfolio-types growth,value,balanced --use-mock

# 특정 포트폴리오만 백테스트
python run_workflow.py --backtest-portfolio portfolio_01_growth --use-mock
```

### 개별 단계 실행

#### Step 1: 데이터 다운로드
```bash
python step1_download_data.py --start 2020-01-01 --end 2025-07-31 --output data/downloaded/
```

#### Step 2: 포트폴리오 생성
```bash
python step2_generate_portfolios.py --data-dir data/downloaded/ --output portfolios/ --num-portfolios 5
```

#### Step 3: 백테스팅
```bash
python step3_backtest.py --portfolio portfolios/portfolio_01_growth --start 2020-01-01 --end 2025-07-31
```

## 포트폴리오 타입

### 1. Growth Portfolio (성장형)
- **목적**: 장기 성장을 추구
- **전략**: 모멘텀과 성장성을 중심으로 구성
- **팩터 가중치**: 모멘텀 60%, 가치 10%, 품질 20%, 저변동성 10%

### 2. Value Portfolio (가치형)
- **목적**: 저평가 주식에 투자하여 가치 발견
- **전략**: P/E, P/B 등 가치 지표 중심
- **팩터 가중치**: 모멘텀 20%, 가치 50%, 품질 20%, 저변동성 10%

### 3. Balanced Portfolio (균형형)
- **목적**: 위험과 수익의 균형을 추구
- **전략**: 모든 팩터를 균등하게 적용
- **팩터 가중치**: 모멘텀 40%, 가치 20%, 품질 20%, 저변동성 20%

### 4. Low Volatility Portfolio (저변동성)
- **목적**: 안정적인 수익을 추구
- **전략**: 저변동성 주식 중심으로 구성
- **팩터 가중치**: 모멘텀 20%, 가치 20%, 품질 20%, 저변동성 40%

### 5. Quality Portfolio (품질형)
- **목적**: 고품질 기업에 투자
- **전략**: ROA, ROE 등 품질 지표 중심
- **팩터 가중치**: 모멘텀 20%, 가치 10%, 품질 50%, 저변동성 20%

## 출력 파일

### 포트폴리오 디렉토리 구조
```
portfolios/portfolio_01_growth/
├── portfolio_description.md    # 포트폴리오 설명
├── weights.csv                 # 포트폴리오 가중치
└── metadata.json              # 메타데이터
```

### 백테스트 결과 디렉토리 구조
```
reports/portfolio_01_growth_2020-01-01_2025-07-31_20241201_143022/
├── backtest_summary.md         # 백테스트 요약
├── equity_curve.csv           # 수익률 곡선
├── trades.csv                 # 거래 내역
├── performance_metrics.json   # 성과 지표
├── portfolio_weights.csv      # 원본 포트폴리오 가중치
├── portfolio_description.md   # 포트폴리오 설명
└── backtest_metadata.json     # 백테스트 메타데이터
```

## 성과 지표

각 백테스트에서 계산되는 성과 지표:

- **Total Return**: 총 수익률
- **Annualized Return**: 연간 수익률
- **Volatility**: 변동성
- **Sharpe Ratio**: 샤프 비율
- **Max Drawdown**: 최대 낙폭
- **Win Rate**: 승률

## 설정 옵션

### 워크플로우 옵션
- `--start`: 시작 날짜 (YYYY-MM-DD)
- `--end`: 종료 날짜 (YYYY-MM-DD)
- `--portfolio-types`: 포트폴리오 타입 (콤마로 구분)
- `--num-portfolios`: 생성할 포트폴리오 수
- `--use-mock`: Mock 데이터 사용
- `--skip-step1`: Step 1 건너뛰기
- `--skip-step2`: Step 2 건너뛰기
- `--backtest-portfolio`: 특정 포트폴리오만 백테스트

### 백테스트 옵션
- `--rebalance-frequency`: 리밸런싱 빈도 (daily, weekly, monthly, quarterly)
- `--transaction-costs`: 거래 비용 (퍼센트)

## 예제 실행

```bash
# 1. Mock 데이터로 전체 워크플로우 실행
python run_workflow.py --use-mock --portfolio-types growth,value,balanced

# 2. 실제 데이터로 성장형 포트폴리오만 백테스트
python run_workflow.py --use-mock --portfolio-types growth --backtest-portfolio portfolio_01_growth

# 3. 개별 단계 실행
python step1_download_data.py --use-mock --output data/downloaded/
python step2_generate_portfolios.py --data-dir data/downloaded/ --output portfolios/
python step3_backtest.py --portfolio portfolios/portfolio_01_growth --use-mock
```

## 의존성

필요한 Python 패키지:
```
pandas
numpy
yfinance
pyyaml
matplotlib (선택사항)
seaborn (선택사항)
```

설치:
```bash
pip install -r requirements.txt
```

## 주의사항

1. **실제 데이터 사용 시**: API 제한 및 네트워크 상태에 따라 다운로드가 실패할 수 있습니다.
2. **Mock 데이터**: 개발 및 테스트 목적으로만 사용하세요.
3. **포트폴리오 선택**: 백테스트할 포트폴리오가 존재하는지 확인하세요.
4. **디스크 공간**: 대용량 데이터 다운로드 시 충분한 디스크 공간을 확보하세요.

## 문제 해결

### 일반적인 오류

1. **데이터 다운로드 실패**
   - 네트워크 연결 확인
   - Mock 데이터 사용 (`--use-mock`)

2. **포트폴리오 생성 실패**
   - 데이터 파일 존재 확인
   - 포트폴리오 타입 이름 확인

3. **백테스트 실패**
   - 포트폴리오 가중치 파일 존재 확인
   - 날짜 범위 확인

### 로그 확인

모든 스크립트는 상세한 로그를 출력합니다. 오류 발생 시 로그를 확인하여 문제를 진단하세요.
