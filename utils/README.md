# Utils 폴더 - 데이터 분석 및 처리 도구들

## 필요한 라이브러리

- `pandas`: 데이터 처리 및 분석
- `numpy`: 수치 계산
- `json`: JSON 파일 처리
- `tqdm`: 진행률 표시 (선택사항)
- `collections`: 데이터 구조 처리

## 파일 구조
├── utils/
│   ├── analyze_product_columns.py
│   ├── ...
├── images/
├──── farfetch/
│   ├── 1.jpg
│   ├── ...
├── articles.csv
├── transactions_train.csv
├── farfetch.json
└── ...

## 파일 목록 및 역할

### `analyze_product_columns.py`
**역할**: `articles_with_price.csv`에서 product 관련 칼럼들의 고유값 분석
- product 관련 칼럼들의 고유값 개수 및 분포 분석
- 각 칼럼별 상세 통계 생성
- 분석 결과를 CSV 파일로 저장

**생성 파일**:
- `product_columns_summary.csv`: 전체 요약
- `product_all_analysis.csv`: 통합 분석
- `product_{column_name}_details.csv`: 각 칼럼별 상세 분석

### `check_price_variations.py`
**역할**: `transactions_train.csv`에서 동일한 article_id의 가격 변동 여부 확인
- 샘플 article_id들의 가격 변동 분석
- 거래량이 많은 article_id들의 가격 패턴 확인
- 시간에 따른 가격 변화 추적

### `convert_farfetch_to_csv.py`
**역할**: `farfetch.json`을 CSV 형식으로 변환

### `create_sample_and_join.py`
**역할**: 샘플 데이터 생성 및 transactions_train.csv과 articles.csv JOIN 작업 수행
- `transactions_train.csv`에서 랜덤 샘플 10개 생성
- article_id별 정확한 평균 가격 계산
- `articles.csv`와 price 데이터를 JOIN하여 `articles_with_price.csv` 생성

**생성 파일**:
- `sample_transactions.csv`: 랜덤 샘플 거래 데이터
- `articles_with_price.csv`: 가격 정보가 포함된 상품 데이터

### `show_random_samples.py`
**역할**: 세 개의 주요 CSV 파일에서 랜덤 샘플 데이터 표시
- `articles_with_price.csv`, `styles.csv`, `farfetch.csv`에서 각각 랜덤 행 추출
- 칼럼별 값들을 보기 좋게 출력
- 데이터 구조 및 내용 확인용
