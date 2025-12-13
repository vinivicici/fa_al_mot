# Utils - 데이터 전처리 유틸리티

H&M 및 Farfetch 데이터셋 전처리를 위한 스크립트 모음

```
utils/
├── hnm_join.py                   # 1. 가격 데이터 JOIN 및 병합
├── hnm_column_drop.py             # 2. 불필요한 칼럼 제거
├── hnm_row_drop.py                # 3. 불필요한 행 제거 및 가격 스케일링
├── hnm_column_split_densify.py    # 4. product_group_name 칼럼 제거
├── hnm_one_hot_encode.py          # 5. 카테고리 칼럼 원핫인코딩
├── detail_desc_to_embedding.py   #  상세 설명 텍스트 임베딩 생성
├── convert_farfetch_to_csv.py     # Farfetch JSON → CSV 변환
└── observation/                   # 데이터 분석 도구
    ├── analyze_product_columns.py
    ├── analyze_farfetch_columns.py
    └── show_random_samples.py
```

## 🚀 메인 전처리 스크립트

### `hnm_join.py`
transactions_train.csv에서 article_id별 평균 가격 계산 후 articles.csv와 JOIN

- article_id별 평균 가격 계산 (청크 단위 처리)
- product_code별 병합 (같은 제품 다른 색상 통합)
- **출력**: `articles_with_price.csv`

### `hnm_column_drop.py`
분석에 불필요한 칼럼 제거

- 제거 대상: prod_name, article_id, *_no, *_code 등
- 25개 칼럼 to 8개 칼럼

### `hnm_row_drop.py`
불필요한 카테고리 제거

- **section_name**: 속옷, 악세서리 제거
- **product_group_name**: 신발 제외한 비의류 제거
- **garment_group_name**: 악세서리, 양말 제거

### `hnm_column_split_densify.py`
중복 정보 칼럼 제거

- product_group_name 칼럼 삭제

### `hnm_one_hot_encode.py`
카테고리 칼럼 원핫인코딩

- 대상: product_type_name, garment_group_name, index_group_name, section_name
- 4개 칼럼 → 약 121개 이진 칼럼

### 'detail_desc_to_embedding.py'
articles_with_price.csv의 제품 상세 설명(detail_desc)을 벡터 임베딩으로 변환

- SentenceTransformer (all-MiniLM-L6-v2 모델) 사용
- detail_desc 텍스트 칼럼을 384차원의 벡터(desc_embedding)로 변환
- **출력**: articles_with_embeddings.csv

## 📊 데이터 분석 도구 (observation/)

### `analyze_product_columns.py`
H&M 제품 칼럼 분석

- 각 칼럼별 고유값 개수 및 분포
- **출력**: `hnm_column_observation/` 폴더

### `analyze_farfetch_columns.py`
Farfetch 데이터 칼럼 분석

- 브랜드, 성별, 가격 등 통계
- **출력**: `farfetch_column_observation/` 폴더

### `show_random_samples.py`
데이터셋 랜덤 샘플 출력

- 3개 CSV 파일에서 샘플 추출 및 출력

## 🔧 기타

### `convert_farfetch_to_csv.py`
Farfetch JSON 데이터 변환

- `farfetch.json` to `farfetch.csv`
- 이미지 정보 파이프(|)로 연결

## 📦 필요한 라이브러리

```bash
pip install pandas numpy sentence-transformers
```

## [사용법] 사용 방법

상위 폴더의 `preprocess.py`를 실행하면 전체 전처리 파이프라인이 자동 실행됩니다.

```bash
python preprocess.py
```
