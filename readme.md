## [구조] 파일 구조

```
dataset/
├── hnm/
│   ├── articles.csv                 # 원본 H&M 기사 메타데이터
│   ├── transactions_train.csv       # 원본 거래 데이터
│   └── articles_with_price.csv      # 전처리 결과(점진적 갱신 대상)
│
└── fashion/                         # 두 번째 데이터셋
    ├── styles.csv                   
    ├── images.csv                   
    └── styles/                      

utils/
├── hnm/                             # H&M 데이터 전처리
│   ├── hnm_join.py                  # 1. 가격 JOIN 및 product_code 병합
│   ├── hnm_column_drop.py           # 2. 불필요 칼럼 제거
│   ├── hnm_row_drop.py              # 3. 불필요 행 제거 + 가격 스케일
│   ├── hnm_column_split_densify.py  # 4. product_group_name 제거
│   └── hnm_one_hot_encode.py        # 5. 카테고리 원핫인코딩
├── fashion/                         # Fashion 데이터 전처리
│   ├── fashion_build_csv.py         # JSON to CSV 통합
│   └── column_drop.py               # 메타데이터 칼럼 제거
└── observation/                     # 데이터 분석 도구
    ├── analyze_product_columns.py
    ├── analyze_farfetch_columns.py
    └── show_random_samples.py
```

- H&M 전처리 스크립트는 `dataset/hnm` 경로를 기준으로 입출력을 수행합니다.
- Fashion 전처리 스크립트는 `dataset/fashion` 경로를 기준으로 입출력을 수행합니다.
- 실행 전 CSV/JSON 파일을 해당 dataset 폴더로 옮겨주세요.