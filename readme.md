## 📁 파일 구조

```
dataset/
├── hnm/
│   ├── articles.csv                 # 원본 H&M 기사 메타데이터
│   ├── transactions_train.csv       # 원본 거래 데이터
│   └── articles_with_price.csv      # 전처리 결과(점진적 갱신 대상)
│
└── fashion/                         # 두 번째 데이터셋
    ├── styles.csv                   # 메타 정보
    ├── images.csv                   # (있다면) 이미지 인덱스/매핑
    └── styles/                      # (있다면) 개별 JSON/메타 파일 폴더

utils/
├── hnm_join.py                      # 1. 가격 JOIN 및 product_code 병합 (입출력: dataset/hnm)
├── hnm_column_drop.py               # 2. 불필요 칼럼 제거 (입출력: dataset/hnm)
├── hnm_row_drop.py                  # 3. 불필요 행 제거 + 가격 스케일 (입출력: dataset/hnm)
├── hnm_column_split_densify.py      # 4. product_group_name 제거 (입출력: dataset/hnm)
├── hnm_one_hot_encode.py            # 5. 카테고리 원핫인코딩 (입출력: dataset/hnm)
├── convert_farfetch_to_csv.py       # Farfetch JSON → CSV 변환 (입출력: dataset/hnm)
└── observation/                     # 데이터 분석 도구
    ├── analyze_product_columns.py
    ├── analyze_farfetch_columns.py
    └── show_random_samples.py
```

- 모든 전처리 스크립트는 `dataset/hnm` 경로를 기준으로 입출력을 수행합니다.
- 실행 전 CSV/JSON 파일을 `dataset/hnm`으로 옮겨주세요.