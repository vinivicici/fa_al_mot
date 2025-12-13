# 데이터셋 병합 도구

H&M과 Fashion 데이터셋을 통합하는 자동화된 파이프라인입니다.

## 파일 구조

```
utils/dataset_join/
├── dataset_join.md          # 병합 전략 및 분석 문서
├── main_pipeline.py         # 메인 파이프라인 (전체 자동화)
├── price_unification.py     # 가격 통일 스크립트
├── category_mapping.py      # 카테고리/성별 매핑 스크립트
└── dataset_merge.py        # 데이터셋 병합 스크립트
```

## 사용법

### 1. 전체 파이프라인 실행 (권장)

```bash
python utils/dataset_join/main_pipeline.py
```

이 명령어는 다음 단계를 자동으로 실행합니다:
1. 전제 조건 확인
2. 가격 통일 (H&M 유로→달러 변환)
3. 카테고리/성별 매핑
4. 데이터셋 병합

### 2. 개별 스크립트 실행

각 단계를 개별적으로 실행할 수도 있습니다:

```bash
# 1단계: 가격 통일
python utils/dataset_join/price_unification.py

# 2단계: 카테고리/성별 매핑
python utils/dataset_join/category_mapping.py

# 3단계: 데이터셋 병합
python utils/dataset_join/dataset_merge.py
```

## 전제 조건

다음 파일들이 존재해야 합니다:
- `dataset/hnm/articles_with_price.csv` (H&M 전처리 완료)
- `dataset/fashion/fashion.csv` (Fashion 전처리 완료)

## 출력 파일

### 중간 파일들
- `dataset/hnm/articles_with_price_unified.csv` - H&M 가격 통일 완료
- `dataset/fashion/fashion_pricied.csv` - Fashion 가격 통일 완료
- `dataset/hnm/articles_with_price_mapped.csv` - H&M 매핑 완료
- `dataset/fashion/fashion_mapped.csv` - Fashion 매핑 완료

### 최종 파일들
- `dataset/merged_dataset.csv` - 통합된 최종 데이터셋
- `dataset/merge_summary.txt` - 병합 결과 요약

## 통합된 데이터셋 구조

최종 데이터셋은 다음 칼럼들을 포함합니다:

| 칼럼명 | 설명 | 예시 |
|--------|------|------|
| `product_id` | 제품 식별자 | 12345 |
| `dataset_source` | 데이터셋 소스 | HNM, FASHION |
| `price_usd` | 달러로 통일된 가격 | 35.20 |
| `normalized_gender` | 통일된 성별 | Women, Men, Children, Unisex, Sports |
| `normalized_category` | 통일된 카테고리 | Tops, Shirts, Shoes, Accessories, etc. |
| `normalized_usage` | 통일된 용도 | Casual, Sports, Formal, Ethnic, etc. |
| `description` | 제품 설명 | "Comfortable cotton t-shirt" |
| `original_gender` | 원본 성별 정보 | Ladieswear, Men, etc. |
| `original_category` | 원본 카테고리 정보 | Jersey Fancy, Tshirts, etc. |

## 매핑 규칙

### 성별 매핑
- `Ladieswear` → `Women`
- `Menswear` → `Men`
- `Baby/Children` → `Children`
- `Divided` → `Unisex`
- `Sport` → `Sports`

### 카테고리 매핑
- `Jersey Fancy` → `Tops`
- `Blouses` → `Shirts`
- `Shoes` → `Shoes`
- `Accessories` → `Accessories`
- `Unknown` → `Sports` (스포츠웨어, 아웃도어 제품들)

### 용도 매핑
- `Sport` → `Sports`
- `Blouses/Shirts` → `Formal`
- `Unknown` → `Sports`
- Fashion의 `usage` 칼럼 그대로 사용

## 가격 통일

- **H&M**: 스케일링된 값 × 678.5
- **Fashion**: 원본 가격 × 0.011

## 문제 해결

### 일반적인 오류
1. **파일을 찾을 수 없음**: 전제 조건 파일들이 존재하는지 확인
2. **인코딩 오류**: Windows 환경에서 발생할 수 있음 (자동 처리됨)
3. **메모리 부족**: 대용량 데이터 처리 시 발생할 수 있음

### 로그 확인
각 스크립트는 상세한 로그를 출력하므로, 오류 발생 시 로그를 확인하세요.

## 추가 정보

자세한 병합 전략과 분석은 `dataset_join.md` 파일을 참조하세요.
