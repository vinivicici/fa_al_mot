# 데이터셋 병합 분석 및 전략 (업데이트)

## 데이터셋 현황 (원핫 인코딩 제거 후)

### H&M 데이터셋 (dataset/hnm/articles_with_price.csv)
- **크기**: 47,224개 행 × 5개 칼럼
- **구조**: 범주형 칼럼들 (원핫 인코딩 제거됨)
- **주요 칼럼**:
  - `product_code`: 제품 식별자
  - `price`: 가격 (스케일링됨, 평균: 0.032, 범위: 0.0004~0.507)
  - `index_group_name`: 성별/연령대 분류
  - `garment_group_name`: 의류 카테고리
  - `detail_desc`: 제품 설명

### Fashion 데이터셋 (dataset/fashion/fashion.csv)
- **크기**: 44,446개 행 × 22개 칼럼 (칼럼 드롭 전)
- **구조**: 범주형 칼럼들
- **주요 칼럼**:
  - `id`: 제품 식별자
  - `discountedPrice`: 할인가격 (평균: 1,615, 범위: 0~28,950)
  - `brandName`: 브랜드명
  - `gender`: 성별 (Men, Women, Boys, Girls, Unisex)
  - `masterCategory`: 대분류 (Apparel, Accessories, Footwear 등)
  - `subCategory`: 중분류 (Topwear, Bottomwear 등)
  - `articleType`: 세부 품목 (Tshirts, Shirts 등)
  - `usage`: 용도 (Casual, Sports, Ethnic 등)

## 매칭 분석 (새로운 관찰)

### 1. 성별/연령대 매칭

**H&M `index_group_name`** (실제 데이터):
- `Ladieswear`: 20,340개 (43.1%)
- `Baby/Children`: 14,240개 (30.2%)
- `Divided`: 6,760개 (14.3%)
- `Menswear`: 4,429개 (9.4%)
- `Sport`: 1,455개 (3.1%)

**Fashion `gender`** (실제 데이터):
- `Men`: 22,165개 (49.8%)
- `Women`: 18,632개 (41.9%)
- `Unisex`: 2,164개 (4.9%)
- `Boys`: 830개 (1.9%)
- `Girls`: 655개 (1.5%)

**매칭 전략**:
- `Ladieswear` ↔ `Women` (성인 여성)
- `Menswear` ↔ `Men` (성인 남성)
- `Baby/Children` ↔ `Boys` + `Girls` (어린이)
- `Divided` ↔ `Unisex` (성별 구분 없음)
- `Sport` ↔ `Sports` usage (스포츠 용도)

### 2. 가격 통일 전략

**가격 스케일 분석**:
- **H&M**: 평균 0.032, 범위 0.0004~0.507 (스케일링됨, 실제로는 유로 × 1000)
- **Fashion**: 평균 1,615, 범위 0~28,950 (달러 원본 가격)

**가격 통일 방법**:
1. **H&M 가격 복원**: 스케일링된 값 × 1000 = 유로
2. **환율 적용**: 유로 → 달러 변환 (예: 1 EUR = 1.1 USD)
3. **최종 통일**: 두 데이터셋 모두 달러로 통일

**예시**:
- H&M: 0.032 × 1000 × 1.1 = 35.2 USD
- Fashion: 1,615 USD (그대로)

### 3. 카테고리 매칭 (실제 데이터 기반)

**H&M `garment_group_name`** (실제 상위 카테고리들):
- `Jersey Fancy`: 8,588개 (18.2%)
- `Accessories`: 6,600개 (14.0%)
- `Knitwear`: 3,112개 (6.6%)
- `Blouses`: 3,067개 (6.5%)
- `Trousers`: 2,998개 (6.3%)
- `Shoes`: 2,677개 (5.7%)
- `Under-, Nightwear`: 2,543개 (5.4%)
- `Unknown`: 2,514개 (5.3%) - 분류되지 않은 제품들 (스포츠웨어, 아웃도어 등)
- `Dresses Ladies`: 2,507개 (5.3%)
- `Outdoor`: 2,460개 (5.2%)

**Fashion `articleType`** (실제 상위 카테고리들):
- `Tshirts`: 7,070개 (15.9%)
- `Shirts`: 3,217개 (7.2%)
- `Casual Shoes`: 2,846개 (6.4%)
- `Watches`: 2,542개 (5.7%)
- `Sports Shoes`: 2,036개 (4.6%)
- `Kurtas`: 1,844개 (4.1%)
- `Tops`: 1,762개 (4.0%)
- `Handbags`: 1,759개 (4.0%)
- `Heels`: 1,323개 (3.0%)
- `Sunglasses`: 1,073개 (2.4%)

**매칭 전략**:
- `Jersey Fancy` ↔ `Tshirts` + `Tops` (상의류)
- `Blouses` ↔ `Shirts` (셔츠류)
- `Shoes` ↔ `Casual Shoes` + `Sports Shoes` (신발류)
- `Accessories` ↔ `Watches` + `Handbags` + `Sunglasses` (액세서리)
- `Trousers` ↔ `Bottomwear` 하위 카테고리 (하의류)
- `Under-, Nightwear` ↔ `Innerwear` (속옷류)
- `Unknown` ↔ `Sports` usage (스포츠웨어, 아웃도어 제품들)

### 4. 용도 매칭 (실제 데이터 기반)

**Fashion `usage`** (실제 데이터):
- `Casual`: 34,414개 (77.4%)
- `Sports`: 4,025개 (9.1%)
- `Ethnic`: 3,208개 (7.2%)
- `Formal`: 2,359개 (5.3%)
- `Smart Casual`: 67개 (0.2%)
- `Party`: 29개 (0.1%)
- `Travel`: 26개 (0.1%)
- `Home`: 1개 (0.0%)

**H&M 매칭 전략**:
- `Sport` ↔ `Sports` usage
- 대부분의 H&M 제품 ↔ `Casual` usage
- 특정 카테고리 ↔ `Formal` usage (예: Blouses, Shirts)
- `Unknown` 카테고리 ↔ `Sports` usage (스포츠웨어, 아웃도어 제품들)

## 새로운 병합 전략 (원핫 인코딩 제거 후)

### 1단계: 데이터 정규화 및 정제
1. **Fashion 데이터 칼럼 드롭**:
   - 불필요한 칼럼들 제거 (이미지 URL, 메타데이터 등)
   - 최종 예상 칼럼: ~12개

2. **가격 통일**:
   - H&M: 스케일링된 값 × 1000 × 환율 = 달러
   - Fashion: 원본 달러 가격 유지
   - 통합된 `price_usd` 칼럼 생성

3. **공통 칼럼 생성**:
   - `dataset_source`: 'HNM' 또는 'FASHION'
   - `normalized_gender`: 통합된 성별 분류
   - `normalized_category`: 통합된 카테고리 분류
   - `price_usd`: 달러로 통일된 가격

### 2단계: 매핑 테이블 생성
```python
# 성별 매핑
gender_mapping = {
    'HNM_Ladieswear': 'Women',
    'HNM_Menswear': 'Men', 
    'HNM_Baby/Children': 'Children',
    'HNM_Divided': 'Unisex',
    'HNM_Sport': 'Sports',
    'FASHION_Women': 'Women',
    'FASHION_Men': 'Men',
    'FASHION_Girls': 'Children',
    'FASHION_Boys': 'Children',
    'FASHION_Unisex': 'Unisex'
}

# 카테고리 매핑
category_mapping = {
    'HNM_Jersey Fancy': 'Tops',
    'HNM_Blouses': 'Shirts',
    'HNM_Shoes': 'Shoes',
    'HNM_Accessories': 'Accessories',
    'HNM_Trousers': 'Bottomwear',
    'HNM_Under-, Nightwear': 'Innerwear',
    'FASHION_Tshirts': 'Tops',
    'FASHION_Shirts': 'Shirts',
    'FASHION_Casual Shoes': 'Shoes',
    'FASHION_Watches': 'Accessories'
}

# 용도 매핑
usage_mapping = {
    'HNM_Sport': 'Sports',
    'HNM_Blouses': 'Formal',
    'HNM_Shirts': 'Formal',
    'HNM_Unknown': 'Casual',
    'FASHION_Sports': 'Sports',
    'FASHION_Ethnic': 'Ethnic',
    'FASHION_Formal': 'Formal',
    'FASHION_Casual': 'Casual'
}
```

### 3단계: 통합 데이터셋 구축
1. **공통 스키마 정의**:
   - 식별자: `product_id`, `dataset_source`
   - 가격: `price_usd` (달러로 통일)
   - 성별: `normalized_gender` (범주형)
   - 카테고리: `normalized_category` (범주형)
   - 용도: `normalized_usage` (범주형)
   - 설명: `description`

2. **데이터 병합**:
   - 수직 병합 (concatenation)
   - 공통 칼럼만 유지
   - 데이터셋 소스 구분
   - 최종 원핫 인코딩 적용 (선택사항)

## 예상 결과 (업데이트)

### 통합 데이터셋 규모
- **총 행 수**: ~91,670개 (47,224 + 44,446)
- **총 칼럼 수**: ~8개 (범주형 유지) 또는 ~50개 (원핫 인코딩 적용)
- **메모리 사용량**: ~200MB (범주형) 또는 ~500MB (원핫)

### 장점
1. **데이터 규모 확대**: 두 배의 데이터로 모델 성능 향상
2. **카테고리 다양성**: 더 다양한 제품 카테고리 커버
3. **가격 범위 확장**: 다양한 가격대의 제품 포함
4. **구조 단순화**: 원핫 인코딩 제거로 더 직관적인 데이터 구조

### 도전과제
1. **가격 통일**: H&M(유로) → 달러 변환 필요 (환율 적용)
2. **카테고리 불일치**: 완벽한 1:1 매칭 어려움
3. **데이터 품질**: 서로 다른 데이터 소스의 품질 차이
4. **Unknown 카테고리**: H&M에 2,514개의 분류되지 않은 제품들 (스포츠웨어, 아웃도어 등)

## 다음 단계 (우선순위)

1. **Fashion 데이터 전처리**: 칼럼 드롭 실행
2. **가격 통일**: H&M 가격을 유로로 복원 후 달러로 변환
3. **매핑 테이블 구현**: 카테고리 매핑 로직 개발
4. **통합 스크립트 작성**: 자동화된 병합 파이프라인
5. **품질 검증**: 병합 후 데이터 무결성 확인
6. **성능 테스트**: 메모리 사용량 및 처리 속도 최적화
