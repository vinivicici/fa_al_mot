files:
1. h&m
2. fashion images

flow
1) h&m dataset 합치기(transactions_train.csv, articles.csv)
2) column selection
3) fashion images를 병합

---------------------------------------------------------------------
1. h&m dataset 합치기
- transactions_train.csv에선 price만 필요
- 어떠한 article에 대해 가격이 다른 경우가 있음 -> 평균 가격 채택 (과한 세일 아래쪽 이상치 커팅 사용)
Article ID 568601007 (가장 극단적인 케이스):
총 8,446개 거래
350개의 서로 다른 가격!
최저: 0.01354237, 최고: 0.05083051 (약 3.7배 차이)
평균: 0.04556165

3. column list정리

## 실제 데이터 샘플

### 1. articles_with_price.csv 샘플 (행 15795/105542)
```
article_id: 567992028
product_code: 567992
prod_name: PRIA
product_type_no: 255
product_type_name: T-shirt
product_group_name: Garment Upper body
graphical_appearance_no: 1010008
graphical_appearance_name: Front print
colour_group_code: 81
colour_group_name: Light Turquoise
perceived_colour_value_id: 3
perceived_colour_value_name: Light
perceived_colour_master_id: 7
perceived_colour_master_name: Turquoise
department_no: 8716
department_name: Young Girl Jersey Fancy
index_code: I
index_name: Children Sizes 134-170
index_group_no: 4
index_group_name: Baby/Children
section_no: 77
section_name: Young Girl
garment_group_no: 1005
garment_group_name: Jersey Fancy
detail_desc: Short-sleeved top in printed cotton jersey.
price: 0.0067627118644067
```
---> 가격에 대한 추측, 또한 이미지가 벡터화돼서 들어가므로 graphical~이 필요없을듯함.
---> Children, teen, adult 정도(가칭) 해서 연령대 분류가 가능할듯 (새로운 범주 필요), 밑의 데이터셋에 추가로 이미지 CNN돌려서 칼럼 만들기
---> Prod_name은 의미가 없는 것 같음 (PRIA이러고있어서..)
---> Product_type_name, product_group_name은 어떤 고유값이 있는지 따로 정리해놓음.(*_details.csv파일들) 고유값 개수가 131, 250개 이래서 통합이 필요할듯함.
### 2. styles.csv 샘플 (행 랜덤/44424)
```
id: 12532
gender: Women
masterCategory: Apparel
subCategory: Topwear
articleType: Jackets
baseColour: Black
season: Fall
year: 2011.0
usage: Casual
productDisplayName: Puma Women Solid Black Jackets
```

4. 칼럼 셀렉션 (utils/hnm_column_drop.py 방식 참고)

제거할 칼럼 목록(실제로 존재하는 컬럼만 제거, 미존재 컬럼은 무시):
  - prod_name
  - article_id
  - product_type_no
  - graphical_appearance_no
  - graphical_appearance_name
  - colour_group_code
  - colour_group_name
  - perceived_colour_value_id
  - perceived_colour_value_name
  - perceived_colour_master_id
  - perceived_colour_master_name
  - department_no
  - department_name
  - index_code
  - index_name
  - index_group_no
  - section_no
  - garment_group_no

제거 후 남는 주요 칼럼 예시:
  product_code          예: 567992
  product_type_name     예: T-shirt
  product_group_name    예: Garment Upper body
  index_group_name      예: Baby/Children
  section_name          예: Young Girl
  garment_group_name    예: Jersey Fancy
  detail_desc           예: Short-sleeved top in printed cotton jersey.
  price                 예: 0.0067627118644067




### Fashion 데이터셋 (dataset/fashion/fashion.csv)
**데이터 규모**: 44,446개 행
:
- `id`: 제품 ID
- `discountedPrice`: 할인가격
- `brandName`: 브랜드명
- `gender`: 성별 (Men, Women, Boys, Girls, Unisex) - 5개 고유값
- `ageGroup`: 연령대 (Kids-Girls, Adults-Men 등)
- `masterCategory`: 대분류 (Apparel, Accessories, Footwear, Personal Care, Free Items) - 7개 고유값
- `subCategory`: 중분류 (Topwear, Shoes, Bags, Bottomwear, Watches 등) - 45개 고유값
- `articleType`: 세부 품목 (Tshirts, Shirts, Casual Shoes, Watches, Sports Shoes 등) - 143개 고유값
- `usage`: 용도 (Casual, Sports, Ethnic, Formal, Smart Casual 등) - 8개 고유값
- `fashionType`: 패션 타입
- `styleType`: 스타일 타입
- `description`: 제품 설명 (HTML 태그 제거, 500자 제한)

**제거된 칼럼들**:
- 이미지 URL들 (불필요)
- 원가격 (`price`)
- 평점 (`myntraRating`)
- 배송/결제 관련 정보 (`codEnabled`, `isEMIEnabled` 등)
- 물리적 속성 (`weight`, `vat` 등)
- 색상 정보 (`baseColour`, `colour1`, `colour2`)
- 시즌/연도 (`season`, `year`)
- 제품 식별자 (`variantName`, `articleNumber`, `displayCategories`, `productDisplayName`)


