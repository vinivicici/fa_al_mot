files:
1. h&m
2. farfetch
3. fashion images

flow
1) h&m dataset 합치기(transactions_train.csv, articles.csv)
2) column selection

---------------------------------------------------------------------
1. h&m dataset 합치기
- transactions_train.csv에선 price만 필요
- 어떠한 article에 대해 가격이 다른 경우가 있음 -> 평균 가격 채택 (이 부분은 이상치 커팅등 다른 데이터 처리 방식을 사용해도 좋을 것 같음)
Article ID 568601007 (가장 극단적인 케이스):
총 8,446개 거래
350개의 서로 다른 가격!
최저: 0.01354237, 최고: 0.05083051 (약 3.7배 차이)
평균: 0.04556165

2. farfetch
.json이라 .csv로 변환

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

### 3. farfetch.csv 샘플 (행 92/141)
```
url: https://www.farfetch.com/shopping/men/dsquared2-high-shine-lace-up-shoes-item-16654437.aspx
title: high-shine lace-up shoes
brand: Dsquared2
price: 775.0
currency: USD
formatted_price: $775
availability: In Stock
item_id: 16654437
sku: 16654437
condition: NewCondition
images: https://cdn-images.farfetch-contents.com/16/65/44/37/16654437_32716111_1000.jpg, ...
description: Looking great from head to toes is easy with Dsquared2. In a high-shine finish, these lace-up shoes are a simple yet refined option for your next formal event. Smarten up.
breadcrumbs: Men, Dsquared2, Shoes
gender: Men
details: [HTML 상세 정보 - 매우 긴 내용]
uniq_id: 84fbee69-8de5-56c6-8132-ab9799cc2029
scraped_at: 14/10/2022 11:58:30
image_file: images/16654437_0.jpg | images/16654437_1.jpg | images/16654437_2.jpg | images/16654437_3.jpg
```

4. 칼럼 셀렉션
제거할 칼럼들 (15개):
  - prod_name
  - article_id
  - product_type_no
  - graphical_appearance_no
  - colour_group_code
  - colour_group_name
  - perceived_colour_value_id
  - perceived_colour_value_name
  - perceived_colour_master_id
  - perceived_colour_master_name
  - department_no
  - index_code
  - index_group_no
  - section_no
  - garment_group_no
  - graphical_appearance_name
  - department_name
  - index_name
**SIZE 추가 드랍**
**one-hot vector 빨리 할 것**
**img file name을 클러스터링할때 
  남은 칼럼들 (제거 후):
    product_code: 567992
    product_type_name: T-shirt
    product_group_name: Garment Upper body
    index_group_name: Baby/Children
    section_name: Young Girl
    garment_group_name: Jersey Fancy
    detail_desc: Short-sleeved top in printed cotton jersey.
    price: 0.0067627118644067

이상치 커팅: 아직 수행 X

문제1: garment_group_name, product_type_name이 서로 스타일을 나타내서 비슷하고, index_group_name, section_name 이 착용자의 성별, 나이를 나타냄. 
중분류-소분류 관계인데 어떻게 처리할지(하나가 다른 하나에 포함관계)
예를 들어, 여성복 - 여성 중간 가격대 시리즈
만약 중분류만 사용하면 모델 성능이 처참할 가능성이 높고, 소분류를 사용하면 다른 데이터셋이랑 병합이 거의 불가능에 가까워짐
우선은 다 놔두고 원핫인코딩으로 처리만 해놓음