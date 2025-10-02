#!/usr/bin/env python3
"""
product_code별로 평균 가격을 계산하여 articles_with_price.csv 수정
"""

import pandas as pd
from collections import defaultdict

def calculate_product_code_prices():
    """transactions_train.csv에서 product_code별 평균 가격 계산"""
    print("=== product_code별 평균 가격 계산 ===")
    
    # articles.csv에서 product_code와 article_id 매핑 정보 가져오기
    print("articles.csv 로딩 중...")
    articles_df = pd.read_csv('articles.csv')
    print(f"articles.csv 로드: {len(articles_df):,}개 행")
    
    # article_id -> product_code 매핑 생성
    article_to_product = dict(zip(articles_df['article_id'], articles_df['product_code']))
    print(f"article_id -> product_code 매핑 생성: {len(article_to_product):,}개")
    
    # product_code별 가격 수집
    product_sums = defaultdict(float)
    product_counts = defaultdict(int)
    
    chunk_size = 1000000
    chunk_num = 0
    
    print("transactions_train.csv 처리 중...")
    for chunk in pd.read_csv('transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        if chunk_num % 5 == 0:
            print(f"청크 {chunk_num} 처리 중... (크기: {len(chunk):,})")
        
        for _, row in chunk.iterrows():
            article_id = row['article_id']
            price = row['price']
            
            # article_id에 해당하는 product_code 찾기
            if article_id in article_to_product:
                product_code = article_to_product[article_id]
                product_sums[product_code] += price
                product_counts[product_code] += 1
    
    # product_code별 평균 가격 계산
    product_avg_prices = {}
    for product_code in product_sums:
        product_avg_prices[product_code] = product_sums[product_code] / product_counts[product_code]
    
    print(f"총 {len(product_avg_prices):,}개의 고유 product_code 평균 가격 계산 완료")
    
    return product_avg_prices

def update_articles_with_price(product_avg_prices):
    """articles_with_price.csv를 product_code별 평균 가격으로 업데이트"""
    print("\n=== articles_with_price.csv 업데이트 ===")
    
    # articles_with_price.csv 로드
    df = pd.read_csv('articles_with_price.csv')
    print(f"원본 데이터: {len(df):,}개 행")
    
    # product_code별로 평균 가격 적용
    df['price'] = df['product_code'].map(product_avg_prices)
    
    # 매칭 결과 확인
    matched = df['price'].notna().sum()
    unmatched = df['price'].isna().sum()
    
    print(f"매칭 결과:")
    print(f"  - 매칭됨: {matched:,}개")
    print(f"  - 매칭 안됨: {unmatched:,}개")
    
    # 저장
    df.to_csv('articles_with_price.csv', index=False)
    print("✅ articles_with_price.csv 업데이트 완료!")
    
    return df

def verify_consistency():
    """product_code별 price 일관성 재검증"""
    print("\n=== 일관성 재검증 ===")
    
    df = pd.read_csv('articles_with_price.csv')
    product_prices = {}
    
    for _, row in df.iterrows():
        product_code = row['product_code']
        price = row['price']
        
        if product_code in product_prices:
            if product_prices[product_code] != price:
                print(f"❌ 불일치 발견: product_code {product_code}")
                print(f"  기존: {product_prices[product_code]}")
                print(f"  새로운: {price}")
                return False
        else:
            product_prices[product_code] = price
    
    print("✅ 모든 product_code의 price가 일관됩니다!")
    return True

def main():
    print("product_code별 평균 가격으로 articles_with_price.csv 수정")
    print("=" * 60)
    
    # 1. product_code별 평균 가격 계산
    product_avg_prices = calculate_product_code_prices()
    
    # 2. articles_with_price.csv 업데이트
    updated_df = update_articles_with_price(product_avg_prices)
    
    # 3. 일관성 검증
    is_consistent = verify_consistency()
    
    if is_consistent:
        print("\n🎉 수정 완료: 모든 product_code의 price가 일관됩니다!")
    else:
        print("\n⚠️ 수정 후에도 불일치가 발견되었습니다.")

if __name__ == "__main__":
    main()
