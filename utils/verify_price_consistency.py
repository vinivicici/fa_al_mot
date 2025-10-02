#!/usr/bin/env python3
"""
articles_with_price.csv에서 product_code별 price 일관성 검증
"""

import pandas as pd
from collections import defaultdict

def verify_price_consistency():
    print("=== product_code별 price 일관성 검증 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('articles_with_price.csv')
        print(f"총 데이터: {len(df):,}개 행")
        
        # product_code별 price 수집
        product_prices = defaultdict(set)
        
        for _, row in df.iterrows():
            product_code = row['product_code']
            price = row['price']
            product_prices[product_code].add(price)
        
        print(f"고유 product_code 수: {len(product_prices):,}개")
        
        # price가 일치하지 않는 product_code 찾기
        inconsistent_products = []
        for product_code, prices in product_prices.items():
            if len(prices) > 1:
                inconsistent_products.append((product_code, prices))
        
        print(f"\n=== 검증 결과 ===")
        print(f"일관된 product_code: {len(product_prices) - len(inconsistent_products):,}개")
        print(f"불일치 product_code: {len(inconsistent_products):,}개")
        
        if inconsistent_products:
            print(f"\n불일치 사례 (처음 10개):")
            for i, (product_code, prices) in enumerate(inconsistent_products[:10]):
                print(f"  {i+1}. product_code: {product_code}")
                print(f"     prices: {sorted(prices)}")
                print()
            
            if len(inconsistent_products) > 10:
                print(f"  ... 총 {len(inconsistent_products)}개의 불일치 사례")
        else:
            print("✅ 모든 product_code의 price가 일관됩니다!")
        
        # 통계 정보
        price_counts = [len(prices) for prices in product_prices.values()]
        if price_counts:
            print(f"\n=== 통계 정보 ===")
            print(f"product_code당 평균 price 개수: {sum(price_counts) / len(price_counts):.2f}")
            print(f"최대 price 개수: {max(price_counts)}")
            print(f"price 개수 분포:")
            from collections import Counter
            count_distribution = Counter(price_counts)
            for count, freq in sorted(count_distribution.items()):
                print(f"  {count}개 price: {freq:,}개 product_code")
        
        return len(inconsistent_products) == 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    is_consistent = verify_price_consistency()
    
    if is_consistent:
        print("\n🎉 검증 완료: 모든 product_code의 price가 일관됩니다!")
    else:
        print("\n⚠️ 검증 완료: 일부 product_code에서 price 불일치 발견")
