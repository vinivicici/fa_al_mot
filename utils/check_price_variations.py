#!/usr/bin/env python3
"""
transactions_train.csv에서 한 article_id에 대해 다른 price가 있는지 확인하는 스크립트
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def check_price_variations():
    print("=== article_id별 가격 변동 확인 ===")
    
    # 샘플 article_id들 먼저 확인
    sample_df = pd.read_csv('new_sample_transactions.csv')
    sample_article_ids = sample_df['article_id'].tolist()
    
    print(f"샘플 article_id들: {sample_article_ids}")
    print()
    
    # 각 article_id별 가격 정보 수집
    article_prices = defaultdict(list)
    article_dates = defaultdict(list)
    
    chunk_size = 1000000
    chunk_num = 0
    
    print("transactions_train.csv 전체 스캔 중...")
    
    for chunk in pd.read_csv('transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        if chunk_num % 5 == 0:
            print(f"청크 {chunk_num} 처리 중...")
        
        # 샘플 article_id들만 필터링
        sample_chunk = chunk[chunk['article_id'].isin(sample_article_ids)]
        
        if not sample_chunk.empty:
            for _, row in sample_chunk.iterrows():
                article_id = row['article_id']
                price = row['price']
                date = row['t_dat']
                
                article_prices[article_id].append(price)
                article_dates[article_id].append(date)
    
    print(f"\n총 {chunk_num}개 청크 처리 완료")
    print("\n=== 샘플 article_id들의 가격 변동 분석 ===")
    
    for article_id in sample_article_ids:
        if article_id in article_prices:
            prices = article_prices[article_id]
            dates = article_dates[article_id]
            
            unique_prices = list(set(prices))
            unique_prices.sort()
            
            print(f"\nArticle ID: {article_id}")
            print(f"  총 거래 수: {len(prices):,}개")
            print(f"  고유 가격 수: {len(unique_prices)}개")
            print(f"  최저 가격: {min(prices):.8f}")
            print(f"  최고 가격: {max(prices):.8f}")
            print(f"  평균 가격: {np.mean(prices):.8f}")
            print(f"  가격 표준편차: {np.std(prices):.8f}")
            
            if len(unique_prices) > 1:
                print(f"  🔍 가격 변동 있음!")
                print(f"  고유 가격들: {unique_prices}")
                
                # 날짜별 가격 변화 확인 (처음 10개만)
                price_date_pairs = list(zip(prices, dates))
                price_date_pairs.sort(key=lambda x: x[1])  # 날짜순 정렬
                
                print(f"  시간순 가격 변화 (처음 10개):")
                for i, (price, date) in enumerate(price_date_pairs[:10]):
                    print(f"    {date}: {price:.8f}")
                if len(price_date_pairs) > 10:
                    print(f"    ... (총 {len(price_date_pairs)}개)")
            else:
                print(f"  ✅ 가격 변동 없음 (항상 {unique_prices[0]:.8f})")
        else:
            print(f"\nArticle ID {article_id}: 거래 데이터 없음")

def check_random_articles_for_variations():
    """랜덤한 article_id들도 확인해보기"""
    print("\n\n=== 랜덤 article_id들의 가격 변동 확인 ===")
    
    # 첫 번째 청크에서 몇 개 article_id 선택
    first_chunk = pd.read_csv('transactions_train.csv', nrows=100000)
    
    # 거래가 많은 article_id들 선택 (가격 변동 가능성이 높음)
    article_counts = first_chunk['article_id'].value_counts()
    top_articles = article_counts.head(5).index.tolist()
    
    print(f"거래량이 많은 article_id들: {top_articles}")
    
    article_price_info = {}
    
    chunk_size = 1000000
    chunk_num = 0
    
    for chunk in pd.read_csv('transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        if chunk_num % 10 == 0:
            print(f"청크 {chunk_num} 처리 중...")
        
        # 선택된 article_id들만 필터링
        filtered_chunk = chunk[chunk['article_id'].isin(top_articles)]
        
        if not filtered_chunk.empty:
            for article_id in top_articles:
                article_data = filtered_chunk[filtered_chunk['article_id'] == article_id]
                if not article_data.empty:
                    if article_id not in article_price_info:
                        article_price_info[article_id] = []
                    article_price_info[article_id].extend(article_data['price'].tolist())
    
    print(f"\n총 {chunk_num}개 청크 처리 완료")
    
    for article_id, prices in article_price_info.items():
        unique_prices = list(set(prices))
        unique_prices.sort()
        
        print(f"\nArticle ID: {article_id}")
        print(f"  총 거래 수: {len(prices):,}개")
        print(f"  고유 가격 수: {len(unique_prices)}개")
        print(f"  최저 가격: {min(prices):.8f}")
        print(f"  최고 가격: {max(prices):.8f}")
        print(f"  평균 가격: {np.mean(prices):.8f}")
        
        if len(unique_prices) > 1:
            print(f"  🔍 가격 변동 있음!")
            if len(unique_prices) <= 10:
                print(f"  모든 고유 가격: {unique_prices}")
            else:
                print(f"  고유 가격 (처음 10개): {unique_prices[:10]}")
                print(f"  ... 총 {len(unique_prices)}개의 서로 다른 가격")
        else:
            print(f"  ✅ 가격 변동 없음")

def main():
    print("한 article_id에 대한 가격 변동 여부 확인")
    print("=" * 60)
    
    # 1. 샘플 데이터의 article_id들 확인
    check_price_variations()
    
    # 2. 거래량이 많은 랜덤 article_id들도 확인
    check_random_articles_for_variations()
    
    print("\n" + "=" * 60)
    print("결론: 같은 article_id라도 시간이나 조건에 따라 다른 가격으로 판매될 수 있음")
    print("따라서 평균 가격을 계산하여 JOIN하는 것이 합리적인 접근법입니다.")

if __name__ == "__main__":
    main()
