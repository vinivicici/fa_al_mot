#!/usr/bin/env python3
"""
articles.csv에 가격 데이터를 JOIN하고 product_code별로 병합하는 스크립트
"""

import pandas as pd

def calculate_article_prices():
    """transactions_train.csv에서 article_id별 평균 가격 계산"""
    print("=== 1. article_id별 평균 가격 계산 ===")
    
    # 메모리 효율적인 방법으로 평균 계산
    chunk_size = 1000000
    article_sums = {}
    article_counts = {}
    
    chunk_num = 0
    for chunk in pd.read_csv('dataset/hnm/transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        print(f"  청크 {chunk_num} 처리 중... (크기: {len(chunk):,})")
        
        # article_id별 합계와 개수 계산
        grouped = chunk.groupby('article_id')['price'].agg(['sum', 'count'])
        
        for article_id, (price_sum, count) in grouped.iterrows():
            if article_id in article_sums:
                article_sums[article_id] += price_sum
                article_counts[article_id] += count
            else:
                article_sums[article_id] = price_sum
                article_counts[article_id] = count
    
    # 평균 계산
    article_avg_prices = {}
    for article_id in article_sums:
        article_avg_prices[article_id] = article_sums[article_id] / article_counts[article_id]
    
    print(f"[완료] 총 {len(article_avg_prices):,}개의 고유 article_id 평균 가격 계산 완료")
    
    return article_avg_prices

def join_articles_with_prices(article_avg_prices):
    """articles.csv와 가격 데이터를 JOIN하고 product_code별로 병합"""
    print("\n=== 2. articles.csv와 가격 데이터 JOIN ===")
    
    # articles.csv 로드
    articles_df = pd.read_csv('dataset/hnm/articles.csv')
    print(f"  articles.csv 로드: {len(articles_df):,}개 행")
    
    # price 칼럼 추가
    articles_df['price'] = articles_df['article_id'].map(article_avg_prices)
    
    # 결과 확인
    matched = articles_df['price'].notna().sum()
    unmatched = articles_df['price'].isna().sum()
    
    print(f"  매칭 결과:")
    print(f"    - 매칭됨: {matched:,}개")
    print(f"    - 매칭 안됨: {unmatched:,}개")
    
    return articles_df

def merge_by_product_code(articles_df):
    """product_code별로 병합하여 가격을 평균"""
    print("\n=== 3. product_code별 병합 ===")
    
    # product_code 칼럼 확인
    if 'product_code' not in articles_df.columns:
        print("[오류] 'product_code' 칼럼이 존재하지 않습니다.")
        return articles_df
    
    # 원본 product_code 개수
    unique_products_before = articles_df['product_code'].nunique()
    total_rows_before = len(articles_df)
    
    print(f"  병합 전:")
    print(f"    - 전체 행 수: {total_rows_before:,}")
    print(f"    - 고유 product_code: {unique_products_before:,}")
    
    # product_code별 그룹화 및 집계
    print(f"  병합 중...")
    
    # 가격이 있는 행만 사용하여 평균 계산
    agg_dict = {}
    
    # 숫자형 칼럼들은 평균
    numeric_cols = articles_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col != 'article_id' and col != 'product_code':  # article_id는 제외
            agg_dict[col] = 'mean'
    
    # price는 명시적으로 평균
    if 'price' in articles_df.columns:
        agg_dict['price'] = 'mean'
    
    # 문자형 칼럼들은 첫 번째 값
    text_cols = articles_df.select_dtypes(include=['object']).columns.tolist()
    for col in text_cols:
        if col != 'product_code':  # product_code는 groupby key
            agg_dict[col] = 'first'
    
    # 그룹화 수행
    merged_df = articles_df.groupby('product_code', as_index=False).agg(agg_dict)
    
    # 결과 확인
    unique_products_after = len(merged_df)
    
    print(f"  병합 후:")
    print(f"    - 전체 행 수: {unique_products_after:,}")
    print(f"    - 고유 product_code: {unique_products_after:,}")
    print(f"    - 제거된 행 수: {total_rows_before - unique_products_after:,}")
    
    # 가격 통계
    if 'price' in merged_df.columns:
        print(f"\n  가격 통계:")
        print(f"    - 평균: {merged_df['price'].mean():.2f}")
        print(f"    - 중간값: {merged_df['price'].median():.2f}")
        print(f"    - 최소: {merged_df['price'].min():.2f}")
        print(f"    - 최대: {merged_df['price'].max():.2f}")
        print(f"    - 결측값: {merged_df['price'].isna().sum():,}개")
    
    # 저장
    output_file = 'dataset/hnm/articles_with_price.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n[완료] 병합 완료! 결과가 '{output_file}'에 저장되었습니다.")
    
    return merged_df

def main():
    print("=" * 80)
    print("articles.csv + 가격 데이터 JOIN + product_code별 병합")
    print("=" * 80)
    
    # 1. article_id별 평균 가격 계산
    article_avg_prices = calculate_article_prices()
    
    # 2. articles.csv와 JOIN
    articles_df = join_articles_with_prices(article_avg_prices)
    
    # 3. product_code별로 병합
    merged_df = merge_by_product_code(articles_df)
    
    print("\n" + "=" * 80)
    print("[완료] 모든 작업 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
