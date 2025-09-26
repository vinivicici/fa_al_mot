#!/usr/bin/env python3
"""
샘플 생성과 JOIN을 정확하게 수행하는 스크립트
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def create_sample_transactions():
    """transactions_train.csv에서 랜덤 샘플 10개 생성"""
    print("=== 1. 랜덤 샘플 생성 ===")
    
    # 전체 파일 라인 수 확인
    with open('transactions_train.csv', 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # 헤더 제외
    
    print(f"총 거래 데이터: {total_lines:,}개")
    
    # 랜덤 시드 고정
    np.random.seed(123)
    random_indices = np.random.choice(total_lines, size=10, replace=False)
    random_indices = sorted(random_indices)
    
    print(f"선택된 행 번호: {random_indices}")
    
    # 선택된 행들 추출
    sample_rows = []
    current_line = 0
    
    with open('transactions_train.csv', 'r') as f:
        header = next(f)  # 헤더 읽기
        for line in f:
            if current_line in random_indices:
                sample_rows.append(line.strip())
            current_line += 1
            if current_line > max(random_indices):
                break
    
    # DataFrame으로 변환
    columns = header.strip().split(',')
    sample_data = []
    for row in sample_rows:
        sample_data.append(row.split(','))
    
    sample_df = pd.DataFrame(sample_data, columns=columns)
    
    # 데이터 타입 변환
    sample_df['price'] = sample_df['price'].astype(float)
    
    print("\n선택된 샘플 데이터:")
    print(sample_df[['article_id', 'price']].to_string(index=False))
    
    # 저장
    sample_df.to_csv('new_sample_transactions.csv', index=False)
    print("\n샘플 데이터가 'new_sample_transactions.csv'에 저장되었습니다.")
    
    return sample_df

def calculate_article_prices():
    """transactions_train.csv에서 article_id별 정확한 평균 가격 계산"""
    print("\n=== 2. article_id별 평균 가격 계산 ===")
    
    # 메모리 효율적인 방법으로 평균 계산
    chunk_size = 1000000
    article_sums = {}
    article_counts = {}
    
    chunk_num = 0
    for chunk in pd.read_csv('transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        print(f"청크 {chunk_num} 처리 중... (크기: {len(chunk):,})")
        
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
    
    print(f"총 {len(article_avg_prices):,}개의 고유 article_id 평균 가격 계산 완료")
    
    return article_avg_prices

def join_articles_with_prices(article_avg_prices):
    """articles.csv와 가격 데이터를 JOIN"""
    print("\n=== 3. articles.csv와 가격 데이터 JOIN ===")
    
    # articles.csv 로드
    articles_df = pd.read_csv('articles.csv')
    print(f"articles.csv 로드: {len(articles_df):,}개 행")
    
    # price 칼럼 추가
    articles_df['price'] = articles_df['article_id'].map(article_avg_prices)
    
    # 결과 확인
    matched = articles_df['price'].notna().sum()
    unmatched = articles_df['price'].isna().sum()
    
    print(f"매칭 결과:")
    print(f"  - 매칭됨: {matched:,}개")
    print(f"  - 매칭 안됨: {unmatched:,}개")
    
    # 저장
    articles_df.to_csv('articles_with_price.csv', index=False)
    print("결과가 'articles_with_price.csv'에 저장되었습니다.")
    
    return articles_df

def verify_results(sample_df, article_avg_prices, articles_df):
    """결과 검증"""
    print("\n=== 4. 결과 검증 ===")
    
    print("샘플 데이터와 계산된 평균 가격 비교:")
    print("=" * 80)
    
    for _, row in sample_df.iterrows():
        article_id = int(row['article_id'])
        sample_price = float(row['price'])
        
        # 계산된 평균 가격
        if article_id in article_avg_prices:
            avg_price = article_avg_prices[article_id]
            
            # articles_with_price.csv에서 확인
            joined_row = articles_df[articles_df['article_id'] == article_id]
            if not joined_row.empty:
                joined_price = joined_row['price'].iloc[0]
                
                print(f"Article ID: {article_id}")
                print(f"  샘플 거래 가격:    {sample_price:.8f}")
                print(f"  계산된 평균 가격:  {avg_price:.8f}")
                print(f"  JOIN된 가격:      {joined_price:.8f}")
                print(f"  평균-JOIN 일치:   {'✓' if abs(avg_price - joined_price) < 1e-10 else '✗'}")
                print()
            else:
                print(f"Article ID {article_id}: articles_with_price.csv에서 찾을 수 없음")
        else:
            print(f"Article ID {article_id}: 평균 가격 계산 안됨")

def main():
    print("가격 매칭 문제 해결을 위한 정확한 처리 시작")
    print("=" * 60)
    
    # 1. 샘플 생성
    sample_df = create_sample_transactions()
    
    # 2. 평균 가격 계산
    article_avg_prices = calculate_article_prices()
    
    # 3. JOIN 수행
    articles_df = join_articles_with_prices(article_avg_prices)
    
    # 4. 결과 검증
    verify_results(sample_df, article_avg_prices, articles_df)
    
    print("\n작업 완료!")

if __name__ == "__main__":
    main()
