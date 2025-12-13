#!/usr/bin/env python3
"""
같은 product_id를 가진 제품들의 가격 변동 분석
값의 개수 5개 이상이고 최저/최대값 차이가 심한 케이스 찾기
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_transaction_data():
    """transactions_train.csv에서 데이터 로드"""
    print("=== 1. 거래 데이터 로드 ===")
    
    # 메모리 효율적인 방법으로 데이터 로드
    chunk_size = 1000000
    all_data = []
    
    chunk_num = 0
    for chunk in pd.read_csv('dataset/hnm/transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        print(f"  청크 {chunk_num} 처리 중... (크기: {len(chunk):,})")
        
        # 필요한 칼럼만 선택
        chunk_filtered = chunk[['article_id', 'price']].copy()
        all_data.append(chunk_filtered)
    
    # 데이터 병합
    df = pd.concat(all_data, ignore_index=True)
    print(f"  총 거래 데이터: {len(df):,}개 행")
    
    return df

def load_article_data():
    """articles.csv에서 제품 정보 로드"""
    print("\n=== 2. 제품 정보 로드 ===")
    
    articles_df = pd.read_csv('dataset/hnm/articles.csv')
    print(f"  제품 데이터: {len(articles_df):,}개 행")
    
    # product_code와 article_id 매핑
    product_mapping = articles_df[['article_id', 'product_code']].copy()
    
    return product_mapping

def analyze_price_variation(transactions_df, product_mapping):
    """같은 product_id를 가진 제품들의 가격 변동 분석"""
    print("\n=== 3. 가격 변동 분석 ===")
    
    # 거래 데이터에 product_code 추가
    df_with_product = transactions_df.merge(product_mapping, on='article_id', how='left')
    
    # product_code별 가격 통계 계산
    print("  product_code별 가격 통계 계산 중...")
    
    price_stats = df_with_product.groupby('product_code')['price'].agg([
        'count',      # 거래 횟수
        'min',        # 최저가
        'max',        # 최고가
        'mean',       # 평균가
        'std',        # 표준편차
        'median'      # 중간값
    ]).reset_index()
    
    # 조건 필터링: 거래 횟수 5개 이상
    price_stats = price_stats[price_stats['count'] >= 5].copy()
    
    # 가격 변동률 계산
    price_stats['price_range'] = price_stats['max'] - price_stats['min']
    price_stats['price_ratio'] = price_stats['max'] / price_stats['min']
    price_stats['cv'] = price_stats['std'] / price_stats['mean']  # 변동계수
    
    print(f"  거래 횟수 5개 이상인 product_code: {len(price_stats):,}개")
    
    return price_stats, df_with_product

def find_extreme_cases(price_stats):
    """극단적인 가격 변동 케이스 찾기"""
    print("\n=== 4. 극단적인 가격 변동 케이스 찾기 ===")
    
    # 가격 비율 기준으로 정렬 (최고가/최저가)
    extreme_by_ratio = price_stats.nlargest(20, 'price_ratio')
    
    # 가격 범위 기준으로 정렬 (최고가 - 최저가)
    extreme_by_range = price_stats.nlargest(20, 'price_range')
    
    # 변동계수 기준으로 정렬
    extreme_by_cv = price_stats.nlargest(20, 'cv')
    
    print("  가격 비율 기준 상위 20개 (최고가/최저가):")
    for i, row in extreme_by_ratio.head(10).iterrows():
        print(f"    {row['product_code']}: {row['count']}회 거래, "
              f"최저가 {row['min']:.2f} → 최고가 {row['max']:.2f} "
              f"(비율: {row['price_ratio']:.2f}배)")
    
    return extreme_by_ratio, extreme_by_range, extreme_by_cv

def create_detailed_visualizations(price_stats, df_with_product, extreme_cases):
    """상세한 시각화 생성"""
    print("\n=== 5. 상세 시각화 생성 ===")
    
    # 1. 전체 가격 변동 분포
    plt.figure(figsize=(20, 15))
    
    # 가격 비율 분포
    plt.subplot(3, 3, 1)
    plt.hist(price_stats['price_ratio'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Price Ratio (Max/Min)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Price Ratios')
    plt.axvline(price_stats['price_ratio'].median(), color='red', linestyle='--', 
                label=f'Median: {price_stats["price_ratio"].median():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 가격 범위 분포
    plt.subplot(3, 3, 2)
    plt.hist(price_stats['price_range'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Price Range (Max - Min)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Price Ranges')
    plt.axvline(price_stats['price_range'].median(), color='red', linestyle='--',
                label=f'Median: {price_stats["price_range"].median():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 변동계수 분포
    plt.subplot(3, 3, 3)
    plt.hist(price_stats['cv'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Coefficient of Variation')
    plt.axvline(price_stats['cv'].median(), color='red', linestyle='--',
                label=f'Median: {price_stats["cv"].median():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 거래 횟수 vs 가격 비율
    plt.subplot(3, 3, 4)
    plt.scatter(price_stats['count'], price_stats['price_ratio'], alpha=0.6, s=20)
    plt.xlabel('Number of Transactions')
    plt.ylabel('Price Ratio (Max/Min)')
    plt.title('Transaction Count vs Price Ratio')
    plt.grid(True, alpha=0.3)
    
    # 평균 가격 vs 가격 비율
    plt.subplot(3, 3, 5)
    plt.scatter(price_stats['mean'], price_stats['price_ratio'], alpha=0.6, s=20)
    plt.xlabel('Average Price')
    plt.ylabel('Price Ratio (Max/Min)')
    plt.title('Average Price vs Price Ratio')
    plt.grid(True, alpha=0.3)
    
    # 상위 20개 극단 케이스 막대 그래프
    plt.subplot(3, 3, 6)
    top_20 = extreme_cases.head(20)
    bars = plt.bar(range(len(top_20)), top_20['price_ratio'], 
                   color='red', alpha=0.7, edgecolor='black')
    plt.xlabel('Product Code (Top 20)')
    plt.ylabel('Price Ratio (Max/Min)')
    plt.title('Top 20 Extreme Price Variation Cases')
    plt.xticks(range(len(top_20)), [f"PC{int(x)}" for x in top_20['product_code']], 
               rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 가격 비율 상위 10개 제품의 상세 정보
    plt.subplot(3, 3, 7)
    top_10 = extreme_cases.head(10)
    y_pos = np.arange(len(top_10))
    plt.barh(y_pos, top_10['price_ratio'], color='darkred', alpha=0.7)
    plt.yticks(y_pos, [f"PC{int(x)}" for x in top_10['product_code']])
    plt.xlabel('Price Ratio (Max/Min)')
    plt.title('Top 10 Price Variation Cases')
    plt.grid(True, alpha=0.3)
    
    # 가격 범위 상위 10개
    plt.subplot(3, 3, 8)
    top_10_range = price_stats.nlargest(10, 'price_range')
    y_pos = np.arange(len(top_10_range))
    plt.barh(y_pos, top_10_range['price_range'], color='darkblue', alpha=0.7)
    plt.yticks(y_pos, [f"PC{int(x)}" for x in top_10_range['product_code']])
    plt.xlabel('Price Range (Max - Min)')
    plt.title('Top 10 Price Range Cases')
    plt.grid(True, alpha=0.3)
    
    # 거래 횟수 분포
    plt.subplot(3, 3, 9)
    plt.hist(price_stats['count'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Frequency')
    plt.title('Distribution of Transaction Counts')
    plt.axvline(price_stats['count'].median(), color='red', linestyle='--',
                label=f'Median: {price_stats["count"].median():.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset/hnm/price_variation_analysis.png', dpi=300, bbox_inches='tight')
    print("  가격 변동 분석 차트 저장: dataset/hnm/price_variation_analysis.png")

def analyze_specific_products(df_with_product, extreme_cases, top_n=5):
    """상위 N개 극단 케이스의 상세 분석"""
    print(f"\n=== 6. 상위 {top_n}개 극단 케이스 상세 분석 ===")
    
    plt.figure(figsize=(20, 4 * top_n))
    
    for i, (_, row) in enumerate(extreme_cases.head(top_n).iterrows()):
        product_code = row['product_code']
        
        # 해당 제품의 모든 거래 데이터
        product_data = df_with_product[df_with_product['product_code'] == product_code]['price']
        
        plt.subplot(top_n, 3, i*3 + 1)
        plt.hist(product_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title(f'Product {int(product_code)}: Price Distribution\n'
                 f'Count: {row["count"]}, Ratio: {row["price_ratio"]:.2f}x')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(top_n, 3, i*3 + 2)
        plt.boxplot(product_data, vert=True)
        plt.ylabel('Price')
        plt.title(f'Product {int(product_code)}: Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(top_n, 3, i*3 + 3)
        # 시간 순서대로 정렬 (인덱스 기준)
        sorted_prices = product_data.sort_index()
        plt.plot(range(len(sorted_prices)), sorted_prices, 'o-', alpha=0.7, markersize=3)
        plt.xlabel('Transaction Order')
        plt.ylabel('Price')
        plt.title(f'Product {int(product_code)}: Price Over Time')
        plt.grid(True, alpha=0.3)
        
        print(f"  제품 {int(product_code)}:")
        print(f"    거래 횟수: {row['count']}회")
        print(f"    최저가: {row['min']:.2f}")
        print(f"    최고가: {row['max']:.2f}")
        print(f"    가격 비율: {row['price_ratio']:.2f}배")
        print(f"    평균가: {row['mean']:.2f}")
        print(f"    표준편차: {row['std']:.2f}")
        print(f"    변동계수: {row['cv']:.3f}")
        print()
    
    plt.tight_layout()
    plt.savefig('dataset/hnm/extreme_cases_detailed.png', dpi=300, bbox_inches='tight')
    print(f"  상위 {top_n}개 극단 케이스 상세 분석 차트 저장: dataset/hnm/extreme_cases_detailed.png")

def save_results(price_stats, extreme_cases):
    """결과 저장"""
    print("\n=== 7. 결과 저장 ===")
    
    # 전체 통계 저장
    output_file = 'dataset/hnm/price_variation_stats.csv'
    price_stats.to_csv(output_file, index=False)
    print(f"  가격 변동 통계 저장: {output_file}")
    
    # 극단 케이스 저장
    extreme_file = 'dataset/hnm/extreme_price_cases.csv'
    extreme_cases.to_csv(extreme_file, index=False)
    print(f"  극단 케이스 저장: {extreme_file}")
    
    # 요약 보고서 저장
    summary_file = 'dataset/hnm/price_variation_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== H&M 데이터 가격 변동 분석 결과 ===\n\n")
        f.write(f"분석 대상: 거래 횟수 5회 이상인 product_code\n")
        f.write(f"총 분석 대상: {len(price_stats):,}개 제품\n\n")
        
        f.write("전체 통계:\n")
        f.write(f"  평균 가격 비율 (최고가/최저가): {price_stats['price_ratio'].mean():.2f}배\n")
        f.write(f"  중간값 가격 비율: {price_stats['price_ratio'].median():.2f}배\n")
        f.write(f"  최대 가격 비율: {price_stats['price_ratio'].max():.2f}배\n")
        f.write(f"  평균 가격 범위: {price_stats['price_range'].mean():.2f}\n")
        f.write(f"  평균 변동계수: {price_stats['cv'].mean():.3f}\n\n")
        
        f.write("상위 10개 극단 케이스:\n")
        for i, (_, row) in enumerate(extreme_cases.head(10).iterrows(), 1):
            f.write(f"  {i:2d}. 제품 {int(row['product_code'])}: "
                   f"{row['count']}회 거래, "
                   f"{row['min']:.2f} → {row['max']:.2f} "
                   f"({row['price_ratio']:.2f}배)\n")
    
    print(f"  요약 보고서 저장: {summary_file}")

def main():
    print("=" * 80)
    print("H&M 데이터 가격 변동 분석")
    print("=" * 80)
    
    # 1. 거래 데이터 로드
    transactions_df = load_transaction_data()
    
    # 2. 제품 정보 로드
    product_mapping = load_article_data()
    
    # 3. 가격 변동 분석
    price_stats, df_with_product = analyze_price_variation(transactions_df, product_mapping)
    
    # 4. 극단 케이스 찾기
    extreme_by_ratio, extreme_by_range, extreme_by_cv = find_extreme_cases(price_stats)
    
    # 5. 시각화
    create_detailed_visualizations(price_stats, df_with_product, extreme_by_ratio)
    
    # 6. 상세 분석
    analyze_specific_products(df_with_product, extreme_by_ratio, top_n=5)
    
    # 7. 결과 저장
    save_results(price_stats, extreme_by_ratio)
    
    print("\n" + "=" * 80)
    print("[완료] 가격 변동 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

