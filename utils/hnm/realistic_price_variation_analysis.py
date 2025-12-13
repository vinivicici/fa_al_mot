#!/usr/bin/env python3
"""
가격이 0인 거래를 제외한 현실적인 가격 변동 분석
제품별 상세 분포 시각화
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

def load_and_filter_data():
    """거래 데이터 로드 및 가격 0인 거래 제외"""
    print("=== 1. 거래 데이터 로드 및 필터링 ===")
    
    # 메모리 효율적인 방법으로 데이터 로드
    chunk_size = 1000000
    all_data = []
    
    chunk_num = 0
    for chunk in pd.read_csv('dataset/hnm/transactions_train.csv', chunksize=chunk_size):
        chunk_num += 1
        print(f"  청크 {chunk_num} 처리 중... (크기: {len(chunk):,})")
        
        # 가격이 0보다 큰 거래만 필터링
        chunk_filtered = chunk[chunk['price'] > 0][['article_id', 'price']].copy()
        all_data.append(chunk_filtered)
    
    # 데이터 병합
    df = pd.concat(all_data, ignore_index=True)
    print(f"  필터링 후 거래 데이터: {len(df):,}개 행")
    
    # 제품 정보 로드
    articles_df = pd.read_csv('dataset/hnm/articles.csv')
    product_mapping = articles_df[['article_id', 'product_code']].copy()
    
    # 거래 데이터에 product_code 추가
    df_with_product = df.merge(product_mapping, on='article_id', how='left')
    
    return df_with_product

def analyze_realistic_price_variation(df):
    """현실적인 가격 변동 분석"""
    print("\n=== 2. 현실적인 가격 변동 분석 ===")
    
    # product_code별 가격 통계 계산
    print("  product_code별 가격 통계 계산 중...")
    
    price_stats = df.groupby('product_code')['price'].agg([
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
    price_stats['price_range_ratio'] = price_stats['price_range'] / price_stats['mean']  # 범위/평균 비율
    
    print(f"  거래 횟수 5개 이상인 product_code: {len(price_stats):,}개")
    
    return price_stats

def find_interesting_cases(price_stats):
    """흥미로운 케이스 찾기"""
    print("\n=== 3. 흥미로운 케이스 찾기 ===")
    
    # 상식적인 범위 내에서 가격 변동이 큰 케이스들
    # 가격 비율 2배 이상 10배 이하, 평균 가격 0.01 이상
    realistic_cases = price_stats[
        (price_stats['price_ratio'] >= 2.0) & 
        (price_stats['price_ratio'] <= 10.0) & 
        (price_stats['mean'] >= 0.01)
    ].copy()
    
    # 가격 비율 기준으로 정렬
    interesting_cases = realistic_cases.nlargest(20, 'price_ratio')
    
    print(f"  상식적인 범위 내 흥미로운 케이스: {len(realistic_cases):,}개")
    print("  상위 10개 케이스:")
    for i, (_, row) in enumerate(interesting_cases.head(10).iterrows(), 1):
        print(f"    {i:2d}. 제품 {int(row['product_code'])}: "
              f"{row['count']}회 거래, "
              f"{row['min']:.3f} → {row['max']:.3f} "
              f"({row['price_ratio']:.2f}배, 평균: {row['mean']:.3f})")
    
    return interesting_cases

def create_individual_product_analysis(df, interesting_cases, top_n=6):
    """개별 제품 상세 분석 및 시각화"""
    print(f"\n=== 4. 상위 {top_n}개 제품 개별 분석 ===")
    
    fig, axes = plt.subplots(top_n, 3, figsize=(18, 4*top_n))
    if top_n == 1:
        axes = axes.reshape(1, -1)
    
    for i, (_, row) in enumerate(interesting_cases.head(top_n).iterrows()):
        product_code = row['product_code']
        
        # 해당 제품의 모든 거래 데이터
        product_data = df[df['product_code'] == product_code]['price']
        
        # 1. 히스토그램
        axes[i, 0].hist(product_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i, 0].axvline(row['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {row["mean"]:.3f}')
        axes[i, 0].axvline(row['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {row["median"]:.3f}')
        axes[i, 0].set_xlabel('Price')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].set_title(f'Product {int(product_code)}: Price Distribution\n'
                           f'Count: {row["count"]}, Ratio: {row["price_ratio"]:.2f}x')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # 2. 박스 플롯
        box_data = [product_data.values]
        bp = axes[i, 1].boxplot(box_data, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        axes[i, 1].set_ylabel('Price')
        axes[i, 1].set_title(f'Product {int(product_code)}: Box Plot')
        axes[i, 1].set_xticklabels(['Price'])
        axes[i, 1].grid(True, alpha=0.3)
        
        # 3. 시간 순서 플롯 (인덱스 기준)
        sorted_prices = product_data.sort_index()
        axes[i, 2].plot(range(len(sorted_prices)), sorted_prices, 'o-', alpha=0.7, markersize=2)
        axes[i, 2].axhline(row['mean'], color='red', linestyle='--', alpha=0.7, label=f'Mean: {row["mean"]:.3f}')
        axes[i, 2].axhline(row['min'], color='green', linestyle=':', alpha=0.7, label=f'Min: {row["min"]:.3f}')
        axes[i, 2].axhline(row['max'], color='red', linestyle=':', alpha=0.7, label=f'Max: {row["max"]:.3f}')
        axes[i, 2].set_xlabel('Transaction Order')
        axes[i, 2].set_ylabel('Price')
        axes[i, 2].set_title(f'Product {int(product_code)}: Price Over Time')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        
        print(f"  제품 {int(product_code)}:")
        print(f"    거래 횟수: {row['count']}회")
        print(f"    최저가: {row['min']:.3f}")
        print(f"    최고가: {row['max']:.3f}")
        print(f"    평균가: {row['mean']:.3f}")
        print(f"    중간값: {row['median']:.3f}")
        print(f"    가격 비율: {row['price_ratio']:.2f}배")
        print(f"    표준편차: {row['std']:.3f}")
        print(f"    변동계수: {row['cv']:.3f}")
        print()
    
    plt.tight_layout()
    plt.savefig('dataset/hnm/realistic_price_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  현실적인 가격 변동 분석 차트 저장: dataset/hnm/realistic_price_analysis.png")

def create_overview_visualization(price_stats, interesting_cases):
    """전체 개요 시각화"""
    print("\n=== 5. 전체 개요 시각화 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 가격 비율 분포 (상식적인 범위)
    axes[0, 0].hist(price_stats['price_ratio'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(price_stats['price_ratio'].median(), color='red', linestyle='--', 
                      label=f'Median: {price_stats["price_ratio"].median():.2f}')
    axes[0, 0].set_xlabel('Price Ratio (Max/Min)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Price Ratios (Realistic Range)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 평균 가격 분포
    axes[0, 1].hist(price_stats['mean'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(price_stats['mean'].median(), color='red', linestyle='--',
                      label=f'Median: {price_stats["mean"].median():.3f}')
    axes[0, 1].set_xlabel('Average Price')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Average Prices')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 변동계수 분포
    axes[0, 2].hist(price_stats['cv'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].axvline(price_stats['cv'].median(), color='red', linestyle='--',
                      label=f'Median: {price_stats["cv"].median():.3f}')
    axes[0, 2].set_xlabel('Coefficient of Variation')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Coefficient of Variation')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 거래 횟수 vs 가격 비율
    scatter = axes[1, 0].scatter(price_stats['count'], price_stats['price_ratio'], 
                                alpha=0.6, s=20, c=price_stats['mean'], cmap='viridis')
    axes[1, 0].set_xlabel('Number of Transactions')
    axes[1, 0].set_ylabel('Price Ratio (Max/Min)')
    axes[1, 0].set_title('Transaction Count vs Price Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Average Price')
    
    # 5. 평균 가격 vs 가격 비율
    scatter2 = axes[1, 1].scatter(price_stats['mean'], price_stats['price_ratio'], 
                                 alpha=0.6, s=20, c=price_stats['count'], cmap='plasma')
    axes[1, 1].set_xlabel('Average Price')
    axes[1, 1].set_ylabel('Price Ratio (Max/Min)')
    axes[1, 1].set_title('Average Price vs Price Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1, 1], label='Transaction Count')
    
    # 6. 상위 케이스들
    top_cases = interesting_cases.head(10)
    bars = axes[1, 2].bar(range(len(top_cases)), top_cases['price_ratio'], 
                          color='red', alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Product Code (Top 10)')
    axes[1, 2].set_ylabel('Price Ratio (Max/Min)')
    axes[1, 2].set_title('Top 10 Interesting Cases')
    axes[1, 2].set_xticks(range(len(top_cases)))
    axes[1, 2].set_xticklabels([f"PC{int(x)}" for x in top_cases['product_code']], 
                               rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset/hnm/realistic_overview.png', dpi=300, bbox_inches='tight')
    print("  전체 개요 차트 저장: dataset/hnm/realistic_overview.png")

def save_results(price_stats, interesting_cases):
    """결과 저장"""
    print("\n=== 6. 결과 저장 ===")
    
    # 전체 통계 저장
    output_file = 'dataset/hnm/realistic_price_stats.csv'
    price_stats.to_csv(output_file, index=False)
    print(f"  현실적인 가격 통계 저장: {output_file}")
    
    # 흥미로운 케이스 저장
    interesting_file = 'dataset/hnm/interesting_price_cases.csv'
    interesting_cases.to_csv(interesting_file, index=False)
    print(f"  흥미로운 케이스 저장: {interesting_file}")
    
    # 요약 보고서 저장
    summary_file = 'dataset/hnm/realistic_price_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== H&M 데이터 현실적인 가격 변동 분석 결과 ===\n\n")
        f.write("분석 조건: 가격 > 0, 거래 횟수 >= 5회\n")
        f.write(f"총 분석 대상: {len(price_stats):,}개 제품\n\n")
        
        f.write("전체 통계:\n")
        f.write(f"  평균 가격 비율 (최고가/최저가): {price_stats['price_ratio'].mean():.2f}배\n")
        f.write(f"  중간값 가격 비율: {price_stats['price_ratio'].median():.2f}배\n")
        f.write(f"  최대 가격 비율: {price_stats['price_ratio'].max():.2f}배\n")
        f.write(f"  평균 가격: {price_stats['mean'].mean():.3f}\n")
        f.write(f"  중간값 가격: {price_stats['mean'].median():.3f}\n")
        f.write(f"  평균 변동계수: {price_stats['cv'].mean():.3f}\n\n")
        
        f.write("흥미로운 케이스 (2배 <= 비율 <= 10배, 평균가 >= 0.01):\n")
        for i, (_, row) in enumerate(interesting_cases.head(15).iterrows(), 1):
            f.write(f"  {i:2d}. 제품 {int(row['product_code'])}: "
                   f"{row['count']}회 거래, "
                   f"{row['min']:.3f} → {row['max']:.3f} "
                   f"({row['price_ratio']:.2f}배, 평균: {row['mean']:.3f})\n")
    
    print(f"  요약 보고서 저장: {summary_file}")

def main():
    print("=" * 80)
    print("H&M 데이터 현실적인 가격 변동 분석")
    print("=" * 80)
    
    # 1. 데이터 로드 및 필터링
    df = load_and_filter_data()
    
    # 2. 현실적인 가격 변동 분석
    price_stats = analyze_realistic_price_variation(df)
    
    # 3. 흥미로운 케이스 찾기
    interesting_cases = find_interesting_cases(price_stats)
    
    # 4. 개별 제품 분석
    create_individual_product_analysis(df, interesting_cases, top_n=6)
    
    # 5. 전체 개요 시각화
    create_overview_visualization(price_stats, interesting_cases)
    
    # 6. 결과 저장
    save_results(price_stats, interesting_cases)
    
    print("\n" + "=" * 80)
    print("[완료] 현실적인 가격 변동 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

