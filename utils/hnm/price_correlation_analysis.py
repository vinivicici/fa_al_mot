#!/usr/bin/env python3
"""
transactions_train과 articles.csv join 직후 칼럼별 가격 상관계수 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
    """articles.csv와 가격 데이터를 JOIN"""
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

def analyze_price_correlations(df):
    """각 칼럼과 가격 간의 상관계수 분석"""
    print("\n=== 3. 칼럼별 가격 상관계수 분석 ===")
    
    # 가격이 있는 데이터만 사용
    df_with_price = df.dropna(subset=['price']).copy()
    print(f"가격 데이터가 있는 행 수: {len(df_with_price):,}")
    
    # 분석할 칼럼들 (가격 제외)
    analysis_columns = [col for col in df_with_price.columns if col != 'price']
    
    correlation_results = []
    
    print(f"\n총 {len(analysis_columns)}개 칼럼 분석 중...")
    
    for col in analysis_columns:
        print(f"  분석 중: {col}")
        
        # 칼럼 타입에 따른 처리
        if df_with_price[col].dtype in ['int64', 'float64']:
            # 숫자형 칼럼 - 피어슨 상관계수
            corr, p_value = pearsonr(df_with_price[col], df_with_price['price'])
            corr_type = 'pearson'
        else:
            # 범주형 칼럼 - 스피어만 상관계수 (순서형으로 변환)
            try:
                # 범주형 데이터를 숫자로 변환
                if df_with_price[col].dtype == 'object':
                    # 문자열을 숫자로 변환 (고유값의 순서대로)
                    unique_vals = df_with_price[col].unique()
                    val_to_num = {val: i for i, val in enumerate(unique_vals)}
                    numeric_col = df_with_price[col].map(val_to_num)
                else:
                    numeric_col = df_with_price[col]
                
                corr, p_value = spearmanr(numeric_col, df_with_price['price'])
                corr_type = 'spearman'
            except:
                corr, p_value = np.nan, np.nan
                corr_type = 'failed'
        
        # 결측값 비율 계산
        missing_ratio = df_with_price[col].isna().sum() / len(df_with_price)
        
        # 고유값 개수 (범주형의 경우)
        unique_count = df_with_price[col].nunique()
        
        correlation_results.append({
            'column': col,
            'correlation': corr,
            'p_value': p_value,
            'correlation_type': corr_type,
            'missing_ratio': missing_ratio,
            'unique_count': unique_count,
            'data_type': str(df_with_price[col].dtype)
        })
    
    return pd.DataFrame(correlation_results)

def create_visualizations(correlation_df, df_with_price):
    """상관계수 시각화"""
    print("\n=== 4. 시각화 생성 ===")
    
    # 1. 상관계수 막대 그래프
    plt.figure(figsize=(15, 10))
    
    # 상관계수 절댓값으로 정렬
    sorted_df = correlation_df.sort_values('correlation', key=abs, ascending=False)
    
    # 상관계수가 유효한 것만 필터링
    valid_corr = sorted_df.dropna(subset=['correlation'])
    
    plt.subplot(2, 2, 1)
    colors = ['red' if x < 0 else 'blue' for x in valid_corr['correlation']]
    bars = plt.barh(range(len(valid_corr)), valid_corr['correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(valid_corr)), valid_corr['column'], fontsize=8)
    plt.xlabel('Correlation with Price')
    plt.title('Price Correlation by Column')
    plt.grid(True, alpha=0.3)
    
    # 상관계수 값 표시
    for i, (bar, corr) in enumerate(zip(bars, valid_corr['correlation'])):
        plt.text(corr + (0.01 if corr >= 0 else -0.01), i, f'{corr:.3f}', 
                va='center', ha='left' if corr >= 0 else 'right', fontsize=7)
    
    # 2. p-value 히트맵
    plt.subplot(2, 2, 2)
    p_values = valid_corr['p_value'].values
    p_value_colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'green' for p in p_values]
    
    bars2 = plt.barh(range(len(valid_corr)), p_values, color=p_value_colors, alpha=0.7)
    plt.yticks(range(len(valid_corr)), valid_corr['column'], fontsize=8)
    plt.xlabel('P-value')
    plt.title('Statistical Significance (P-value)')
    plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='p=0.1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 결측값 비율
    plt.subplot(2, 2, 3)
    missing_data = valid_corr.sort_values('missing_ratio', ascending=True)
    plt.barh(range(len(missing_data)), missing_data['missing_ratio'], color='orange', alpha=0.7)
    plt.yticks(range(len(missing_data)), missing_data['column'], fontsize=8)
    plt.xlabel('Missing Value Ratio')
    plt.title('Missing Value Ratio by Column')
    plt.grid(True, alpha=0.3)
    
    # 4. 고유값 개수 (범주형 데이터)
    plt.subplot(2, 2, 4)
    unique_data = valid_corr.sort_values('unique_count', ascending=True)
    plt.barh(range(len(unique_data)), unique_data['unique_count'], color='green', alpha=0.7)
    plt.yticks(range(len(unique_data)), unique_data['column'], fontsize=8)
    plt.xlabel('Unique Value Count')
    plt.title('Unique Value Count by Column')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset/hnm/price_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("  상관계수 분석 차트 저장: dataset/hnm/price_correlation_analysis.png")
    
    # 5. 색상 관련 칼럼들만 따로 분석
    color_columns = [col for col in valid_corr['column'] if 'colour' in col.lower() or 'color' in col.lower()]
    
    if color_columns:
        plt.figure(figsize=(12, 6))
        
        color_corr = valid_corr[valid_corr['column'].isin(color_columns)]
        
        plt.subplot(1, 2, 1)
        colors = ['red' if x < 0 else 'blue' for x in color_corr['correlation']]
        bars = plt.bar(range(len(color_corr)), color_corr['correlation'], color=colors, alpha=0.7)
        plt.xticks(range(len(color_corr)), color_corr['column'], rotation=45, ha='right')
        plt.ylabel('Correlation with Price')
        plt.title('Color-related Columns Correlation')
        plt.grid(True, alpha=0.3)
        
        # 상관계수 값 표시
        for i, (bar, corr) in enumerate(zip(bars, color_corr['correlation'])):
            plt.text(i, corr + (0.01 if corr >= 0 else -0.01), f'{corr:.3f}', 
                    ha='center', va='bottom' if corr >= 0 else 'top', fontsize=9)
        
        plt.subplot(1, 2, 2)
        p_values = color_corr['p_value'].values
        p_value_colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'green' for p in p_values]
        
        plt.bar(range(len(color_corr)), p_values, color=p_value_colors, alpha=0.7)
        plt.xticks(range(len(color_corr)), color_corr['column'], rotation=45, ha='right')
        plt.ylabel('P-value')
        plt.title('Color-related Columns P-value')
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='p=0.1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset/hnm/color_columns_correlation.png', dpi=300, bbox_inches='tight')
        print("  색상 관련 칼럼 분석 차트 저장: dataset/hnm/color_columns_correlation.png")

def save_detailed_results(correlation_df, df_with_price):
    """상세 결과를 CSV로 저장"""
    print("\n=== 5. 상세 결과 저장 ===")
    
    # 상관계수 결과 저장
    output_file = 'dataset/hnm/price_correlation_results.csv'
    correlation_df.to_csv(output_file, index=False)
    print(f"  상관계수 결과 저장: {output_file}")
    
    # 요약 통계 저장
    summary_file = 'dataset/hnm/price_correlation_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== H&M 데이터 가격 상관계수 분석 결과 ===\n\n")
        f.write(f"분석 대상 데이터: {len(df_with_price):,}개 행\n")
        f.write(f"분석 대상 칼럼: {len(correlation_df)}개\n\n")
        
        # 높은 상관계수 칼럼들
        high_corr = correlation_df[correlation_df['correlation'].abs() > 0.1].sort_values('correlation', key=abs, ascending=False)
        f.write("높은 상관계수 칼럼들 (|r| > 0.1):\n")
        for _, row in high_corr.iterrows():
            f.write(f"  {row['column']}: {row['correlation']:.4f} (p={row['p_value']:.4f})\n")
        
        f.write("\n색상 관련 칼럼들:\n")
        color_cols = correlation_df[correlation_df['column'].str.contains('colour|color', case=False, na=False)]
        for _, row in color_cols.iterrows():
            f.write(f"  {row['column']}: {row['correlation']:.4f} (p={row['p_value']:.4f})\n")
        
        f.write("\n통계적으로 유의한 칼럼들 (p < 0.05):\n")
        significant = correlation_df[correlation_df['p_value'] < 0.05].sort_values('correlation', key=abs, ascending=False)
        for _, row in significant.iterrows():
            f.write(f"  {row['column']}: {row['correlation']:.4f} (p={row['p_value']:.4f})\n")
    
    print(f"  요약 결과 저장: {summary_file}")

def main():
    print("=" * 80)
    print("H&M 데이터 가격 상관계수 분석")
    print("=" * 80)
    
    # 1. article_id별 평균 가격 계산
    article_avg_prices = calculate_article_prices()
    
    # 2. articles.csv와 JOIN
    articles_df = join_articles_with_prices(article_avg_prices)
    
    # 3. 상관계수 분석
    correlation_df = analyze_price_correlations(articles_df)
    
    # 4. 시각화
    df_with_price = articles_df.dropna(subset=['price'])
    create_visualizations(correlation_df, df_with_price)
    
    # 5. 결과 저장
    save_detailed_results(correlation_df, df_with_price)
    
    print("\n" + "=" * 80)
    print("[완료] 가격 상관계수 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
