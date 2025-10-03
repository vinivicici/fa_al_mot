#!/usr/bin/env python3
"""
farfetch.csv의 각 칼럼별 고유값과 분포를 분석하는 스크립트
"""

import pandas as pd
import os

def analyze_farfetch_columns():
    print("=== Farfetch 데이터 칼럼 분석 ===")
    
    try:
        # CSV 파일 로드
        print("\n1. CSV 파일 로딩 중...")
        df = pd.read_csv('farfetch.csv')
        print(f"데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 출력 디렉토리 생성
        output_dir = 'farfetch_column_observation'
        os.makedirs(output_dir, exist_ok=True)
        
        # 분석할 카테고리 칼럼들 (텍스트 기반, 너무 길지 않은 것들)
        categorical_columns = [
            'brand', 'currency', 'availability', 'condition', 'gender'
        ]
        
        print(f"\n2. 카테고리 칼럼 분석 중...")
        
        all_analysis = []
        
        for col in categorical_columns:
            if col not in df.columns:
                print(f"  ⚠️ '{col}' 칼럼이 존재하지 않습니다.")
                continue
                
            print(f"  - {col} 분석 중...")
            
            # 고유값 개수
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            # 값 분포
            value_counts = df[col].value_counts()
            
            # 전체 분석 요약에 추가
            all_analysis.append({
                'column': col,
                'unique_values': unique_count,
                'null_count': null_count,
                'null_percentage': f"{(null_count / len(df) * 100):.2f}%",
                'most_common': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0
            })
            
            # 개별 칼럼 상세 정보 저장
            details = pd.DataFrame({
                'value': value_counts.index,
                'count': value_counts.values,
                'percentage': (value_counts.values / len(df) * 100).round(2)
            })
            
            detail_file = os.path.join(output_dir, f'farfetch_{col}_details.csv')
            details.to_csv(detail_file, index=False, encoding='utf-8-sig')
            print(f"    ✓ {detail_file} 저장 완료 ({len(details)} 개 고유값)")
        
        # 전체 요약 저장
        print(f"\n3. 전체 요약 저장 중...")
        summary_df = pd.DataFrame(all_analysis)
        summary_file = os.path.join(output_dir, 'farfetch_columns_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"  ✓ {summary_file} 저장 완료")
        
        # 요약 출력
        print(f"\n{'='*80}")
        print("분석 요약:")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        
        # 추가 분석: price 통계
        print(f"\n{'='*80}")
        print("가격(price) 통계:")
        print(f"{'='*80}")
        print(df['price'].describe())
        
        # Price 분포를 파일로 저장
        price_stats = pd.DataFrame({
            'statistic': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
            'value': [
                df['price'].count(),
                df['price'].mean(),
                df['price'].std(),
                df['price'].min(),
                df['price'].quantile(0.25),
                df['price'].quantile(0.50),
                df['price'].quantile(0.75),
                df['price'].max()
            ]
        })
        price_file = os.path.join(output_dir, 'farfetch_price_stats.csv')
        price_stats.to_csv(price_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ {price_file} 저장 완료")
        
        # breadcrumbs 샘플 저장 (카테고리 정보 포함)
        if 'breadcrumbs' in df.columns:
            print(f"\n4. Breadcrumbs (카테고리) 샘플 저장 중...")
            breadcrumbs_sample = df[['brand', 'title', 'breadcrumbs', 'gender']].head(20)
            breadcrumbs_file = os.path.join(output_dir, 'farfetch_breadcrumbs_sample.csv')
            breadcrumbs_sample.to_csv(breadcrumbs_file, index=False, encoding='utf-8-sig')
            print(f"  ✓ {breadcrumbs_file} 저장 완료")
        
        print(f"\n{'='*80}")
        print(f"✅ 모든 분석 완료! 결과는 '{output_dir}/' 폴더에 저장되었습니다.")
        print(f"{'='*80}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_farfetch_columns()

