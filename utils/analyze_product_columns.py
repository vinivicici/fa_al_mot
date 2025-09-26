#!/usr/bin/env python3
"""
articles_with_price.csv에서 product 관련 칼럼들의 고유값을 분석하고 정리하는 스크립트
"""

import pandas as pd
from collections import Counter

def analyze_product_columns():
    print("=== articles_with_price.csv의 product 칼럼 분석 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('articles_with_price.csv')
        print(f"총 {len(df):,}개 행 로드 완료")
        
        # product 관련 칼럼들 식별
        product_columns = [col for col in df.columns if 'product' in col.lower()]
        print(f"\nproduct 관련 칼럼들: {product_columns}")
        
        # 추가로 분석할 중요한 칼럼들
        important_columns = [
            'product_type_name', 'product_group_name', 
            'department_name', 'section_name', 'garment_group_name',
            'colour_group_name', 'index_group_name'
        ]
        
        # 결과를 저장할 딕셔너리
        column_analysis = {}
        
        for col in important_columns:
            if col in df.columns:
                print(f"\n분석 중: {col}")
                unique_values = df[col].value_counts().sort_values(ascending=False)
                
                column_analysis[col] = {
                    'unique_count': len(unique_values),
                    'top_values': unique_values.head(20).to_dict(),
                    'total_count': len(df)
                }
                
                print(f"  고유값 개수: {len(unique_values)}")
                print(f"  상위 5개: {list(unique_values.head(5).index)}")
        
        # 결과를 CSV 파일들로 저장
        print(f"\n=== CSV 파일 생성 중 ===")
        
        # 1. 전체 요약 파일
        summary_data = []
        for col, info in column_analysis.items():
            summary_data.append({
                'column_name': col,
                'unique_count': info['unique_count'],
                'total_rows': info['total_count'],
                'top_3_values': ', '.join(list(info['top_values'].keys())[:3])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('product_columns_summary.csv', index=False)
        print("✅ product_columns_summary.csv 생성 완료")
        
        # 2. 각 칼럼별 상세 파일들
        for col, info in column_analysis.items():
            detail_data = []
            for value, count in info['top_values'].items():
                percentage = (count / info['total_count']) * 100
                detail_data.append({
                    'value': value,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
            
            detail_df = pd.DataFrame(detail_data)
            filename = f"product_{col}_details.csv"
            detail_df.to_csv(filename, index=False)
            print(f"✅ {filename} 생성 완료 ({len(detail_data)}개 고유값)")
        
        # 3. 통합 분석 파일 (모든 칼럼의 상위 값들)
        print(f"\n=== 통합 분석 파일 생성 ===")
        all_analysis = []
        
        for col, info in column_analysis.items():
            for i, (value, count) in enumerate(list(info['top_values'].items())[:10]):
                percentage = (count / info['total_count']) * 100
                all_analysis.append({
                    'column_name': col,
                    'rank': i + 1,
                    'value': value,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
        
        all_df = pd.DataFrame(all_analysis)
        all_df.to_csv('product_all_analysis.csv', index=False)
        print("✅ product_all_analysis.csv 생성 완료")
        
        # 결과 요약 출력
        print(f"\n" + "="*60)
        print("📊 분석 결과 요약:")
        for col, info in column_analysis.items():
            print(f"\n{col}:")
            print(f"  - 고유값: {info['unique_count']}개")
            print(f"  - 상위 3개: {list(info['top_values'].keys())[:3]}")
        
        print(f"\n생성된 파일들:")
        print(f"  - product_columns_summary.csv (전체 요약)")
        print(f"  - product_all_analysis.csv (통합 분석)")
        for col in column_analysis.keys():
            print(f"  - product_{col}_details.csv (상세 분석)")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    analyze_product_columns()

