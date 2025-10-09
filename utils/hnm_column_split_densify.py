#!/usr/bin/env python3
"""
articles_with_price.csv의 칼럼 처리 스크립트
- product_group_name 칼럼 제거
"""

import pandas as pd

def remove_product_group_column():
    """
    product_group_name 칼럼 제거
    """
    print("=== product_group_name 칼럼 제거 ===")
    
    try:
        # CSV 파일 로드
        print("\nCSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 현재 칼럼 목록
        print(f"\n현재 칼럼 목록 ({len(df.columns)}개):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # product_group_name 칼럼 제거
        if 'product_group_name' in df.columns:
            df_filtered = df.drop(columns=['product_group_name'])
            print(f"\n[완료] 'product_group_name' 칼럼 제거 완료!")
        else:
            print(f"\n⚠️ 'product_group_name' 칼럼이 존재하지 않습니다.")
            df_filtered = df
        
        # 제거 후 칼럼 목록
        print(f"\n제거 후 칼럼 목록 ({len(df_filtered.columns)}개):")
        for i, col in enumerate(df_filtered.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n최종 데이터 크기: {df_filtered.shape[0]:,} 행 x {df_filtered.shape[1]} 칼럼")
        
        # 결과 저장
        output_file = 'dataset/hnm/articles_with_price.csv'
        df_filtered.to_csv(output_file, index=False)
        print(f"\n[완료] 칼럼 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    remove_product_group_column()

