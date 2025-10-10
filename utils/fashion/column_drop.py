#!/usr/bin/env python3
"""
fashion.csv에서 불필요한 메타데이터 칼럼들을 제거하는 스크립트
- meta.code
- meta.requestId
"""

import pandas as pd

def drop_meta_columns():
    print("=== fashion.csv 메타데이터 칼럼 제거 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/fashion/fashion.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 제거할 칼럼들
        columns_to_drop = [
            'meta.code',
            'meta.requestId',
            'baseColour',
            'colour1', 
            'colour2',
            'season',
            'year',
            'variantName',
            'articleNumber',
            'displayCategories',
            'productDisplayName'
        ]
        
        # 실제로 존재하는 칼럼들만 필터링
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        missing_columns = [col for col in columns_to_drop if col not in df.columns]
        
        print(f"\n제거할 칼럼들 ({len(existing_columns_to_drop)}개):")
        for col in existing_columns_to_drop:
            print(f"  - {col}")
        
        if missing_columns:
            print(f"\n존재하지 않는 칼럼들 ({len(missing_columns)}개):")
            for col in missing_columns:
                print(f"  - {col}")
        
        # 칼럼 제거
        df_dropped = df.drop(columns=existing_columns_to_drop)
        
        print(f"\n제거 후 데이터 크기: {df_dropped.shape[0]:,} 행 x {df_dropped.shape[1]} 칼럼")
        print(f"제거된 칼럼 수: {len(existing_columns_to_drop)}개")
        
        # 남은 칼럼들 출력
        print(f"\n남은 칼럼들 ({len(df_dropped.columns)}개):")
        for i, col in enumerate(df_dropped.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 결과 저장
        output_file = 'dataset/fashion/fashion.csv'
        df_dropped.to_csv(output_file, index=False)
        print(f"\n[완료] 칼럼 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        return False

if __name__ == "__main__":
    drop_meta_columns()
