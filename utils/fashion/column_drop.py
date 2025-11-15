#!/usr/bin/env python3
"""
fashion_2_rowdrop.csv에서 불필요한 메타데이터 칼럼들을 제거하는 스크립트
- meta.code
- meta.requestId
"""

import pandas as pd
import os
import sys
import argparse

def drop_meta_columns(input_csv: str = 'dataset/fashion/fashion_2_rowdrop.csv',
                      output_csv: str = 'dataset/fashion/fashion_3_columndrop.csv'):
    print("=== fashion 데이터 메타데이터 칼럼 제거 ===")
    
    if not os.path.exists(input_csv):
        print(f"[오류] 입력 CSV가 없습니다: {input_csv}")
        return False
    
    try:
        # CSV 파일 로드
        print(f"CSV 파일 로딩 중: {input_csv}")
        df = pd.read_csv(input_csv)
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
        df_dropped.to_csv(output_csv, index=False)
        print(f"\n[완료] 칼럼 제거 완료!")
        print(f"결과가 '{output_csv}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fashion CSV 메타데이터 칼럼 제거')
    parser.add_argument(
        '--input',
        default='dataset/fashion/fashion_2_rowdrop.csv',
        help='입력 CSV 경로 (기본: dataset/fashion/fashion_2_rowdrop.csv)'
    )
    parser.add_argument(
        '--output',
        default='dataset/fashion/fashion_3_columndrop.csv',
        help='출력 CSV 경로 (기본: dataset/fashion/fashion_3_columndrop.csv)'
    )
    args = parser.parse_args()
    
    success = drop_meta_columns(args.input, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
