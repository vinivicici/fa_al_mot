#!/usr/bin/env python3
"""
이름 일관성 디버깅용 스크립트
"""

import pandas as pd

def debug_name_consistency():
    print("=== 이름 일관성 디버깅 ===")
    
    # CSV 파일 로드
    df = pd.read_csv('articles_with_price.csv')
    
    # 샘플 데이터 확인
    print("샘플 데이터 (처음 10개):")
    sample_cols = ['department_name', 'section_name', 'garment_group_name']
    sample_df = df[sample_cols].head(10)
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        combined = row['section_name'] + ' ' + row['garment_group_name']
        is_match = row['department_name'].strip() == combined.strip()
        
        print(f"\n{i+1}. department_name: '{row['department_name']}'")
        print(f"   section_name: '{row['section_name']}'")
        print(f"   garment_group_name: '{row['garment_group_name']}'")
        print(f"   combined: '{combined}'")
        print(f"   일치: {is_match}")
        
        if not is_match:
            print(f"   차이점:")
            print(f"     department_name 길이: {len(row['department_name'])}")
            print(f"     combined 길이: {len(combined)}")
            print(f"     department_name repr: {repr(row['department_name'])}")
            print(f"     combined repr: {repr(combined)}")

if __name__ == "__main__":
    debug_name_consistency()
