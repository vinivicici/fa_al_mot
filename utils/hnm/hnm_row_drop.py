#!/usr/bin/env python3
"""
articles_with_price.csv 전처리 스크립트
- 특정 section_name 값을 가진 행 제거 (악세서리, 속옷)
- 특정 product_group_name 값을 가진 행 제거 (속옷, 수영복, 양말, 화장품, 가방, 가구 등)
- 특정 garment_group_name 값을 가진 행 제거 (악세서리, 양말)
"""

import pandas as pd

def remove_specific_sections():
    print("=== articles_with_price.csv 특정 section_name 행 제거 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 제거할 section_name 값들
        sections_to_remove = [
            # 악세서리
            '*Womens Small accessories',
            # 속옷 관련
            'Womens Lingerie',
            'Men Underwear',
            'Girls Underwear & Basics',
            'Boys Underwear & Basics'
        ]
        
        # 제거 전 각 섹션의 행 수 확인
        print(f"\n제거할 section_name 값들:")
        for section in sections_to_remove:
            count = (df['section_name'] == section).sum()
            print(f"  - '{section}': {count:,} 행")
        
        # 해당 섹션들을 제외한 데이터 필터링
        df_filtered = df[~df['section_name'].isin(sections_to_remove)]
        
        removed_count = len(df) - len(df_filtered)
        print(f"\n제거된 총 행 수: {removed_count:,}")
        print(f"제거 후 데이터 크기: {df_filtered.shape[0]:,} 행 x {df_filtered.shape[1]} 칼럼")
        
        # 결과 저장
        output_file = 'dataset/hnm/articles_with_price.csv'
        df_filtered.to_csv(output_file, index=False)
        print(f"\n[완료] 행 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def remove_specific_product_groups():
    print("\n=== articles_with_price.csv 특정 product_group_name 행 제거 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 제거할 product_group_name 값들
        product_groups_to_remove = [
            'Accessories',
            'Underwear',
            'Swimwear',
            'Socks & Tights',
            'Cosmetic',
            'Bags',
            'Furniture',
            'Garment and Shoe care',
            'Stationery',
            'Interior textile',
            'Fun'
        ]
        
        # 제거 전 각 product_group의 행 수 확인
        print(f"\n제거할 product_group_name 값들:")
        for group in product_groups_to_remove:
            count = (df['product_group_name'] == group).sum()
            print(f"  - '{group}': {count:,} 행")
        
        # 해당 product_group들을 제외한 데이터 필터링
        df_filtered = df[~df['product_group_name'].isin(product_groups_to_remove)]
        
        removed_count = len(df) - len(df_filtered)
        print(f"\n제거된 총 행 수: {removed_count:,}")
        print(f"제거 후 데이터 크기: {df_filtered.shape[0]:,} 행 x {df_filtered.shape[1]} 칼럼")
        
        # 결과 저장
        output_file = 'dataset/hnm/articles_with_price.csv'
        df_filtered.to_csv(output_file, index=False)
        print(f"\n[완료] 행 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def remove_specific_garment_groups():
    print("\n=== articles_with_price.csv 특정 garment_group_name 행 제거 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 제거할 garment_group_name 값들
        garment_groups_to_remove = [
            'Accessories',
            'Socks and Tights'
        ]
        
        # 제거 전 각 garment_group의 행 수 확인
        print(f"\n제거할 garment_group_name 값들:")
        for group in garment_groups_to_remove:
            count = (df['garment_group_name'] == group).sum()
            print(f"  - '{group}': {count:,} 행")
        
        # 해당 garment_group들을 제외한 데이터 필터링
        df_filtered = df[~df['garment_group_name'].isin(garment_groups_to_remove)]
        
        removed_count = len(df) - len(df_filtered)
        print(f"\n제거된 총 행 수: {removed_count:,}")
        print(f"제거 후 데이터 크기: {df_filtered.shape[0]:,} 행 x {df_filtered.shape[1]} 칼럼")
        
        # 결과 저장
        output_file = 'dataset/hnm/articles_with_price.csv'
        df_filtered.to_csv(output_file, index=False)
        print(f"\n[완료] 행 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 1단계: 특정 section_name 행 제거
    if remove_specific_sections():
        # 2단계: 특정 product_group_name 행 제거
        if remove_specific_product_groups():
            # 3단계: 특정 garment_group_name 행 제거
            remove_specific_garment_groups()

