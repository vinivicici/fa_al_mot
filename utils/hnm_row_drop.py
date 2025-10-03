#!/usr/bin/env python3
"""
articles_with_price.csv 전처리 스크립트
- 특정 section_name 값을 가진 행 제거 (악세서리, 속옷)
- 특정 product_group_name 값을 가진 행 제거 (속옷, 수영복, 양말, 화장품, 가방, 가구 등)
- 특정 garment_group_name 값을 가진 행 제거 (악세서리, 양말)
- 가격 스케일링 (1000배로 실제 가격으로 변환)
"""

import pandas as pd

def remove_specific_sections():
    print("=== articles_with_price.csv 특정 section_name 행 제거 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('articles_with_price.csv')
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
        output_file = 'articles_with_price.csv'
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
        df = pd.read_csv('articles_with_price.csv')
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
        output_file = 'articles_with_price.csv'
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
        df = pd.read_csv('articles_with_price.csv')
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
        output_file = 'articles_with_price.csv'
        df_filtered.to_csv(output_file, index=False)
        print(f"\n[완료] 행 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def rescale_price(scale_factor=1000):
    """
    스케일된 price 값을 실제 가격으로 변환
    
    분석 결과:
    - 원본 데이터의 price는 개인정보 보호를 위해 1000으로 나눠진 값
    - 예: 0.01899 -> 18.99 (탱크톱 ~8, 브라 10-20, 바지 22-30)
    - H&M은 스웨덴 회사이므로 유로 또는 SEK(스웨덴 크로나)일 가능성 높음
    """
    print(f"\n=== price 칼럼 스케일링 (x{scale_factor}) ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('articles_with_price.csv')
        print(f"데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 원본 가격 통계
        print(f"\n원본 price 통계:")
        print(f"  - 최소값: {df['price'].min():.6f}")
        print(f"  - 평균값: {df['price'].mean():.6f}")
        print(f"  - 중간값: {df['price'].median():.6f}")
        print(f"  - 최대값: {df['price'].max():.6f}")
        
        # 가격 스케일링
        df['price'] = df['price'] * scale_factor
        
        # 스케일링 후 가격 통계
        print(f"\n스케일링 후 price 통계:")
        print(f"  - 최소값: {df['price'].min():.2f}")
        print(f"  - 평균값: {df['price'].mean():.2f}")
        print(f"  - 중간값: {df['price'].median():.2f}")
        print(f"  - 최대값: {df['price'].max():.2f}")
        
        # 샘플 확인
        print(f"\n샘플 제품 가격:")
        sample = df[['product_type_name', 'price']].head(10)
        for idx, row in sample.iterrows():
            print(f"  - {row['product_type_name']}: {row['price']:.2f}")
        
        # 결과 저장
        output_file = 'articles_with_price.csv'
        df.to_csv(output_file, index=False)
        print(f"\n[완료] 가격 스케일링 완료!")
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
            if remove_specific_garment_groups():
                # 4단계: 가격 스케일링
                rescale_price()

