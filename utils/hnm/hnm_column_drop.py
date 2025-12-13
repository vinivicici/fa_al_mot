#!/usr/bin/env python3
"""
articles_with_price.csv에서 불필요한 칼럼들을 제거하는 스크립트
"""

import pandas as pd

def drop_columns():
    print("=== articles_with_price.csv 칼럼 제거 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 제거할 칼럼들
        columns_to_drop = [
            'prod_name', 'article_id', 'product_type_no', 'graphical_appearance_no',
            'colour_group_code', 'colour_group_name', 'perceived_colour_value_id',
            'perceived_colour_value_name', 'perceived_colour_master_id', 
            'perceived_colour_master_name', 'department_no', 'index_code',
            'index_group_no', 'section_no', 'garment_group_no', 'graphical_appearance_name',
            'department_name', 'index_name',
            # 하위군 드랍
            'section_name', 'product_type_name'
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
        
        # description에 소분류 정보와 브랜드 추가
        print("\ndescription에 추가 정보 병합 중...")
        if 'detail_desc' in df.columns:
            def enrich_description(row):
                original_desc = str(row.get('detail_desc', '')) if pd.notna(row.get('detail_desc')) else ''
                
                # 추가할 정보 수집
                additions = []
                
                # section_name 추가
                if 'section_name' in row and pd.notna(row['section_name']):
                    additions.append(f"section_name: {row['section_name']}")
                
                # product_type_name 추가
                if 'product_type_name' in row and pd.notna(row['product_type_name']):
                    additions.append(f"product_type_name: {row['product_type_name']}")
                
                # 브랜드 추가 (항상 HNM)
                additions.append("brand: HNM")
                
                # description에 추가 정보 병합
                if additions:
                    additional_info = "\n".join(additions)
                    if original_desc:
                        enriched_desc = f"{original_desc}\n\n{additional_info}"
                    else:
                        enriched_desc = additional_info
                    return enriched_desc
                else:
                    return original_desc
            
            df['detail_desc'] = df.apply(enrich_description, axis=1)
            print("  [완료] section_name, product_type_name, brand 정보를 description에 추가했습니다.")
        
        # 칼럼 제거
        df_dropped = df.drop(columns=existing_columns_to_drop)
        
        print(f"\n제거 후 데이터 크기: {df_dropped.shape[0]:,} 행 x {df_dropped.shape[1]} 칼럼")
        print(f"제거된 칼럼 수: {len(existing_columns_to_drop)}개")
        
        # 남은 칼럼들 출력
        print(f"\n남은 칼럼들 ({len(df_dropped.columns)}개):")
        for i, col in enumerate(df_dropped.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 결과 저장
        output_file = 'dataset/hnm/articles_with_price.csv'
        df_dropped.to_csv(output_file, index=False)
        print(f"\n[완료] 칼럼 제거 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        return False

if __name__ == "__main__":
    drop_columns()
