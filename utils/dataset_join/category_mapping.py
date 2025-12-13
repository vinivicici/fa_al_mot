#!/usr/bin/env python3
"""
카테고리/성별 매핑 스크립트
H&M과 Fashion 데이터셋의 카테고리와 성별을 통일된 형태로 매핑
"""

import pandas as pd
import os
from datetime import datetime

def create_mapping_tables():
    """매핑 테이블 생성"""
    
    # 성별 매핑 테이블 (woman, man, unisex)
    gender_mapping = {
        # H&M 매핑
        'HNM_Ladieswear': 'woman',
        'HNM_Menswear': 'man',
        'HNM_Baby/Children': 'unisex',
        'HNM_Divided': 'unisex',
        'HNM_Sport': 'unisex',
        
        # Fashion 매핑
        'FASHION_Women': 'woman',
        'FASHION_Men': 'man',
        'FASHION_Girls': 'woman',
        'FASHION_Boys': 'man',
        'FASHION_Unisex': 'unisex'
    }

    # 연령 매핑 테이블 (children, teen, adult)
    age_mapping = {
        # H&M 매핑
        'HNM_Ladieswear': 'adult',
        'HNM_Menswear': 'adult',
        'HNM_Baby/Children': 'children',
        'HNM_Divided': 'teen',
        'HNM_Sport': 'adult',
        
        # Fashion 매핑
        'FASHION_Women': 'adult',
        'FASHION_Men': 'adult',
        'FASHION_Girls': 'teen',
        'FASHION_Boys': 'teen',
        'FASHION_Unisex': 'adult'
    }
    
    # 카테고리 매핑 테이블
    category_mapping = {
        # H&M 매핑
        'HNM_Jersey Fancy': 'Tops',
        'HNM_Blouses': 'Shirts',
        'HNM_Shoes': 'Shoes',
        'HNM_Accessories': 'Accessories',
        'HNM_Trousers': 'Bottomwear',
        'HNM_Under-, Nightwear': 'Innerwear',
        'HNM_Unknown': 'Sports',
        'HNM_Knitwear': 'Tops',
        'HNM_Dresses Ladies': 'Dresses',
        'HNM_Outdoor': 'Outdoor',
        
        # Fashion 매핑
        'FASHION_Tshirts': 'Tops',
        'FASHION_Shirts': 'Shirts',
        'FASHION_Casual Shoes': 'Shoes',
        'FASHION_Watches': 'Accessories',
        'FASHION_Sports Shoes': 'Shoes',
        'FASHION_Kurtas': 'Ethnic',
        'FASHION_Tops': 'Tops',
        'FASHION_Handbags': 'Accessories',
        'FASHION_Heels': 'Shoes',
        'FASHION_Sunglasses': 'Accessories'
    }
    
    # 용도 매핑 테이블
    usage_mapping = {
        # H&M 매핑
        'HNM_Sport': 'Sports',
        'HNM_Blouses': 'Formal',
        'HNM_Shirts': 'Formal',
        'HNM_Unknown': 'Sports',
        'HNM_Dresses Ladies': 'Formal',
        
        # Fashion 매핑
        'FASHION_Sports': 'Sports',
        'FASHION_Ethnic': 'Ethnic',
        'FASHION_Formal': 'Formal',
        'FASHION_Casual': 'Casual',
        'FASHION_Smart Casual': 'Smart Casual',
        'FASHION_Party': 'Party',
        'FASHION_Travel': 'Travel',
        'FASHION_Home': 'Home'
    }
    
    return gender_mapping, age_mapping, category_mapping, usage_mapping

def apply_mappings():
    """매핑 적용 함수"""
    print("=== 카테고리/성별 매핑 시작 ===")
    
    try:
        # 매핑 테이블 생성
        gender_mapping, age_mapping, category_mapping, usage_mapping = create_mapping_tables()
        
        # H&M 데이터 로드
        print("1. H&M 데이터 로딩...")
        hnm_df = pd.read_csv('dataset/hnm/articles_with_price_unified.csv')
        print(f"   H&M 데이터: {len(hnm_df):,}개 행")
        
        # H&M 매핑 적용
        print("2. H&M 매핑 적용...")
        
        # 성별/연령 매핑
        hnm_df['normalized_gender'] = hnm_df['index_group_name'].apply(
            lambda x: gender_mapping.get(f'HNM_{x}', 'unknown')
        )
        hnm_df['normalized_age'] = hnm_df['index_group_name'].apply(
            lambda x: age_mapping.get(f'HNM_{x}', 'unknown')
        )
        
        # 카테고리 매핑
        hnm_df['normalized_category'] = hnm_df['garment_group_name'].apply(
            lambda x: category_mapping.get(f'HNM_{x}', 'Other')
        )
        
        # 용도 매핑 (garment_group_name 기반)
        hnm_df['normalized_usage'] = hnm_df['garment_group_name'].apply(
            lambda x: usage_mapping.get(f'HNM_{x}', 'Casual')
        )
        
        # Fashion 데이터 로드
        print("3. Fashion 데이터 로딩...")
        fashion_df = pd.read_csv('dataset/fashion/fashion_pricied.csv')
        print(f"   Fashion 데이터: {len(fashion_df):,}개 행")
        
        # Fashion 매핑 적용
        print("4. Fashion 매핑 적용...")
        
        # 성별/연령 매핑
        fashion_df['normalized_gender'] = fashion_df['gender'].apply(
            lambda x: gender_mapping.get(f'FASHION_{x}', 'unknown')
        )
        fashion_df['normalized_age'] = fashion_df['gender'].apply(
            lambda x: age_mapping.get(f'FASHION_{x}', 'unknown')
        )
        
        # 카테고리 매핑 (articleType 기반)
        fashion_df['normalized_category'] = fashion_df['articleType'].apply(
            lambda x: category_mapping.get(f'FASHION_{x}', 'Other')
        )
        
        # 용도 매핑
        fashion_df['normalized_usage'] = fashion_df['usage'].apply(
            lambda x: usage_mapping.get(f'FASHION_{x}', 'Casual')
        )
        
        # 매핑 결과 확인
        print("5. 매핑 결과 확인...")
        
        print("   H&M 성별 분포:")
        print(hnm_df['normalized_gender'].value_counts())
        print("   H&M 연령 분포:")
        print(hnm_df['normalized_age'].value_counts())
        
        print("   H&M 카테고리 분포:")
        print(hnm_df['normalized_category'].value_counts())
        
        print("   H&M 용도 분포:")
        print(hnm_df['normalized_usage'].value_counts())
        
        print("   Fashion 성별 분포:")
        print(fashion_df['normalized_gender'].value_counts())
        print("   Fashion 연령 분포:")
        print(fashion_df['normalized_age'].value_counts())
        
        print("   Fashion 카테고리 분포:")
        print(fashion_df['normalized_category'].value_counts())
        
        print("   Fashion 용도 분포:")
        print(fashion_df['normalized_usage'].value_counts())
        
        # 결과 저장
        print("6. 결과 저장...")
        
        hnm_output_path = 'dataset/hnm/articles_with_price_mapped.csv'
        hnm_df.to_csv(hnm_output_path, index=False)
        print(f"   H&M 매핑된 데이터 저장: {hnm_output_path}")
        
        fashion_output_path = 'dataset/fashion/fashion_mapped.csv'
        fashion_df.to_csv(fashion_output_path, index=False)
        print(f"   Fashion 매핑된 데이터 저장: {fashion_output_path}")
        
        return True
        
    except Exception as e:
        print(f"[오류] 매핑 적용 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    start_time = datetime.now()
    print("="*80)
    print("카테고리/성별 매핑 스크립트 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = apply_mappings()
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    if success:
        print("[완료] 카테고리/성별 매핑 완료!")
    else:
        print("[오류] 카테고리/성별 매핑 실패!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")

if __name__ == "__main__":
    main()
