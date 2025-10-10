#!/usr/bin/env python3
"""
데이터셋 병합 스크립트
H&M과 Fashion 데이터셋을 통합된 형태로 병합
"""

import pandas as pd
import os
from datetime import datetime

def merge_datasets():
    """데이터셋 병합 함수"""
    print("=== 데이터셋 병합 시작 ===")
    
    try:
        # 매핑된 데이터 로드
        print("1. 매핑된 데이터 로딩...")
        
        hnm_df = pd.read_csv('dataset/hnm/articles_with_price_mapped.csv')
        fashion_df = pd.read_csv('dataset/fashion/fashion_mapped.csv')
        
        print(f"   H&M 데이터: {len(hnm_df):,}개 행 x {len(hnm_df.columns)}개 칼럼")
        print(f"   Fashion 데이터: {len(fashion_df):,}개 행 x {len(fashion_df.columns)}개 칼럼")
        
        # 공통 스키마 정의
        print("2. 공통 스키마 정의...")
        
        # H&M 데이터 정리
        hnm_clean = pd.DataFrame({
            'product_id': hnm_df['product_code'],
            'dataset_source': hnm_df['dataset_source'],
            'price_usd': hnm_df['price_usd'],
            'normalized_gender': hnm_df['normalized_gender'],
            'normalized_category': hnm_df['normalized_category'],
            'normalized_usage': hnm_df['normalized_usage'],
            'description': hnm_df['detail_desc'],
            'original_gender': hnm_df['index_group_name'],
            'original_category': hnm_df['garment_group_name']
        })
        
        # Fashion 데이터 정리
        fashion_clean = pd.DataFrame({
            'product_id': fashion_df['id'],
            'dataset_source': fashion_df['dataset_source'],
            'price_usd': fashion_df['price_usd'],
            'normalized_gender': fashion_df['normalized_gender'],
            'normalized_category': fashion_df['normalized_category'],
            'normalized_usage': fashion_df['normalized_usage'],
            'description': fashion_df['description'],
            'original_gender': fashion_df['gender'],
            'original_category': fashion_df['articleType']
        })
        
        print(f"   H&M 정리된 데이터: {len(hnm_clean):,}개 행 x {len(hnm_clean.columns)}개 칼럼")
        print(f"   Fashion 정리된 데이터: {len(fashion_clean):,}개 행 x {len(fashion_clean.columns)}개 칼럼")
        
        # 데이터 병합
        print("3. 데이터 병합...")
        
        merged_df = pd.concat([hnm_clean, fashion_clean], ignore_index=True)
        
        print(f"   병합된 데이터: {len(merged_df):,}개 행 x {len(merged_df.columns)}개 칼럼")
        
        # 데이터 품질 확인
        print("4. 데이터 품질 확인...")
        
        print("   데이터셋 소스 분포:")
        print(merged_df['dataset_source'].value_counts())
        
        print("   통일된 성별 분포:")
        print(merged_df['normalized_gender'].value_counts())
        
        print("   통일된 카테고리 분포:")
        print(merged_df['normalized_category'].value_counts())
        
        print("   통일된 용도 분포:")
        print(merged_df['normalized_usage'].value_counts())
        
        print("   가격 통계:")
        print(f"     최소: ${merged_df['price_usd'].min():.2f}")
        print(f"     최대: ${merged_df['price_usd'].max():.2f}")
        print(f"     평균: ${merged_df['price_usd'].mean():.2f}")
        print(f"     중간값: ${merged_df['price_usd'].median():.2f}")
        
        # 결측값 확인
        print("   결측값 확인:")
        missing_data = merged_df.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                print(f"     {col}: {missing_count:,}개 ({missing_count/len(merged_df)*100:.1f}%)")
        
        # 중복 제거
        print("5. 중복 제거...")
        original_count = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['product_id', 'dataset_source'])
        final_count = len(merged_df)
        removed_count = original_count - final_count
        
        print(f"   원본 행 수: {original_count:,}")
        print(f"   중복 제거 후: {final_count:,}")
        print(f"   제거된 중복: {removed_count:,}")
        
        # 결과 저장
        print("6. 결과 저장...")
        
        output_path = 'dataset/merged_dataset.csv'
        merged_df.to_csv(output_path, index=False)
        
        print(f"   병합된 데이터셋 저장: {output_path}")
        
        # 요약 통계 저장
        summary_path = 'dataset/merge_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 데이터셋 병합 요약 ===\n")
            f.write(f"병합 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 행 수: {final_count:,}\n")
            f.write(f"총 칼럼 수: {len(merged_df.columns)}\n")
            f.write(f"제거된 중복: {removed_count:,}\n\n")
            
            f.write("데이터셋 소스 분포:\n")
            for source, count in merged_df['dataset_source'].value_counts().items():
                f.write(f"  {source}: {count:,}개 ({count/final_count*100:.1f}%)\n")
            
            f.write("\n통일된 성별 분포:\n")
            for gender, count in merged_df['normalized_gender'].value_counts().items():
                f.write(f"  {gender}: {count:,}개 ({count/final_count*100:.1f}%)\n")
            
            f.write("\n통일된 카테고리 분포:\n")
            for category, count in merged_df['normalized_category'].value_counts().items():
                f.write(f"  {category}: {count:,}개 ({count/final_count*100:.1f}%)\n")
            
            f.write("\n통일된 용도 분포:\n")
            for usage, count in merged_df['normalized_usage'].value_counts().items():
                f.write(f"  {usage}: {count:,}개 ({count/final_count*100:.1f}%)\n")
            
            f.write(f"\n가격 통계:\n")
            f.write(f"  최소: ${merged_df['price_usd'].min():.2f}\n")
            f.write(f"  최대: ${merged_df['price_usd'].max():.2f}\n")
            f.write(f"  평균: ${merged_df['price_usd'].mean():.2f}\n")
            f.write(f"  중간값: ${merged_df['price_usd'].median():.2f}\n")
        
        print(f"   병합 요약 저장: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"[오류] 데이터셋 병합 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    start_time = datetime.now()
    print("="*80)
    print("데이터셋 병합 스크립트 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = merge_datasets()
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    if success:
        print("[완료] 데이터셋 병합 완료!")
    else:
        print("[오류] 데이터셋 병합 실패!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")

if __name__ == "__main__":
    main()
