#!/usr/bin/env python3
"""
가격 통일 스크립트
H&M과 Fashion 데이터의 가격을 590을 곱하여 정규화
Kaggle 토론 참고: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/310496
"""

import pandas as pd
import os
from datetime import datetime

def unify_prices():
    """가격 통일 함수"""
    print("=== 가격 통일 시작 ===")
    
    # 가격 정규화 계수 (Kaggle 토론 참고)
    PRICE_NORMALIZATION_FACTOR = 590
    
    try:
        # H&M 데이터 로드
        print("1. H&M 데이터 로딩...")
        hnm_df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"   H&M 데이터: {len(hnm_df):,}개 행")
        
        # H&M 가격 통일
        print("2. H&M 가격 통일 (스케일링된 값 × 590)...")
        # 스케일링된 값 × 590 = 정규화된 가격
        hnm_df['price_usd'] = hnm_df['price'] * PRICE_NORMALIZATION_FACTOR
        
        # 가격 통계 출력
        print(f"   원본 스케일링된 가격 범위: {hnm_df['price'].min():.6f} ~ {hnm_df['price'].max():.6f}")
        print(f"   변환된 달러 가격 범위: ${hnm_df['price_usd'].min():.2f} ~ ${hnm_df['price_usd'].max():.2f}")
        print(f"   변환된 달러 가격 평균: ${hnm_df['price_usd'].mean():.2f}")
        
        # Fashion 데이터 로드
        print("3. Fashion 데이터 로딩...")
        fashion_df = pd.read_csv('dataset/fashion/fashion.csv')
        print(f"   Fashion 데이터: {len(fashion_df):,}개 행")
        
        # Fashion 가격 통일 (원본 가격 그대로 사용)
        print("4. Fashion 가격 통일 (원본 가격 그대로 사용)...")
        fashion_df['price_usd'] = fashion_df['discountedPrice']
        
        # 가격 통계 출력
        print(f"   Fashion 달러 가격 범위: ${fashion_df['price_usd'].min():.2f} ~ ${fashion_df['price_usd'].max():.2f}")
        print(f"   Fashion 달러 가격 평균: ${fashion_df['price_usd'].mean():.2f}")
        
        # 통일된 가격 범위 확인
        print("5. 통일된 가격 범위 확인...")
        all_prices = pd.concat([hnm_df['price_usd'], fashion_df['price_usd']])
        print(f"   전체 통일된 가격 범위: ${all_prices.min():.2f} ~ ${all_prices.max():.2f}")
        print(f"   전체 통일된 가격 평균: ${all_prices.mean():.2f}")
        
        # 데이터셋 소스 칼럼 추가
        hnm_df['dataset_source'] = 'HNM'
        fashion_df['dataset_source'] = 'FASHION'
        
        # 결과 저장
        print("6. 결과 저장...")
        
        # H&M 결과 저장
        hnm_output_path = 'dataset/hnm/articles_with_price_unified.csv'
        hnm_df.to_csv(hnm_output_path, index=False)
        print(f"   H&M 통일된 데이터 저장: {hnm_output_path}")
        
        # Fashion 결과 저장
        fashion_output_path = 'dataset/fashion/fashion_unified.csv'
        fashion_df.to_csv(fashion_output_path, index=False)
        print(f"   Fashion 통일된 데이터 저장: {fashion_output_path}")
        
        # 요약 정보 출력
        print("\n=== 가격 통일 완료 ===")
        print(f"H&M 가격 정규화 계수: {PRICE_NORMALIZATION_FACTOR}")
        print(f"H&M 변환 공식: 스케일링된_가격 × {PRICE_NORMALIZATION_FACTOR}")
        print(f"Fashion 변환 공식: 원본_가격 그대로 사용")
        print(f"최종 통일된 가격 단위: 혼합 (H&M 정규화, Fashion 원본)")
        
        return True
        
    except Exception as e:
        print(f"[오류] 가격 통일 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    start_time = datetime.now()
    print("="*80)
    print("가격 통일 스크립트 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = unify_prices()
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    if success:
        print("[완료] 가격 통일 완료!")
    else:
        print("[오류] 가격 통일 실패!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")

if __name__ == "__main__":
    main()
