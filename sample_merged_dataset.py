#!/usr/bin/env python3
"""
merged_dataset.csv에서 랜덤하게 1000개 행을 샘플링하여 merged_dataset_sampled.csv를 생성
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def sample_dataset(input_file='dataset/merged_dataset.csv',
                   output_file='dataset/merged_dataset_sampled.csv',
                   sample_size=1000,
                   random_seed=42):
    """
    데이터셋에서 랜덤 샘플링 수행
    
    Parameters:
    -----------
    input_file : str
        입력 CSV 파일 경로
    output_file : str
        출력 CSV 파일 경로
    sample_size : int
        샘플링할 행 수
    random_seed : int
        랜덤 시드 (재현성을 위해)
    """
    print("="*80)
    print("데이터셋 랜덤 샘플링")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 입력 파일 확인
    if not os.path.exists(input_file):
        print(f"\n[오류] 입력 파일을 찾을 수 없습니다: {input_file}")
        return False
    
    try:
        # 데이터 로드
        print(f"\n1. 데이터 로딩: {input_file}")
        df = pd.read_csv(input_file)
        total_rows = len(df)
        print(f"   전체 행 수: {total_rows:,}개")
        print(f"   전체 칼럼 수: {len(df.columns)}개")
        
        # 샘플 크기 확인
        if sample_size > total_rows:
            print(f"\n[경고] 요청한 샘플 크기 ({sample_size:,})가 전체 행 수 ({total_rows:,})보다 큽니다.")
            print(f"        전체 데이터를 샘플로 저장합니다.")
            sample_size = total_rows
        
        # 랜덤 샘플링
        print(f"\n2. 랜덤 샘플링 (n={sample_size:,}, seed={random_seed})...")
        np.random.seed(random_seed)
        sampled_df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        
        print(f"   샘플링된 행 수: {len(sampled_df):,}개")
        
        # 샘플 통계 출력
        print(f"\n3. 샘플 통계:")
        print(f"   칼럼: {list(sampled_df.columns)}")
        
        # 데이터셋 소스 분포
        if 'dataset_source' in sampled_df.columns:
            print(f"\n   데이터셋 소스 분포:")
            source_counts = sampled_df['dataset_source'].value_counts()
            for source, count in source_counts.items():
                print(f"     {source}: {count:,}개 ({count/len(sampled_df)*100:.1f}%)")
                print(f" ")
        
        # 성별 분포
        if 'normalized_gender' in sampled_df.columns:
            print(f"\n   통일된 성별 분포:")
            gender_counts = sampled_df['normalized_gender'].value_counts()
            for gender, count in gender_counts.items():
                print(f"     {gender}: {count:,}개 ({count/len(sampled_df)*100:.1f}%)")
        
        # 가격 통계
        if 'price_usd' in sampled_df.columns:
            print(f"\n   가격 통계:")
            print(f"     최소: ${sampled_df['price_usd'].min():.2f}")
            print(f"     최대: ${sampled_df['price_usd'].max():.2f}")
            print(f"     평균: ${sampled_df['price_usd'].mean():.2f}")
            print(f"     중간값: ${sampled_df['price_usd'].median():.2f}")
        
        # 결과 저장
        print(f"\n4. 결과 저장: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sampled_df.to_csv(output_file, index=False)
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"   저장 완료! 파일 크기: {file_size:.2f} MB")
        
        print("\n" + "="*80)
        print("[완료] 샘플링 완료!")
        print("="*80)
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"출력 파일: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n[오류] 샘플링 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    success = sample_dataset(
        input_file='dataset/merged_dataset.csv',
        output_file='dataset/merged_dataset_sampled.csv',
        sample_size=1000,
        random_seed=42
    )
    
    if success:
        print("\n다음 단계:")
        print("1. dataset/merged_dataset_sampled.csv 파일을 확인하세요")
        print("2. 필요시 다른 샘플 크기로 다시 실행하세요 (sample_size 파라미터 수정)")
    else:
        print("\n샘플링 실패. 오류 메시지를 확인하세요.")

if __name__ == "__main__":
    main()

