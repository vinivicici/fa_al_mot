#!/usr/bin/env python3
"""
clip_clusters_p256.csv 파일을 merged_dataset_onehot.csv의 product_id 기준으로 마스킹하는 스크립트

동작:
- 입력: dataset/clip_clusters_p256.csv
- 마스크 기준: dataset/merged_dataset_onehot.csv의 product_id
- 출력: dataset/clip_clusters_p256_masked.csv (마스킹된 결과)
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime


def mask_clip_clusters(
    clip_clusters_csv: str = 'dataset/clip_clusters_p256.csv',
    merged_dataset_csv: str = 'dataset/merged_dataset_onehot.csv',
    output_csv: str = 'dataset/clip_clusters_p256_masked.csv'
) -> bool:
    """
    clip_clusters_p256.csv를 merged_dataset_onehot.csv의 product_id로 마스킹
    
    Args:
        clip_clusters_csv: 입력 CSV 파일 경로
        merged_dataset_csv: 마스크 기준 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
    
    Returns:
        성공 여부 (bool)
    """
    print("=== CLIP Clusters 마스킹 시작 ===")
    print(f"입력 파일: {clip_clusters_csv}")
    print(f"마스크 기준 파일: {merged_dataset_csv}")
    print(f"출력 파일: {output_csv}")
    
    # 파일 존재 확인
    if not os.path.exists(clip_clusters_csv):
        print(f"[오류] 입력 CSV가 없습니다: {clip_clusters_csv}")
        return False
    
    if not os.path.exists(merged_dataset_csv):
        print(f"[오류] 마스크 기준 CSV가 없습니다: {merged_dataset_csv}")
        return False
    
    try:
        # CLIP clusters 데이터 로드
        print("\n1. CLIP clusters 데이터 로딩...")
        clip_df = pd.read_csv(clip_clusters_csv)
        print(f"   CLIP clusters 데이터: {len(clip_df):,}개 행 x {len(clip_df.columns)}개 칼럼")
        
        # product_id 컬럼 확인
        if 'product_id' not in clip_df.columns:
            print("[오류] clip_clusters_p256.csv에 'product_id' 컬럼이 없습니다.")
            print(f"       존재하는 컬럼: {list(clip_df.columns)}")
            return False
        
        # merged_dataset 데이터 로드 (product_id만 필요)
        print("\n2. merged_dataset 데이터 로딩 (product_id만)...")
        merged_df = pd.read_csv(merged_dataset_csv, usecols=['product_id'])
        print(f"   merged_dataset product_id: {len(merged_df):,}개")
        
        # product_id 컬럼 확인
        if 'product_id' not in merged_df.columns:
            print("[오류] merged_dataset_onehot.csv에 'product_id' 컬럼이 없습니다.")
            return False
        
        # 고유한 product_id 집합 생성
        valid_product_ids = set(merged_df['product_id'].unique())
        print(f"   고유한 product_id 개수: {len(valid_product_ids):,}개")
        
        # 마스킹: clip_clusters에서 valid_product_ids에 포함된 행만 남기기
        print("\n3. 마스킹 수행...")
        original_count = len(clip_df)
        mask = clip_df['product_id'].isin(valid_product_ids)
        masked_df = clip_df[mask].copy()
        masked_count = len(masked_df)
        removed_count = original_count - masked_count
        
        print(f"   원본 행 수: {original_count:,}")
        print(f"   마스킹 후 행 수: {masked_count:,}")
        print(f"   제거된 행 수: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
        
        # 결과 저장
        print("\n4. 결과 저장...")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        masked_df.to_csv(output_csv, index=False)
        print(f"   저장 완료: {output_csv}")
        
        # 통계 정보 출력
        print("\n=== 마스킹 완료 ===")
        print(f"원본 데이터: {original_count:,}개 행")
        print(f"마스킹 후: {masked_count:,}개 행")
        print(f"제거된 행: {removed_count:,}개 ({removed_count/original_count*100:.1f}%)")
        print(f"보존률: {masked_count/original_count*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[오류] 마스킹 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='CLIP clusters를 merged_dataset의 product_id로 마스킹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python utils/dataset_masking.py
  python utils/dataset_masking.py --clip dataset/clip_clusters_p256.csv --merged dataset/merged_dataset_onehot.csv
        """
    )
    parser.add_argument(
        '--clip',
        default='dataset/clip_clusters_p256.csv',
        help='입력 CLIP clusters CSV 경로 (기본: dataset/clip_clusters_p256.csv)'
    )
    parser.add_argument(
        '--merged',
        default='dataset/merged_dataset_onehot.csv',
        help='마스크 기준 merged dataset CSV 경로 (기본: dataset/merged_dataset_onehot.csv)'
    )
    parser.add_argument(
        '--output',
        default='dataset/clip_clusters_p256_masked.csv',
        help='출력 CSV 경로 (기본: dataset/clip_clusters_p256_masked.csv)'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print("="*80)
    print("CLIP Clusters 마스킹 스크립트 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = mask_clip_clusters(args.clip, args.merged, args.output)
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    if success:
        print("[완료] CLIP Clusters 마스킹 완료!")
    else:
        print("[오류] CLIP Clusters 마스킹 실패!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()


