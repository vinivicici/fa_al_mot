#!/usr/bin/env python3
"""
통합 데이터셋의 카테고리 칼럼들을 원핫인코딩하는 스크립트
- normalized_gender
- normalized_category  
- normalized_usage
"""

import pandas as pd
import os
from datetime import datetime

def one_hot_encode_categories():
    """카테고리 칼럼 원핫인코딩 함수"""
    print("=== 통합 데이터셋 카테고리 칼럼 원핫인코딩 ===")
    
    try:
        # CSV 파일 로드
        print("\n1. 통합 데이터셋 로딩 중...")
        df = pd.read_csv('dataset/merged_dataset.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 원핫인코딩할 칼럼들
        categorical_columns = [
            'normalized_gender',
            'normalized_category',
            'normalized_usage'
        ]
        
        # 존재하는 칼럼만 필터링
        existing_columns = [col for col in categorical_columns if col in df.columns]
        missing_columns = [col for col in categorical_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n[경고] 존재하지 않는 칼럼들: {missing_columns}")
        
        if not existing_columns:
            print("[오류] 원핫인코딩할 칼럼이 없습니다.")
            return False
        
        print(f"\n2. 원핫인코딩할 칼럼들 ({len(existing_columns)}개):")
        for col in existing_columns:
            unique_count = df[col].nunique()
            unique_values = df[col].unique()[:5]  # 처음 5개 값만 표시
            print(f"  - {col}: {unique_count}개 고유값 {unique_values}")
        
        # 원핫인코딩 전 다른 칼럼들 저장 (original_gender, original_category 제외)
        columns_to_drop = ['original_gender', 'original_category']
        existing_drop_columns = [col for col in columns_to_drop if col in df.columns]
        columns_to_keep = [col for col in df.columns if col not in existing_columns and col not in columns_to_drop]
        
        if existing_drop_columns:
            print(f"\n   제거할 칼럼들: {existing_drop_columns}")
        df_other = df[columns_to_keep].copy()
        
        print(f"\n3. 원핫인코딩 수행 중...")
        
        # 각 칼럼별로 원핫인코딩
        encoded_dfs = []
        total_encoded_columns = 0
        
        for col in existing_columns:
            print(f"  - {col} 인코딩 중...")
            # get_dummies로 원핫인코딩 (prefix 추가)
            encoded = pd.get_dummies(df[col], prefix=col, dtype=int)
            encoded_dfs.append(encoded)
            total_encoded_columns += len(encoded.columns)
            print(f"    [완료] {len(encoded.columns)}개 칼럼 생성")
        
        # 모든 인코딩된 칼럼 합치기
        df_encoded = pd.concat(encoded_dfs, axis=1)
        
        # 다른 칼럼들과 합치기
        df_final = pd.concat([df_other, df_encoded], axis=1)
        
        print(f"\n4. 인코딩 완료!")
        print(f"  - 원본 칼럼 수: {df.shape[1]}")
        print(f"  - 원핫인코딩 대상 제거: {len(existing_columns)}개")
        print(f"  - original 칼럼 제거: {len(existing_drop_columns)}개")
        print(f"  - 추가된 칼럼: {total_encoded_columns}개")
        print(f"  - 최종 칼럼 수: {df_final.shape[1]}")
        print(f"  - 최종 데이터 크기: {df_final.shape[0]:,} 행 x {df_final.shape[1]} 칼럼")
        
        # 칼럼 타입 확인
        print(f"\n5. 칼럼 타입 요약:")
        print(f"  - 숫자형 칼럼: {df_final.select_dtypes(include=['int64', 'float64']).shape[1]}개")
        print(f"  - 문자형 칼럼: {df_final.select_dtypes(include=['object']).shape[1]}개")
        
        # 남은 문자형 칼럼 확인
        remaining_text_columns = df_final.select_dtypes(include=['object']).columns.tolist()
        if remaining_text_columns:
            print(f"\n  남은 문자형 칼럼들:")
            for col in remaining_text_columns:
                print(f"    - {col}")
        
        # 원핫인코딩된 칼럼 통계
        print(f"\n6. 원핫인코딩된 칼럼 통계:")
        for col in existing_columns:
            encoded_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
            print(f"  - {col}: {len(encoded_cols)}개 칼럼")
            # 각 칼럼별 활성화된 개수 확인
            for encoded_col in encoded_cols[:3]:  # 처음 3개만 표시
                active_count = df_encoded[encoded_col].sum()
                print(f"    {encoded_col}: {active_count:,}개 활성화")
        
        # 결과 저장
        output_file = 'dataset/merged_dataset_onehot.csv'
        print(f"\n7. 결과 저장 중...")
        df_final.to_csv(output_file, index=False)
        print(f"[완료] 원핫인코딩 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        # 파일 크기 확인
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024**2)
            print(f"파일 크기: {size_mb:.1f} MB")
        
        # 샘플 확인
        print(f"\n8. 인코딩된 칼럼 샘플 (처음 10개):")
        encoded_sample_cols = df_encoded.columns[:10].tolist()
        print(f"  {encoded_sample_cols}")
        
        # 데이터셋별 원핫인코딩 결과 확인
        print(f"\n9. 데이터셋별 원핫인코딩 결과:")
        for source in ['HNM', 'FASHION']:
            subset = df_final[df_final['dataset_source'] == source]
            print(f"  {source}: {len(subset):,}개 행")
            
            # 각 카테고리별 활성화된 칼럼 수 확인
            for col in existing_columns:
                encoded_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
                active_cols = subset[encoded_cols].sum().sum()
                print(f"    {col}: {active_cols:,}개 활성화")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    start_time = datetime.now()
    print("="*80)
    print("통합 데이터셋 원핫인코딩 스크립트 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 전제 조건 확인
    if not os.path.exists('dataset/merged_dataset.csv'):
        print("[오류] 통합 데이터셋 파일이 없습니다: dataset/merged_dataset.csv")
        print("먼저 데이터셋 병합을 실행하세요.")
        return
    
    success = one_hot_encode_categories()
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    if success:
        print("[완료] 통합 데이터셋 원핫인코딩 완료!")
    else:
        print("[오류] 통합 데이터셋 원핫인코딩 실패!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")

if __name__ == "__main__":
    main()
