#!/usr/bin/env python3
"""
articles_with_price.csv의 카테고리 칼럼들을 원핫인코딩하는 스크립트
- product_type_name
- garment_group_name
- index_group_name
- section_name
"""

import pandas as pd

def one_hot_encode_categories():
    print("=== 카테고리 칼럼 원핫인코딩 ===")
    
    try:
        # CSV 파일 로드
        print("\n1. CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"원본 데이터 크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 원핫인코딩할 칼럼들
        categorical_columns = [
            'product_type_name',
            'garment_group_name',
            'index_group_name',
            'section_name'
        ]
        
        # 존재하는 칼럼만 필터링
        existing_columns = [col for col in categorical_columns if col in df.columns]
        missing_columns = [col for col in categorical_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n⚠️ 존재하지 않는 칼럼들: {missing_columns}")
        
        if not existing_columns:
            print("[오류] 원핫인코딩할 칼럼이 없습니다.")
            return False
        
        print(f"\n2. 원핫인코딩할 칼럼들 ({len(existing_columns)}개):")
        for col in existing_columns:
            unique_count = df[col].nunique()
            print(f"  - {col}: {unique_count}개 고유값")
        
        # 원핫인코딩 전 다른 칼럼들 저장
        non_categorical_columns = [col for col in df.columns if col not in existing_columns]
        df_other = df[non_categorical_columns].copy()
        
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
            print(f"    ✓ {len(encoded.columns)}개 칼럼 생성")
        
        # 모든 인코딩된 칼럼 합치기
        df_encoded = pd.concat(encoded_dfs, axis=1)
        
        # 다른 칼럼들과 합치기
        df_final = pd.concat([df_other, df_encoded], axis=1)
        
        print(f"\n4. 인코딩 완료!")
        print(f"  - 원본 칼럼 수: {df.shape[1]}")
        print(f"  - 제거된 칼럼: {len(existing_columns)}개")
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
        
        # 결과 저장
        output_file = 'dataset/hnm/articles_with_price.csv'
        print(f"\n6. 결과 저장 중...")
        df_final.to_csv(output_file, index=False)
        print(f"[완료] 원핫인코딩 완료!")
        print(f"결과가 '{output_file}'에 저장되었습니다.")
        
        # 샘플 확인
        print(f"\n7. 인코딩된 칼럼 샘플 (처음 5개):")
        encoded_sample_cols = df_encoded.columns[:5].tolist()
        print(f"  {encoded_sample_cols}")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    one_hot_encode_categories()

