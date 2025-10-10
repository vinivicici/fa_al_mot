#!/usr/bin/env python3
"""
H&M과 Fashion 데이터셋의 product 관련 칼럼들의 고유값을 분석하고 정리하는 스크립트
"""

import pandas as pd
from collections import Counter
import os

def analyze_hnm_columns():
    print("=== articles_with_price.csv의 product 칼럼 분석 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('dataset/hnm/articles_with_price.csv')
        print(f"총 {len(df):,}개 행 로드 완료")
        
        # product 관련 칼럼들 식별
        product_columns = [col for col in df.columns if 'product' in col.lower()]
        print(f"\nproduct 관련 칼럼들: {product_columns}")
        
        # 추가로 분석할 중요한 칼럼들
        important_columns = [
            'product_type_name', 'product_group_name', 
            'department_name', 'section_name', 'garment_group_name',
            'colour_group_name', 'index_group_name'
        ]
        
        # 결과를 저장할 딕셔너리
        column_analysis = {}
        
        for col in important_columns:
            if col in df.columns:
                print(f"\n분석 중: {col}")
                unique_values = df[col].value_counts().sort_values(ascending=False)
                
                column_analysis[col] = {
                    'unique_count': len(unique_values),
                    'top_values': unique_values.head(20).to_dict(),
                    'total_count': len(df)
                }
                
                print(f"  고유값 개수: {len(unique_values)}")
                print(f"  상위 5개: {list(unique_values.head(5).index)}")
        
        # 결과를 CSV 파일들로 저장
        print(f"\n=== CSV 파일 생성 중 ===")
        
        # 출력 폴더 생성
        output_dir = 'hnm_column_observation'
        os.makedirs(output_dir, exist_ok=True)
        print(f"[폴더] '{output_dir}' 폴더 생성 완료")
        
        # 1. 전체 요약 파일
        summary_data = []
        for col, info in column_analysis.items():
            summary_data.append({
                'column_name': col,
                'unique_count': info['unique_count'],
                'total_rows': info['total_count'],
                'top_3_values': ', '.join([str(v) if v is not None else '' for v in list(info['top_values'].keys())[:3]])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/product_columns_summary.csv', index=False)
        print("[성공] product_columns_summary.csv 생성 완료")
        
        # 2. 각 칼럼별 상세 파일들
        for col, info in column_analysis.items():
            detail_data = []
            for value, count in info['top_values'].items():
                percentage = (count / info['total_count']) * 100
                detail_data.append({
                    'value': value,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
            
            detail_df = pd.DataFrame(detail_data)
            filename = f"product_{col}_details.csv"
            detail_df.to_csv(f'{output_dir}/{filename}', index=False)
            print(f"[성공] {filename} 생성 완료 ({len(detail_data)}개 고유값)")
        
        # 3. 통합 분석 파일 (모든 칼럼의 상위 값들)
        print(f"\n=== 통합 분석 파일 생성 ===")
        all_analysis = []
        
        for col, info in column_analysis.items():
            for i, (value, count) in enumerate(list(info['top_values'].items())[:10]):
                percentage = (count / info['total_count']) * 100
                all_analysis.append({
                    'column_name': col,
                    'rank': i + 1,
                    'value': value,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
        
        all_df = pd.DataFrame(all_analysis)
        all_df.to_csv(f'{output_dir}/product_all_analysis.csv', index=False)
        print("[성공] product_all_analysis.csv 생성 완료")
        
        # 결과 요약 출력
        print(f"\n" + "="*60)
        print("[분석] 분석 결과 요약:")
        for col, info in column_analysis.items():
            print(f"\n{col}:")
            print(f"  - 고유값: {info['unique_count']}개")
            print(f"  - 상위 3개: {list(info['top_values'].keys())[:3]}")
        
        print(f"\n생성된 파일들 ({output_dir}/ 폴더):")
        print(f"  - product_columns_summary.csv (전체 요약)")
        print(f"  - product_all_analysis.csv (통합 분석)")
        for col in column_analysis.keys():
            print(f"  - product_{col}_details.csv (상세 분석)")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        return False

def analyze_fashion_columns():
    print("=== fashion.csv의 칼럼 분석 ===")
    
    try:
        # CSV 파일 로드 (오류 처리 포함)
        print("CSV 파일 로딩 중...")
        try:
            # 최신 pandas 버전용
            df = pd.read_csv('dataset/fashion/fashion.csv', on_bad_lines='skip', encoding='utf-8')
            print(f"총 {len(df):,}개 행 로드 완료 (문제 있는 줄 건너뜀)")
        except Exception as e:
            print(f"[오류] 최신 방법 실패: {e}")
            print("대안 방법으로 시도 중...")
            try:
                # 구버전 pandas 호환
                df = pd.read_csv('dataset/fashion/fashion.csv', sep=',', quoting=1, encoding='utf-8')
                print(f"총 {len(df):,}개 행 로드 완료 (대안 방법)")
            except Exception as e2:
                print(f"[오류] 대안 방법도 실패: {e2}")
                print("CSV 파일이 손상되었거나 형식이 잘못되었습니다.")
                return False
        
        # 분석할 중요한 칼럼들
        important_columns = [
            'gender', 'masterCategory', 'subCategory', 'articleType',
            'baseColour', 'season', 'year', 'usage', 'productDisplayName'
        ]
        
        # 결과를 저장할 딕셔너리
        column_analysis = {}
        
        for col in important_columns:
            if col in df.columns:
                print(f"\n분석 중: {col}")
                unique_values = df[col].value_counts().sort_values(ascending=False)
                
                column_analysis[col] = {
                    'unique_count': len(unique_values),
                    'top_values': unique_values.head(20).to_dict(),
                    'total_count': len(df)
                }
                
                print(f"  고유값 개수: {len(unique_values)}")
                print(f"  상위 5개: {list(unique_values.head(5).index)}")
        
        # 결과를 CSV 파일들로 저장
        print(f"\n=== CSV 파일 생성 중 ===")
        
        # 출력 폴더 생성
        output_dir = 'fashion_column_observation'
        os.makedirs(output_dir, exist_ok=True)
        print(f"[폴더] '{output_dir}' 폴더 생성 완료")
        
        # 1. 전체 요약 파일
        summary_data = []
        for col, info in column_analysis.items():
            summary_data.append({
                'column_name': col,
                'unique_count': info['unique_count'],
                'total_rows': info['total_count'],
                'top_3_values': ', '.join([str(v) if v is not None else '' for v in list(info['top_values'].keys())[:3]])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/fashion_columns_summary.csv', index=False)
        print("[성공] fashion_columns_summary.csv 생성 완료")
        
        # 2. 각 칼럼별 상세 파일들
        for col, info in column_analysis.items():
            detail_data = []
            for value, count in info['top_values'].items():
                percentage = (count / info['total_count']) * 100
                detail_data.append({
                    'value': value,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
            
            detail_df = pd.DataFrame(detail_data)
            filename = f"fashion_{col}_details.csv"
            detail_df.to_csv(f'{output_dir}/{filename}', index=False)
            print(f"[성공] {filename} 생성 완료 ({len(detail_data)}개 고유값)")
        
        # 3. 통합 분석 파일 (모든 칼럼의 상위 값들)
        print(f"\n=== 통합 분석 파일 생성 ===")
        all_analysis = []
        
        for col, info in column_analysis.items():
            for i, (value, count) in enumerate(list(info['top_values'].items())[:10]):
                percentage = (count / info['total_count']) * 100
                # value를 문자열로 변환 (float 등 숫자 타입 처리)
                str_value = str(value) if value is not None else ''
                all_analysis.append({
                    'column_name': col,
                    'rank': i + 1,
                    'value': str_value,
                    'count': count,
                    'percentage': round(percentage, 2)
                })
        
        all_df = pd.DataFrame(all_analysis)
        all_df.to_csv(f'{output_dir}/fashion_all_analysis.csv', index=False)
        print("[성공] fashion_all_analysis.csv 생성 완료")
        
        # 결과 요약 출력
        print(f"\n" + "="*60)
        print("[분석] Fashion 분석 결과 요약:")
        for col, info in column_analysis.items():
            print(f"\n{col}:")
            print(f"  - 고유값: {info['unique_count']}개")
            print(f"  - 상위 3개: {list(info['top_values'].keys())[:3]}")
        
        print(f"\n생성된 파일들 ({output_dir}/ 폴더):")
        print(f"  - fashion_columns_summary.csv (전체 요약)")
        print(f"  - fashion_all_analysis.csv (통합 분석)")
        for col in column_analysis.keys():
            print(f"  - fashion_{col}_details.csv (상세 분석)")
        
        return True
        
    except Exception as e:
        print(f"[오류] 오류 발생: {e}")
        return False

def main():
    print("="*80)
    print("데이터셋 칼럼 분석 도구")
    print("="*80)
    
    # H&M 데이터셋 분석
    print("\n1. H&M 데이터셋 분석")
    hnm_success = analyze_hnm_columns()
    
    # Fashion 데이터셋 분석
    print("\n2. Fashion 데이터셋 분석")
    fashion_success = analyze_fashion_columns()
    
    # 결과 요약
    print("\n" + "="*80)
    print("분석 완료 요약:")
    print(f"H&M 분석: {'[성공] 성공' if hnm_success else '[오류] 실패'}")
    print(f"Fashion 분석: {'[성공] 성공' if fashion_success else '[오류] 실패'}")
    print("="*80)

if __name__ == "__main__":
    main()

