#!/usr/bin/env python3
"""
articles_with_price.csv에서 department_name = garment_group_name 검증
"""

import pandas as pd
from collections import defaultdict

def verify_name_consistency():
    print("=== department_name = garment_group_name 검증 ===")
    
    try:
        # CSV 파일 로드
        print("CSV 파일 로딩 중...")
        df = pd.read_csv('articles_with_price.csv')
        print(f"총 데이터: {len(df):,}개 행")
        
        # 필요한 칼럼 확인
        required_columns = ['department_name', 'garment_group_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 누락된 칼럼: {missing_columns}")
            return False
        
        print(f"검증 대상 칼럼: {required_columns}")
        
        # department_name이 garment_group_name과 일치하는지 검증
        df['is_match'] = (df['department_name'].str.replace(' ', '') == 
                         df['garment_group_name'].str.replace(' ', ''))
        
        # 결과 분석
        total_rows = len(df)
        matched_rows = df['is_match'].sum()
        unmatched_rows = total_rows - matched_rows
        
        print(f"\n=== 검증 결과 ===")
        print(f"총 행 수: {total_rows:,}")
        print(f"일치: {matched_rows:,}개 ({matched_rows/total_rows*100:.2f}%)")
        print(f"불일치: {unmatched_rows:,}개 ({unmatched_rows/total_rows*100:.2f}%)")
        
        if unmatched_rows > 0:
            print(f"\n=== 불일치 사례 (처음 20개) ===")
            unmatched_df = df[~df['is_match']][['department_name', 'garment_group_name']]
            
            for i, (_, row) in enumerate(unmatched_df.head(20).iterrows()):
                print(f"{i+1:2d}. department_name: '{row['department_name']}'")
                print(f"    garment_group_name: '{row['garment_group_name']}'")
                print()
            
            if unmatched_rows > 20:
                print(f"    ... 총 {unmatched_rows}개의 불일치 사례")
            
            # 불일치 패턴 분석
            print(f"\n=== 불일치 패턴 분석 ===")
            pattern_analysis = defaultdict(int)
            
            for _, row in unmatched_df.iterrows():
                dept = row['department_name']
                garment = row['garment_group_name']
                
                if dept in garment:
                    pattern_analysis['department_name이 garment_group_name에 포함됨'] += 1
                elif garment in dept:
                    pattern_analysis['garment_group_name이 department_name에 포함됨'] += 1
                else:
                    pattern_analysis['완전히 다른 패턴'] += 1
            
            for pattern, count in pattern_analysis.items():
                print(f"  {pattern}: {count:,}개")
        
        # 고유값 개수 비교
        print(f"\n=== 고유값 개수 비교 ===")
        print(f"department_name 고유값: {df['department_name'].nunique():,}개")
        print(f"garment_group_name 고유값: {df['garment_group_name'].nunique():,}개")
        
        return matched_rows == total_rows
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def analyze_name_relationships():
    """이름들 간의 관계를 더 자세히 분석"""
    print("\n=== 이름 관계 상세 분석 ===")
    
    try:
        df = pd.read_csv('articles_with_price.csv')
        
        # section_name별 department_name 분포
        print("\n1. section_name별 department_name 분포 (상위 10개):")
        section_dept = df.groupby('section_name')['department_name'].nunique().sort_values(ascending=False)
        for section, count in section_dept.head(10).items():
            print(f"  {section}: {count}개 department_name")
        
        # garment_group_name별 department_name 분포
        print("\n2. garment_group_name별 department_name 분포 (상위 10개):")
        garment_dept = df.groupby('garment_group_name')['department_name'].nunique().sort_values(ascending=False)
        for garment, count in garment_dept.head(10).items():
            print(f"  {garment}: {count}개 department_name")
        
        # section_name + garment_group_name 조합별 department_name 분포
        print("\n3. section_name + garment_group_name 조합별 department_name 분포:")
        combination_dept = df.groupby(['section_name', 'garment_group_name'])['department_name'].nunique()
        multi_dept_combinations = combination_dept[combination_dept > 1]
        
        if len(multi_dept_combinations) > 0:
            print(f"  {len(multi_dept_combinations)}개 조합이 여러 department_name을 가짐:")
            for (section, garment), count in multi_dept_combinations.head(10).items():
                print(f"    {section} + {garment}: {count}개 department_name")
        else:
            print("  모든 조합이 고유한 department_name을 가짐")
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")

if __name__ == "__main__":
    is_consistent = verify_name_consistency()
    
    if is_consistent:
        print("\n🎉 검증 완료: department_name = garment_group_name이 항상 성립합니다!")
    else:
        print("\n⚠️ 검증 완료: 일부 경우에서 불일치가 발견되었습니다.")
        analyze_name_relationships()
