#!/usr/bin/env python3
"""
farfetch.json 파일의 구조를 분석하고 CSV 변환 가능성을 확인하는 스크립트
"""

import json
import pandas as pd
from collections import Counter

def analyze_json_structure():
    print("=== farfetch.json 파일 구조 분석 ===")
    
    try:
        # JSON 파일 로드
        print("JSON 파일 로딩 중...")
        with open('farfetch.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON 파일 로드 완료!")
        print(f"데이터 타입: {type(data)}")
        
        if isinstance(data, list):
            print(f"리스트 형태, 총 {len(data):,}개 항목")
            
            # 첫 몇 개 항목 구조 확인
            print("\n=== 첫 3개 항목 구조 확인 ===")
            for i, item in enumerate(data[:3]):
                print(f"\n항목 {i+1}:")
                print(f"  타입: {type(item)}")
                if isinstance(item, dict):
                    print(f"  키들: {list(item.keys())}")
                    for key, value in item.items():
                        print(f"    {key}: {type(value)} - {str(value)[:100]}")
                else:
                    print(f"  값: {str(item)[:100]}")
            
            # 모든 키 수집 (딕셔너리인 경우)
            if data and isinstance(data[0], dict):
                all_keys = set()
                for item in data:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                
                print(f"\n=== 전체 고유 키 목록 ({len(all_keys)}개) ===")
                for key in sorted(all_keys):
                    print(f"  - {key}")
                
                # 키별 데이터 타입 분석
                print(f"\n=== 키별 데이터 타입 분석 (샘플 100개) ===")
                sample_data = data[:100]
                key_types = {}
                
                for key in sorted(all_keys):
                    types = []
                    for item in sample_data:
                        if isinstance(item, dict) and key in item:
                            types.append(type(item[key]).__name__)
                    
                    type_counter = Counter(types)
                    key_types[key] = type_counter
                    print(f"  {key}: {dict(type_counter)}")
        
        elif isinstance(data, dict):
            print("딕셔너리 형태")
            print(f"최상위 키들: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n키 '{key}':")
                print(f"  타입: {type(value)}")
                if isinstance(value, list):
                    print(f"  리스트 길이: {len(value)}")
                    if value:
                        print(f"  첫 번째 항목 타입: {type(value[0])}")
                        print(f"  첫 번째 항목: {str(value[0])[:100]}")
        
        else:
            print(f"예상치 못한 데이터 타입: {type(data)}")
            print(f"내용: {str(data)[:500]}")
        
        return data
    
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def check_conversion_possibility(data):
    """CSV 변환 가능성 확인"""
    print("\n" + "="*60)
    print("=== CSV 변환 가능성 분석 ===")
    
    if not data:
        print("데이터가 없어 변환 불가능")
        return False
    
    if isinstance(data, list) and data:
        first_item = data[0]
        
        if isinstance(first_item, dict):
            print("✅ 리스트 내 딕셔너리 구조 - CSV 변환 가능!")
            
            # 모든 항목이 비슷한 구조인지 확인
            all_keys = set()
            for item in data[:100]:  # 샘플 100개만 확인
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            print(f"예상 CSV 칼럼 수: {len(all_keys)}")
            print(f"예상 CSV 행 수: {len(data)}")
            
            return True
        else:
            print("❌ 딕셔너리가 아닌 구조 - 직접 변환 어려움")
            return False
    
    elif isinstance(data, dict):
        # 딕셔너리 내에 리스트가 있는지 확인
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                print(f"✅ '{key}' 키 내 리스트 구조 발견 - CSV 변환 가능!")
                print(f"예상 CSV 행 수: {len(value)}")
                return True
        
        print("❌ 적절한 변환 구조 찾을 수 없음")
        return False
    
    else:
        print("❌ CSV 변환에 적합하지 않은 구조")
        return False

def compare_with_existing_files():
    """기존 파일들과 비교하여 연결 가능한 키 찾기"""
    print("\n" + "="*60)
    print("=== 기존 파일들과 연결점 분석 ===")
    
    # articles_with_price.csv 확인
    try:
        articles_df = pd.read_csv('../articles_with_price.csv', nrows=5)
        print(f"\narticles_with_price.csv 칼럼들:")
        print(f"  {list(articles_df.columns)}")
    except Exception as e:
        print(f"articles_with_price.csv 읽기 오류: {e}")
    
    # styles.csv 확인
    try:
        styles_df = pd.read_csv('../styles.csv', nrows=5)
        print(f"\nstyles.csv 칼럼들:")
        print(f"  {list(styles_df.columns)}")
    except Exception as e:
        print(f"styles.csv 읽기 오류: {e}")

def main():
    print("farfetch.json 파일 분석 시작")
    print("="*60)
    
    # 1. JSON 구조 분석
    data = analyze_json_structure()
    
    # 2. 변환 가능성 확인
    can_convert = check_conversion_possibility(data)
    
    # 3. 기존 파일들과 비교
    compare_with_existing_files()
    
    if can_convert:
        print("\n" + "="*60)
        print("🎉 결론: CSV 변환 가능!")
        print("다음 단계로 변환 스크립트를 작성할 수 있습니다.")
    else:
        print("\n" + "="*60)
        print("⚠️ 결론: CSV 변환이 어려울 수 있습니다.")
        print("추가 분석이 필요합니다.")

if __name__ == "__main__":
    main()
