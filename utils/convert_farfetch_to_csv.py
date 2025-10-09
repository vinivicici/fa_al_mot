#!/usr/bin/env python3
"""
farfetch.json을 CSV로 변환하는 스크립트
"""

import json
import pandas as pd
import sys
from tqdm import tqdm

def analyze_json_quickly():
    """JSON 파일을 빠르게 분석"""
    print("=== farfetch.json 빠른 분석 ===")
    
    try:
        # 파일 크기 확인
        import os
        file_size = os.path.getsize('dataset/hnm/farfetch.json')
        print(f"파일 크기: {file_size / (1024*1024):.1f} MB")
        
        # 첫 몇 개 항목만 로드하여 구조 파악
        print("첫 3개 항목 구조 분석 중...")
        
        with open('dataset/hnm/farfetch.json', 'r', encoding='utf-8') as f:
            # 첫 부분만 읽어서 구조 파악
            content = f.read(10000)  # 10KB만 읽기
            
            # JSON 시작 부분에서 첫 번째 완전한 객체 찾기
            bracket_count = 0
            first_object = ""
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    first_object += char
                    continue
                    
                if char == '\\':
                    escape_next = True
                    first_object += char
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                
                first_object += char
                
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0 and first_object.strip().startswith('[{'):
                            # 첫 번째 완전한 객체 발견
                            break
            
            # 첫 번째 객체만 파싱
            try:
                # [{ 로 시작하므로 첫 번째 객체만 추출
                start_idx = first_object.find('{')
                end_idx = first_object.find('}') + 1
                first_item_str = first_object[start_idx:end_idx]
                
                first_item = json.loads(first_item_str)
                print("첫 번째 항목 구조:")
                for key, value in first_item.items():
                    print(f"  {key}: {type(value).__name__} - {str(value)[:80]}")
                
                return list(first_item.keys())
                
            except json.JSONDecodeError as e:
                print(f"부분 파싱 실패: {e}")
                return None
                
    except Exception as e:
        print(f"분석 오류: {e}")
        return None

def convert_to_csv_streaming():
    """스트리밍 방식으로 JSON을 CSV로 변환"""
    print("\n=== JSON to CSV 변환 시작 ===")
    
    try:
        # 전체 JSON 로드 (메모리 사용량 주의)
        print("JSON 파일 로딩 중...")
        with open('dataset/hnm/farfetch.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"총 {len(data):,}개 항목 로드 완료")
        
        if not data:
            print("데이터가 비어있습니다.")
            return False
        
        # DataFrame으로 변환
        print("DataFrame 변환 중...")
        df = pd.DataFrame(data)
        
        print(f"변환된 DataFrame 크기: {df.shape}")
        print(f"칼럼들: {list(df.columns)}")
        
        # 기존 파일들과 비교
        print("\n=== 기존 파일들과 연결점 확인 ===")
        
        # articles_with_price.csv 칼럼 확인
        try:
            articles_df = pd.read_csv('dataset/hnm/articles_with_price.csv', nrows=3)
            print(f"articles_with_price.csv 칼럼: {list(articles_df.columns)}")
        except:
            print("articles_with_price.csv 읽기 실패")
        
        # styles.csv 칼럼 확인  
        try:
            styles_df = pd.read_csv('dataset/hnm/styles.csv', nrows=3)
            print(f"styles.csv 칼럼: {list(styles_df.columns)}")
        except:
            print("styles.csv 읽기 실패")
        
        # 연결 가능한 키 찾기
        farfetch_cols = set(df.columns)
        print(f"\nfarfetch.json 칼럼들: {sorted(farfetch_cols)}")
        
        # 공통 키나 유사한 키 찾기
        potential_links = []
        for col in farfetch_cols:
            if any(keyword in col.lower() for keyword in ['id', 'sku', 'item', 'product', 'article']):
                potential_links.append(col)
        
        if potential_links:
            print(f"연결 가능한 키들: {potential_links}")
        else:
            print("명확한 연결 키를 찾을 수 없음")
        
        # CSV로 저장
        output_filename = 'dataset/hnm/farfetch.csv'
        print(f"\n{output_filename}으로 저장 중...")
        df.to_csv(output_filename, index=False)
        
        print("✅ 변환 완료!")
        print(f"출력 파일: {output_filename}")
        print(f"행 수: {len(df):,}")
        print(f"칼럼 수: {len(df.columns)}")
        
        # 샘플 데이터 출력
        print("\n샘플 데이터 (첫 3행):")
        print(df.head(3).to_string())
        
        return True
        
    except MemoryError:
        print("❌ 메모리 부족으로 변환 실패")
        print("파일이 너무 큽니다. 청크 단위 처리가 필요합니다.")
        return False
    except Exception as e:
        print(f"❌ 변환 오류: {e}")
        return False

def main():
    print("farfetch.json → CSV 변환 작업")
    print("=" * 50)
    
    # 1. 빠른 구조 분석
    keys = analyze_json_quickly()
    
    if keys:
        print(f"\n감지된 키들: {keys}")
        
        # 2. CSV 변환 시도
        success = convert_to_csv_streaming()
        
        if success:
            print("\n🎉 변환 성공! 이제 다른 파일들과 합칠 수 있습니다.")
        else:
            print("\n⚠️ 변환 실패. 다른 방법을 시도해야 합니다.")
    else:
        print("\n❌ 구조 분석 실패")

if __name__ == "__main__":
    main()
