#!/usr/bin/env python3
"""
farfetch.json을 간단하게 CSV로 변환하는 스크립트
"""

import json
import pandas as pd

def convert_farfetch_json():
    print("=== farfetch.json 변환 시작 ===")
    
    try:
        # JSON 파일 전체 로드
        print("JSON 파일 로딩 중...")
        with open('farfetch.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON 로드 성공!")
        print(f"데이터 타입: {type(data)}")
        print(f"항목 수: {len(data):,}개")
        
        # 첫 번째 항목 구조 확인
        if data:
            first_item = data[0]
            print(f"\n첫 번째 항목의 키들:")
            for key in first_item.keys():
                value = first_item[key]
                print(f"  {key}: {type(value).__name__} - {str(value)[:60]}")
        
        # DataFrame으로 변환
        print(f"\nDataFrame 변환 중...")
        df = pd.DataFrame(data)
        
        print(f"✅ DataFrame 변환 성공!")
        print(f"크기: {df.shape[0]:,} 행 x {df.shape[1]} 칼럼")
        
        # 칼럼 정보 출력
        print(f"\n모든 칼럼들:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # 기존 파일들과 비교
        print(f"\n=== 기존 파일들과 연결점 분석 ===")
        
        # articles_with_price.csv
        try:
            articles_sample = pd.read_csv('articles_with_price.csv', nrows=3)
            print(f"articles_with_price.csv 주요 칼럼: article_id, prod_name, price 등")
            print(f"  샘플 article_id: {articles_sample['article_id'].tolist()}")
        except Exception as e:
            print(f"articles_with_price.csv 읽기 실패: {e}")
        
        # styles.csv  
        try:
            styles_sample = pd.read_csv('styles.csv', nrows=3)
            print(f"styles.csv 주요 칼럼: id, gender, masterCategory, articleType 등")
            print(f"  샘플 id: {styles_sample['id'].tolist()}")
        except Exception as e:
            print(f"styles.csv 읽기 실패: {e}")
        
        # farfetch 데이터에서 연결 가능한 키 찾기
        farfetch_keys = list(df.columns)
        potential_id_keys = [key for key in farfetch_keys if 'id' in key.lower()]
        potential_name_keys = [key for key in farfetch_keys if any(word in key.lower() for word in ['name', 'title', 'product'])]
        potential_category_keys = [key for key in farfetch_keys if any(word in key.lower() for word in ['category', 'type', 'gender'])]
        
        print(f"\nfarfetch.json에서 발견된 연결 가능한 키들:")
        print(f"  ID 관련: {potential_id_keys}")
        print(f"  이름 관련: {potential_name_keys}")  
        print(f"  카테고리 관련: {potential_category_keys}")
        
        # CSV로 저장
        output_file = 'farfetch.csv'
        print(f"\n{output_file}로 저장 중...")
        df.to_csv(output_file, index=False)
        
        print(f"✅ 변환 완료!")
        print(f"출력 파일: {output_file}")
        
        # 샘플 데이터 출력 (주요 칼럼들만)
        sample_cols = []
        for col in ['item_id', 'sku', 'title', 'brand', 'price', 'gender', 'breadcrumbs']:
            if col in df.columns:
                sample_cols.append(col)
        
        if sample_cols:
            print(f"\n주요 칼럼 샘플 데이터:")
            print(df[sample_cols].head(3).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    convert_farfetch_json()
