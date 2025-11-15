#!/usr/bin/env python3
"""
dataset/fashion/styles 폴더 내 개별 JSON 파일들을 하나의 CSV로 병합
출력: dataset/fashion/fashion.csv
"""

import os
import json
import pandas as pd
import numpy as np


def iter_json_files(directory: str):
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(".json"):
            yield entry.path


def load_json_safe(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON 로드 실패: {path} ({e})")
        return None


def extract_core_data(json_data):
    """JSON에서 핵심 정보만 추출 (이미지 URL 제외)"""
    if not json_data or 'data' not in json_data:
        return None
    
    data = json_data['data']
    
    # 핵심 정보만 추출
    core_data = {
        'id': data.get('id'),
        'discountedPrice': data.get('discountedPrice'),
        'brandName': data.get('brandName'),
        'ageGroup': data.get('ageGroup'),
        'gender': data.get('gender'),
        'baseColour': data.get('baseColour'),
        'colour1': data.get('colour1'),
        'colour2': data.get('colour2'),
        'fashionType': data.get('fashionType'),
        'season': data.get('season'),
        'year': data.get('year'),
        'usage': data.get('usage'),
        'productDisplayName': data.get('productDisplayName'),
        'variantName': data.get('variantName'),
        'styleType': data.get('styleType'),
        'productTypeId': data.get('productTypeId'),
        'articleNumber': data.get('articleNumber'),
        'displayCategories': data.get('displayCategories'),
    }
    
    # 중첩된 객체에서 분류 정보 추출
    if 'masterCategory' in data:
        core_data['masterCategory'] = data['masterCategory'].get('typeName')
    if 'subCategory' in data:
        core_data['subCategory'] = data['subCategory'].get('typeName')
    if 'articleType' in data:
        core_data['articleType'] = data['articleType'].get('typeName')
    
    # 상품 설명 (HTML 태그 제거)
    if 'productDescriptors' in data and 'description' in data['productDescriptors']:
        desc = data['productDescriptors']['description'].get('value', '')
        # 간단한 HTML 태그 제거
        import re
        desc = re.sub(r'<[^>]+>', '', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        desc = desc[:500]  # 500자로 제한
    else:
        desc = ''
    
    # 가격에 영향을 줄 수 있는 추가 정보를 description에 추가
    additional_info = []
    
    # 브랜드 사용자 프로필 (가격 프리미엄 정보)
    if 'brandUserProfile' in data and data['brandUserProfile']:
        brand_profile = data['brandUserProfile']
        if isinstance(brand_profile, dict):
            # brandUserProfile 내부의 주요 정보 추출
            if 'name' in brand_profile:
                additional_info.append(f"brandUserProfile: {brand_profile['name']}")
        elif isinstance(brand_profile, str):
            additional_info.append(f"brandUserProfile: {brand_profile}")
    
    # 평점 (신뢰도/품질 지표, 가격에 영향)
    if 'myntraRating' in data and data['myntraRating']:
        rating = data['myntraRating']
        if isinstance(rating, dict):
            if 'averageRating' in rating:
                additional_info.append(f"myntraRating: {rating['averageRating']}")
        elif pd.notna(rating):
            additional_info.append(f"myntraRating: {rating}")
    
    # 원본 가격 (discountedPrice와 비교 가능)
    if 'price' in data and data['price']:
        price = data['price']
        if isinstance(price, (int, float)) and price > 0:
            additional_info.append(f"price: {price}")
    
    # 부가세 정보 (최종 가격에 영향)
    if 'vat' in data and data['vat']:
        vat = data['vat']
        if isinstance(vat, (int, float)) and vat > 0:
            additional_info.append(f"vat: {vat}")
    
    # description에 추가 정보 병합
    if additional_info:
        additional_text = "\n".join(additional_info)
        if desc:
            core_data['description'] = f"{desc}\n\n{additional_text}"
        else:
            core_data['description'] = additional_text
    else:
        core_data['description'] = desc
    
    return core_data

def build_fashion_csv(styles_dir: str = "dataset/fashion/styles",
                      output_csv: str = "dataset/fashion/fashion_1_raw.csv",
                      write_chunk_size: int = 1000) -> bool:
    print("=== Build fashion.csv from styles/*.json (핵심 정보만 추출) ===")
    print(f"입력 폴더: {styles_dir}")
    print(f"출력 파일: {output_csv}")

    if not os.path.isdir(styles_dir):
        print(f"[오류] 입력 폴더가 존재하지 않습니다: {styles_dir}")
        return False

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    buffer = []
    total = 0
    wrote_header = False

    # 기존 결과가 있다면 덮어쓰기
    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except OSError:
            pass

    for idx, json_path in enumerate(iter_json_files(styles_dir), start=1):
        json_data = load_json_safe(json_path)
        if json_data is None:
            continue
            
        # 핵심 정보만 추출
        core_data = extract_core_data(json_data)
        if core_data is None:
            continue
            
        buffer.append(core_data)

        if len(buffer) >= write_chunk_size:
            df = pd.DataFrame(buffer)
            df.to_csv(output_csv, mode="a", index=False, header=(not wrote_header))
            wrote_header = True
            total += len(df)
            buffer.clear()
            if idx % (write_chunk_size * 2) == 0:
                print(f"  진행: {total:,} 행 누적 저장")

    if buffer:
        df = pd.DataFrame(buffer)
        df.to_csv(output_csv, mode="a", index=False, header=(not wrote_header))
        total += len(df)

    print(f"[완료] 저장된 총 행 수: {total:,}")
    print(f"출력 경로: {output_csv}")
    print(f"칼럼 수: {len(buffer[0]) if buffer else 'N/A'}")
    return True


if __name__ == "__main__":
    build_fashion_csv()


