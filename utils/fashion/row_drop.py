#!/usr/bin/env python3
"""
Fashion CSV에서 제거 대상 subCategory 하드코딩 삭제 스크립트

기준(하드코딩): 사용자가 검토하여 삭제로 확정한 subCategory 목록

동작:
- 입력: dataset/fashion/fashion_1_raw.csv
- 백업: dataset/fashion/fashion_before_row_drop.csv
- 출력: dataset/fashion/fashion_2_rowdrop.csv (제거 후 새 파일)
- 부가 출력: dataset/fashion/fashion_rowdrop_removed.csv (제거된 행 요약)
"""

import os
import sys
import argparse
import pandas as pd


# 제거 대상 subCategory (사용자 승인됨)
DROP_SUBCATEGORIES: list[str] = [
    "Watches",
    "Jewellery",
    "Eyewear",
    "Fragrance",
    "Socks",
    "Lips",
    "Saree",
    "Nails",
    "Makeup",
    "Accessories",
    "Apparel Set",
    "Free Gifts",
    "Skin Care",
    "Skin",
    "Eyes",
    "Shoe Accessories",
    "Sports Equipment",
    "Hair",
    "Bath and Body",
    "Water Bottle",
    "Perfumes",
    "Umbrellas",
    "Beauty Accessories",
    "Wristbands",
    "Sports Accessories",
    "Home Furnishing",
    "Vouchers",
]


def row_drop(input_csv: str) -> int:
    if not os.path.exists(input_csv):
        print(f"[오류] 입력 CSV가 없습니다: {input_csv}")
        return 1

    drop_list = sorted(set(DROP_SUBCATEGORIES))
    if not drop_list:
        print("[정보] 제거 대상 subCategory가 없습니다. 작업을 건너뜁니다.")
        return 0

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[오류] 입력 CSV 로드 실패: {e}")
        return 1

    if 'subCategory' not in df.columns:
        print("[오류] 입력 CSV에 'subCategory' 컬럼이 없습니다.")
        return 1

    total_before = len(df)
    mask_drop = df['subCategory'].astype(str).isin(set(drop_list))
    df_removed = df[mask_drop].copy()
    df_kept = df[~mask_drop].copy()

    removed_count = len(df_removed)
    kept_count = len(df_kept)

    # 백업 및 저장
    backup_path = os.path.join(os.path.dirname(input_csv), 'fashion_before_row_drop.csv')
    try:
        df.to_csv(backup_path, index=False)
        print(f"백업 저장: {backup_path} ({total_before}행)")
    except Exception as e:
        print(f"[경고] 백업 저장 실패: {e}")

    # 제거된 행 정보 저장(요약)
    out_dir = os.path.dirname(input_csv)
    removed_path = os.path.join(out_dir, 'fashion_rowdrop_removed.csv')
    try:
        # 요약: subCategory, articleType, count
        cols_available = [c for c in ['subCategory', 'articleType'] if c in df_removed.columns]
        if cols_available:
            summary = (
                df_removed[cols_available]
                .astype(str)
                .value_counts()
                .reset_index(name='count')
            )
            summary.to_csv(removed_path, index=False)
            print(f"제거 행 요약 저장: {removed_path} ({len(summary)}행)")
        else:
            df_removed.head(200).to_csv(removed_path, index=False)
            print(f"제거 행 샘플 저장: {removed_path} (최대 200행)")
    except Exception as e:
        print(f"[경고] 제거 행 정보 저장 실패: {e}")

    # 새 파일로 저장 (fashion_2_rowdrop.csv)
    try:
        folder = os.path.dirname(input_csv)
        output_path = os.path.join(folder, 'fashion_2_rowdrop.csv')
        df_kept.to_csv(output_path, index=False)
        print(f"저장 완료: {output_path} (이전 {total_before}행 → 현재 {kept_count}행, {removed_count}행 제거)")
    except Exception as e:
        print(f"[오류] 결과 저장 실패: {e}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Fashion subCategory 하드코딩 행 제거')
    parser.add_argument(
        '--input',
        default='dataset/fashion/fashion_1_raw.csv',
        help='입력 CSV 경로 (기본: dataset/fashion/fashion_1_raw.csv)'
    )
    args = parser.parse_args()

    sys.exit(row_drop(args.input))


if __name__ == '__main__':
    main()


