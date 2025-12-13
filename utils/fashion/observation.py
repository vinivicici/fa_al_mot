#!/usr/bin/env python3
"""
fashion.csv 내 주요 카테고리 컬럼의 고유값/분포 관찰 스크립트

입력: dataset/fashion/fashion.csv (기본)
출력: fashion_column_observation/ 하위에 컬럼별 분포 CSV 저장
    - fashion_masterCategory_details.csv
    - fashion_subCategory_details.csv
    - fashion_articleType_details.csv
    - fashion_usage_details.csv
    - fashion_columns_summary.csv (선택 요약)
"""

import os
import sys
import argparse
import pandas as pd


TARGET_COLUMNS = [
    "masterCategory",
    "subCategory",
    "articleType",
    "usage",
]


def print_unique_values(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        print(f"\n[{column}] 열이 없습니다.")
        return
    values = sorted(set(map(str, df[column].dropna().unique())))
    print(f"\n[{column}] ({len(values)}개)")
    for v in values:
        print(v)


def save_value_counts(df: pd.DataFrame, column: str, out_dir: str) -> None:
    if column not in df.columns:
        return
    vc = (
        df[column]
        .dropna()
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )

    # 정렬: count desc, value asc
    vc = vc.sort_values(by=["count", column], ascending=[False, True]).reset_index(drop=True)

    filename_map = {
        "masterCategory": "fashion_masterCategory_details.csv",
        "subCategory": "fashion_subCategory_details.csv",
        "articleType": "fashion_articleType_details.csv",
        "usage": "fashion_usage_details.csv",
    }
    out_name = filename_map.get(column, f"fashion_{column}_details.csv")
    out_path = os.path.join(out_dir, out_name)
    vc.to_csv(out_path, index=False)
    print(f"저장: {out_path} ({len(vc)}행)")


def save_columns_summary(df: pd.DataFrame, columns: list[str], out_dir: str) -> None:
    summary_rows = []
    for col in columns:
        exists = col in df.columns
        unique_cnt = df[col].nunique(dropna=True) if exists else 0
        summary_rows.append({
            "column": col,
            "exists": exists,
            "unique_count": int(unique_cnt),
        })
    out_path = os.path.join(out_dir, "fashion_columns_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"저장: {out_path}")


def run(input_csv: str, out_dir: str) -> int:
    if not os.path.exists(input_csv):
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {input_csv}")
        return 1

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[오류] CSV 로드 실패: {e}")
        return 1

    os.makedirs(out_dir, exist_ok=True)

    print("=== Fashion 카테고리 관찰 ===")
    print(f"입력: {input_csv}")
    print(f"출력 폴더: {out_dir}")

    # 콘솔 출력
    for col in TARGET_COLUMNS:
        print_unique_values(df, col)

    # 파일 저장
    for col in TARGET_COLUMNS:
        save_value_counts(df, col, out_dir)

    save_columns_summary(df, TARGET_COLUMNS, out_dir)
    print("[완료] 관찰 결과 저장 완료")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion 카테고리 고유값/분포 관찰")
    parser.add_argument(
        "--input",
        default="dataset/fashion/fashion.csv",
        help="입력 CSV 경로 (기본: dataset/fashion/fashion.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="fashion_column_observation",
        help="출력 폴더 경로 (기본: fashion_column_observation)",
    )
    # 추가: subCategory details에서 '?' 표시된 항목을 심층 분석
    parser.add_argument(
        "--analyze-unknown-subcats",
        metavar="SUBCAT_DETAILS_CSV",
        help="subCategory details CSV 경로(세 번째 컬럼이 '?'인 서브카테고리들을 분석)",
    )
    args = parser.parse_args()

    # 기본 관찰 수행
    exit_code = run(args.input, args.outdir)

    # 선택: '?' 표시된 서브카테고리 심층 분석
    if args.analyze_unknown_subcats:
        analyze_unknown_subcategories(
            input_csv=args.input,
            subcat_details_csv=args.analyze_unknown_subcats,
            out_dir=args.outdir,
        )

    sys.exit(exit_code)


def analyze_unknown_subcategories(input_csv: str, subcat_details_csv: str, out_dir: str) -> None:
    """subCategory details의 세 번째 컬럼이 '?'인 항목들을 대상으로
    fashion.csv 내 관련 상품의 articleType 분포 및 예시 productDisplayName을 저장한다.
    """
    print("\n=== '?' 표기된 subCategory 심층 분석 ===")
    if not os.path.exists(subcat_details_csv):
        print(f"[경고] subCategory details CSV가 없습니다: {subcat_details_csv}")
        return
    try:
        details = pd.read_csv(subcat_details_csv)
    except Exception as e:
        print(f"[경고] subCategory details CSV 로드 실패: {e}")
        return

    # 세 번째 컬럼명 파악 (헤더가 3개 미만일 수도 있어 보호적 처리)
    cols = list(details.columns)
    flag_col = cols[2] if len(cols) >= 3 else None
    if not flag_col:
        print("[경고] 세 번째 컬럼을 찾을 수 없습니다.")
        return

    unknown_subcats = [
        str(v)
        for v, f in zip(details.get("subCategory", []), details.get(flag_col, []))
        if str(f).strip() == "?"
    ]
    unknown_subcats = [s for s in unknown_subcats if s and s != "nan"]
    if not unknown_subcats:
        print("[정보] '?' 표기된 subCategory가 없습니다.")
        return

    print("대상 subCategory:", ", ".join(unknown_subcats))

    try:
        usecols = [
            c for c in ["subCategory", "articleType", "productDisplayName", "masterCategory"]
            if c  # 존재 여부는 이후 df.columns로 필터링
        ]
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[경고] fashion.csv 로드 실패: {e}")
        return

    # 필요한 컬럼만 보존
    existing = [c for c in ["subCategory", "articleType", "productDisplayName", "masterCategory"] if c in df.columns]
    df = df[existing].copy()

    os.makedirs(out_dir, exist_ok=True)

    # 통합 요약: subCategory, articleType 분포 저장
    summary_rows = []
    for subcat in unknown_subcats:
        sub_df = df[df.get("subCategory").astype(str) == subcat]
        if sub_df.empty:
            continue
        vc = (
            sub_df.get("articleType", pd.Series(dtype=str))
            .astype(str)
            .value_counts()
            .rename_axis("articleType")
            .reset_index(name="count")
        )
        vc.insert(0, "subCategory", subcat)
        summary_rows.append(vc)

        # 예시 상품명 상위 20개 저장
        if "productDisplayName" in sub_df.columns:
            ex = (
                sub_df["productDisplayName"].astype(str).value_counts().head(20)
                .rename_axis("productDisplayName").reset_index(name="count")
            )
            ex_path = os.path.join(out_dir, f"unknown_{subcat}_examples.csv")
            ex.to_csv(ex_path, index=False)
            print(f"저장: {ex_path}")

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        out_path = os.path.join(out_dir, "unknown_subcategory_articleType_summary.csv")
        summary.to_csv(out_path, index=False)
        print(f"저장: {out_path}")
    else:
        print("[정보] 대상 subCategory에 해당하는 행이 없습니다.")


if __name__ == "__main__":
    main()
