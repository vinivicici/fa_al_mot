#!/usr/bin/env python3
"""
데이터셋 병합 메인 파이프라인
H&M과 Fashion 데이터셋을 통합하는 전체 과정을 자동화
"""

import sys
import os
import subprocess
from datetime import datetime

def run_script(script_path, script_name):
    """스크립트 실행 함수"""
    print("\n" + "="*80)
    print(f"실행 중: {script_name}")
    print("="*80)

    try:
        # Python 스크립트 실행 (Windows 환경 고려)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='cp949',  # Windows 콘솔 인코딩 사용
            errors='replace'   # 디코딩 에러 무시
        )

        # 출력 표시
        if result.stdout:
            print(result.stdout)

        # 에러 확인
        if result.returncode != 0:
            print(f"[오류] {script_name} 실행 실패!")
            if result.stderr:
                print("에러 메시지:")
                print(result.stderr)
            return False

        print(f"[완료] {script_name} 완료!")
        return True

    except Exception as e:
        print(f"[오류] {script_name} 실행 중 오류 발생: {e}")
        return False

def check_prerequisites():
    """전제 조건 확인"""
    print("=== 전제 조건 확인 ===")
    
    # 필수 파일 확인
    required_files = [
        'dataset/hnm/articles_with_price.csv',
        'dataset/fashion/fashion.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024**2)  # MB
            print(f"  [O] {file} ({size:.1f} MB)")
        else:
            print(f"  [X] {file} - 파일이 없습니다!")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n[오류] 필수 파일이 없습니다: {missing_files}")
        print("다음 단계를 먼저 실행하세요:")
        print("1. H&M 데이터 전처리: python hnm_preprocess.py")
        print("2. Fashion 데이터 전처리: python utils/fashion/fashion_build_csv.py")
        print("3. Fashion 칼럼 드롭: python utils/fashion/column_drop.py")
        return False
    
    print("[완료] 모든 전제 조건 충족!")
    return True

def main():
    """메인 함수"""
    start_time = datetime.now()

    print("="*80)
    print("데이터셋 병합 메인 파이프라인 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 전제 조건 확인
    if not check_prerequisites():
        print("\n[오류] 전제 조건을 충족하지 않습니다. 작업을 중단합니다.")
        return

    # 스크립트 목록 (실행 순서대로)
    scripts = [
        ("utils/dataset_join/price_unification.py", "1. 가격 통일 - H&M 유로→달러 변환"),
        ("utils/dataset_join/category_mapping.py", "2. 카테고리/성별 매핑 - 통일된 분류 적용"),
        ("utils/dataset_join/dataset_merge.py", "3. 데이터셋 병합 - 최종 통합 데이터셋 생성"),
    ]

    # 각 스크립트 실행
    print("\n=== 데이터셋 병합 파이프라인 실행 ===")

    for script_path, script_name in scripts:
        if not os.path.exists(script_path):
            print(f"\n[오류] 스크립트를 찾을 수 없습니다: {script_path}")
            print("작업을 중단합니다.")
            return

        success = run_script(script_path, script_name)

        if not success:
            print(f"\n[오류] 데이터셋 병합 파이프라인 실패: {script_name}에서 오류 발생")
            print("작업을 중단합니다.")
            return

    # 완료
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    print("\n" + "="*80)
    print("[완료] 데이터셋 병합 파이프라인 완료!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")

    # 최종 결과 확인
    print("\n=== 최종 결과 확인 ===")
    
    final_files = [
        'dataset/hnm/articles_with_price_unified.csv',
        'dataset/fashion/fashion_pricied.csv',
        'dataset/hnm/articles_with_price_mapped.csv',
        'dataset/fashion/fashion_mapped.csv',
        'dataset/merged_dataset.csv',
        'dataset/merge_summary.txt'
    ]
    
    for file in final_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024**2)  # MB
            print(f"  [O] {file} ({size:.1f} MB)")
        else:
            print(f"  [X] {file} - 파일이 없습니다!")

    print("\n=== 다음 단계 ===")
    print("1. dataset/merged_dataset.csv 파일을 확인하세요")
    print("2. dataset/merge_summary.txt 파일로 병합 결과를 검토하세요")
    print("3. 필요시 추가 전처리나 분석을 진행하세요")

if __name__ == "__main__":
    main()
