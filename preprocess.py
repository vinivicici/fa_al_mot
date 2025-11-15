#!/usr/bin/env python3
"""
전체 전처리 파이프라인 메인 스크립트
H&M과 Fashion 데이터셋을 전처리하여 통합 데이터셋을 생성합니다.

실행 순서:
1. H&M 데이터 전처리
   - JOIN: articles + transactions 가격 데이터
   - COLUMN DROP: 불필요한 칼럼 제거
   - ROW DROP: 불필요한 행 제거
   - COLUMN SPLIT: product_group_name 제거
   
2. Fashion 데이터 전처리
   - BUILD CSV: styles/*.json → fashion_1_raw.csv
   - ROW DROP: 불필요한 서브카테고리 제거 → fashion_2_rowdrop.csv
   - COLUMN DROP: 불필요한 칼럼 제거 → fashion_3_columndrop.csv
   
3. 데이터셋 병합
   - PRICE UNIFICATION: 가격 통일 (H&M 유로→달러 변환)
   - CATEGORY MAPPING: 카테고리/성별 매핑
   - DATASET MERGE: 최종 통합 데이터셋 생성
   
4. 원핫 인코딩 (선택사항)
   - ONE HOT ENCODE: 카테고리 칼럼 원핫 인코딩
"""

import sys
import os
import subprocess
import argparse
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

def check_initial_files():
    """초기 파일 확인"""
    print("=== 초기 필수 파일 확인 ===")
    
    required_files = [
        'dataset/hnm/articles.csv',
        'dataset/hnm/transactions_train.csv',
        'dataset/fashion/styles'  # 폴더 확인
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            if os.path.isdir(file):
                json_count = len([f for f in os.listdir(file) if f.endswith('.json')])
                print(f"  [O] {file} ({json_count:,}개 JSON 파일)")
            else:
                size = os.path.getsize(file) / (1024**2)  # MB
                print(f"  [O] {file} ({size:.1f} MB)")
        else:
            print(f"  [X] {file} - 파일/폴더가 없습니다!")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n[오류] 필수 파일/폴더가 없습니다: {missing_files}")
        return False
    
    print("[완료] 모든 초기 파일 확인 완료!")
    return True

def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description='전체 전처리 파이프라인 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python preprocess.py              # 전체 파이프라인 실행
  python preprocess.py --skip       # Fashion JSON→CSV 변환 건너뛰기
        """
    )
    parser.add_argument(
        '--skip',
        action='store_true',
        help='Fashion 데이터셋의 JSON→CSV 변환 과정을 건너뜁니다 (fashion.csv가 이미 존재하는 경우)'
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("="*80)
    print("전체 전처리 파이프라인 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.skip:
        print("\n[옵션] --skip 플래그 감지: Fashion JSON→CSV 변환을 건너뜁니다.")
        print("       (fashion.csv가 이미 존재하는 경우 사용)")
    
    # 초기 파일 확인
    if not check_initial_files():
        print("\n[오류] 필수 파일이 없습니다. 작업을 중단합니다.")
        return
    
    # ===== 1단계: H&M 데이터 전처리 =====
    print("\n" + "="*80)
    print("1단계: H&M 데이터 전처리")
    print("="*80)
    
    hnm_scripts = [
        ("utils/hnm/hnm_join.py", "1-1. JOIN - articles + 가격 데이터 병합"),
        ("utils/hnm/hnm_column_drop.py", "1-2. COLUMN DROP - 불필요한 칼럼 제거"),
        ("utils/hnm/hnm_row_drop.py", "1-3. ROW DROP - 불필요한 행 제거"),
        ("utils/hnm/hnm_column_split_densify.py", "1-4. COLUMN SPLIT - product_group_name 제거"),
    ]
    
    for script_path, script_name in hnm_scripts:
        if not os.path.exists(script_path):
            print(f"\n[오류] 스크립트를 찾을 수 없습니다: {script_path}")
            print("작업을 중단합니다.")
            return
        
        success = run_script(script_path, script_name)
        
        if not success:
            print(f"\n[오류] H&M 전처리 실패: {script_name}에서 오류 발생")
            print("작업을 중단합니다.")
            return
    
    # H&M 전처리 결과 확인
    if not os.path.exists('dataset/hnm/articles_with_price.csv'):
        print("\n[오류] H&M 전처리 결과 파일이 생성되지 않았습니다.")
        print("작업을 중단합니다.")
        return
    
    # ===== 2단계: Fashion 데이터 전처리 =====
    print("\n" + "="*80)
    print("2단계: Fashion 데이터 전처리")
    print("="*80)
    
    fashion_scripts = [
        ("utils/fashion/fashion_build_csv.py", "2-1. BUILD CSV - styles/*.json → fashion_1_raw.csv"),
        ("utils/fashion/row_drop.py", "2-2. ROW DROP - 불필요 서브카테고리 제거"),
        ("utils/fashion/column_drop.py", "2-3. COLUMN DROP - 불필요한 칼럼 제거"),
    ]
    
    # --skip 옵션 처리
    skip_build_csv = args.skip and os.path.exists('dataset/fashion/fashion_1_raw.csv')
    
    for script_path, script_name in fashion_scripts:
        # --skip 옵션이 있고 fashion_build_csv.py를 실행하려는 경우
        if skip_build_csv and 'BUILD CSV' in script_name:
            print(f"\n[건너뛰기] {script_name} - --skip 옵션으로 인해 건너뜁니다.")
            print(f"         기존 파일 사용: dataset/fashion/fashion_1_raw.csv")
            continue
        
        if not os.path.exists(script_path):
            print(f"\n[오류] 스크립트를 찾을 수 없습니다: {script_path}")
            print("작업을 중단합니다.")
            return
        
        success = run_script(script_path, script_name)
        
        if not success:
            print(f"\n[오류] Fashion 전처리 실패: {script_name}에서 오류 발생")
            print("작업을 중단합니다.")
            return
    
    # Fashion 전처리 결과 확인 (column_drop 후 생성된 파일)
    fashion_output_file = 'dataset/fashion/fashion_3_columndrop.csv'
    if not os.path.exists(fashion_output_file):
        print("\n[오류] Fashion 전처리 결과 파일이 생성되지 않았습니다.")
        print(f"      예상 파일: {fashion_output_file}")
        print("작업을 중단합니다.")
        return
    
    # Fashion CSV 헤더 확인 및 검증
    try:
        import pandas as pd
        fashion_check = pd.read_csv(fashion_output_file, nrows=1)
        if 'discountedPrice' not in fashion_check.columns:
            print(f"\n[경고] {fashion_output_file}의 헤더가 손상된 것으로 보입니다.")
            print("fashion_build_csv.py를 다시 실행하거나 fix_fashion_header.py를 실행하세요.")
            print("작업을 중단합니다.")
            return
    except Exception as e:
        print(f"\n[경고] {fashion_output_file} 검증 중 오류: {e}")
        print("작업을 계속 진행하지만 문제가 발생할 수 있습니다.")
    
    # ===== 3단계: 데이터셋 병합 =====
    print("\n" + "="*80)
    print("3단계: 데이터셋 병합")
    print("="*80)
    
    merge_scripts = [
        ("utils/dataset_join/price_unification.py", "3-1. PRICE UNIFICATION - 가격 통일"),
        ("utils/dataset_join/category_mapping.py", "3-2. CATEGORY MAPPING - 카테고리/성별 매핑"),
        ("utils/dataset_join/dataset_merge.py", "3-3. DATASET MERGE - 최종 통합 데이터셋 생성"),
    ]
    
    for script_path, script_name in merge_scripts:
        if not os.path.exists(script_path):
            print(f"\n[오류] 스크립트를 찾을 수 없습니다: {script_path}")
            print("작업을 중단합니다.")
            return
        
        success = run_script(script_path, script_name)
        
        if not success:
            print(f"\n[오류] 데이터셋 병합 실패: {script_name}에서 오류 발생")
            print("\n[문제 진단]")
            
            # 각 단계별로 필요한 파일 확인
            if script_name == "3-1. PRICE UNIFICATION - 가격 통일":
                print("필수 파일 확인:")
                if os.path.exists('dataset/hnm/articles_with_price.csv'):
                    print("  [O] dataset/hnm/articles_with_price.csv")
                else:
                    print("  [X] dataset/hnm/articles_with_price.csv")
                fashion_file = 'dataset/fashion/fashion_3_columndrop.csv'
                if os.path.exists(fashion_file):
                    try:
                        import pandas as pd
                        df_check = pd.read_csv(fashion_file, nrows=1)
                        if 'discountedPrice' in df_check.columns:
                            print(f"  [O] {fashion_file} (discountedPrice 칼럼 존재)")
                        else:
                            print(f"  [X] {fashion_file} (discountedPrice 칼럼 없음 - 헤더 문제)")
                            print("      해결: python fix_fashion_header.py 또는 fashion_build_csv.py 재실행")
                    except:
                        print(f"  [X] {fashion_file} (읽기 오류)")
                else:
                    print(f"  [X] {fashion_file}")
            
            print("작업을 중단합니다.")
            return
    
    # 데이터셋 병합 결과 확인
    if not os.path.exists('dataset/merged_dataset.csv'):
        print("\n[오류] 데이터셋 병합 결과 파일이 생성되지 않았습니다.")
        print("작업을 중단합니다.")
        return
    
    # ===== 4단계: 원핫 인코딩 =====
    print("\n" + "="*80)
    print("4단계: 원핫 인코딩")
    print("="*80)
    
    onehot_script = ("utils/one_hot_encode.py", "4-1. ONE HOT ENCODE - 카테고리 칼럼 원핫 인코딩")
    
    if not os.path.exists(onehot_script[0]):
        print(f"\n[경고] 원핫 인코딩 스크립트를 찾을 수 없습니다: {onehot_script[0]}")
        print("원핫 인코딩을 건너뜁니다.")
    else:
        success = run_script(onehot_script[0], onehot_script[1])
        
        if not success:
            print(f"\n[경고] 원핫 인코딩 실패: {onehot_script[1]}에서 오류 발생")
            print("원본 merged_dataset.csv는 사용 가능합니다.")
        else:
            # 원핫 인코딩 결과 확인
            if os.path.exists('dataset/merged_dataset_onehot.csv'):
                size = os.path.getsize('dataset/merged_dataset_onehot.csv') / (1024**2)  # MB
                print(f"\n[완료] 원핫 인코딩 완료: dataset/merged_dataset_onehot.csv ({size:.1f} MB)")
    
    # ===== 완료 =====
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("[완료] 전체 전처리 파이프라인 완료!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    
    # 최종 결과 확인
    print("\n=== 최종 결과 확인 ===")
    
    final_files = [
        'dataset/hnm/articles_with_price.csv',
        'dataset/fashion/fashion_1_raw.csv',
        'dataset/fashion/fashion_2_rowdrop.csv',
        'dataset/fashion/fashion_3_columndrop.csv',
        'dataset/hnm/articles_with_price_unified.csv',
        'dataset/fashion/fashion_pricied.csv',
        'dataset/hnm/articles_with_price_mapped.csv',
        'dataset/fashion/fashion_mapped.csv',
        'dataset/merged_dataset.csv',
        'dataset/merge_summary.txt',
        'dataset/merged_dataset_onehot.csv'
    ]
    
    all_exist = True
    for file in final_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024**2)  # MB
            print(f"  [O] {file} ({size:.1f} MB)")
        else:
            print(f"  [X] {file} - 파일이 없습니다!")
            all_exist = False
    
    if all_exist:
        print("\n[성공] 모든 최종 파일이 생성되었습니다!")
        print("\n=== 다음 단계 ===")
        print("1. dataset/merged_dataset.csv 파일을 확인하세요 (원본)")
        print("2. dataset/merged_dataset_onehot.csv 파일을 확인하세요 (원핫 인코딩)")
        print("3. dataset/merge_summary.txt 파일로 병합 결과를 검토하세요")
        print("4. 필요시 추가 전처리나 분석을 진행하세요")
    else:
        print("\n[경고] 일부 파일이 생성되지 않았습니다. 로그를 확인하세요.")
        if os.path.exists('dataset/merged_dataset.csv'):
            print("\n[참고] dataset/merged_dataset.csv는 사용 가능합니다.")

if __name__ == "__main__":
    main()

