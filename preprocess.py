#!/usr/bin/env python3
"""
H&M 데이터 전처리 파이프라인
articles.csv와 transactions_train.csv를 전처리하여 최종 데이터셋 생성

실행 순서:
1. hnm_join.py - articles.csv + transactions 가격 데이터 JOIN 및 product_code별 병합
2. hnm_column_drop.py - 불필요한 칼럼 제거
3. hnm_row_drop.py - 불필요한 행 제거 (악세서리, 속옷 등) + 가격 스케일링
4. hnm_column_split_densify.py - product_group_name 칼럼 제거
5. hnm_one_hot_encode.py - 카테고리 칼럼 원핫인코딩
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
            encoding='utf-8',
            errors='replace'  # 디코딩 에러 무시
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

def main():
    start_time = datetime.now()
    
    print("="*80)
    print("H&M 데이터 전처리 파이프라인 시작")
    print("="*80)
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 필수 파일 확인
    print("\n1. 필수 파일 확인...")
    required_files = ['dataset/hnm/articles.csv', 'dataset/hnm/transactions_train.csv']
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
        print("작업을 중단합니다.")
        return
    
    # 스크립트 목록 (실행 순서대로)
    scripts = [
        ("utils/hnm_join.py", "1. JOIN - articles + 가격 데이터 병합"),
        ("utils/hnm_column_drop.py", "2. COLUMN DROP - 불필요한 칼럼 제거"),
        ("utils/hnm_row_drop.py", "3. ROW DROP - 불필요한 행 제거 + 가격 스케일링"),
        ("utils/hnm_column_split_densify.py", "4. COLUMN SPLIT - product_group_name 제거"),
        ("utils/hnm_one_hot_encode.py", "5. ONE-HOT ENCODE - 카테고리 칼럼 인코딩"),
    ]
    
    # 각 스크립트 실행
    print("\n2. 전처리 스크립트 실행...")
    
    for script_path, script_name in scripts:
        if not os.path.exists(script_path):
            print(f"\n[오류] 스크립트를 찾을 수 없습니다: {script_path}")
            print("작업을 중단합니다.")
            return
        
        success = run_script(script_path, script_name)
        
        if not success:
            print(f"\n[오류] 전처리 파이프라인 실패: {script_name}에서 오류 발생")
            print("작업을 중단합니다.")
            return
    
    # 완료
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("[완료] 전처리 파이프라인 완료!")
    print("="*80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    
    # 최종 파일 확인
    if os.path.exists('dataset/hnm/articles_with_price.csv'):
        size = os.path.getsize('dataset/hnm/articles_with_price.csv') / (1024**2)  # MB
        print(f"\n최종 출력 파일: dataset/hnm/articles_with_price.csv ({size:.1f} MB)")
    else:
        print("\n[경고] 최종 출력 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()

