#!/usr/bin/env python3
"""
세 파일에서 랜덤한 행 하나씩 추출하여 칼럼 값들을 보여주는 스크립트
"""

import pandas as pd
import numpy as np

def show_random_samples():
    print("=== 세 파일의 랜덤 샘플 데이터 ===")
    
    # 랜덤 시드 설정
    np.random.seed(42)
    
    # 1. articles_with_price.csv
    print("\n1. articles_with_price.csv 랜덤 샘플:")
    print("=" * 60)
    try:
        df1 = pd.read_csv('articles_with_price.csv')
        random_idx1 = np.random.randint(0, len(df1))
        sample1 = df1.iloc[random_idx1]
        
        print(f"행 번호: {random_idx1} / {len(df1)}")
        for col, value in sample1.items():
            print(f"  {col:<30}: {value}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 2. styles.csv
    print("\n\n2. styles.csv 랜덤 샘플:")
    print("=" * 60)
    try:
        df2 = pd.read_csv('styles.csv')
        random_idx2 = np.random.randint(0, len(df2))
        sample2 = df2.iloc[random_idx2]
        
        print(f"행 번호: {random_idx2} / {len(df2)}")
        for col, value in sample2.items():
            print(f"  {col:<30}: {value}")
    except Exception as e:
        print(f"오류: {e}")
    
    # 3. farfetch.csv
    print("\n\n3. farfetch.csv 랜덤 샘플:")
    print("=" * 60)
    try:
        df3 = pd.read_csv('farfetch.csv')
        random_idx3 = np.random.randint(0, len(df3))
        sample3 = df3.iloc[random_idx3]
        
        print(f"행 번호: {random_idx3} / {len(df3)}")
        for col, value in sample3.items():
            print(f"  {col:<30}: {value}")
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    show_random_samples()
