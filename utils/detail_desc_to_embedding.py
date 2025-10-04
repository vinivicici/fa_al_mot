import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# CSV 파일 불러오기
try:
    df = pd.read_csv('articles_with_price.csv')
    print("CSV 파일을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 'articles_with_price.csv' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# 'detail_desc' column NaN 처리
# NaN 값을 빈 문자열로 대체
df['detail_desc'] = df['detail_desc'].fillna('')
print("'detail_desc' column의 NaN 값을 빈 문자열로 대체했습니다.")

# SentenceTransformer 모델 로드
# 'all-MiniLM-L6-v2' 모델 사용
model = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer 모델을 성공적으로 로드했습니다.")

# 'detail_desc' column의 텍스트를 리스트로 변환
sentences = df['detail_desc'].tolist()

# 텍스트 임베딩 생성
print(f"{len(sentences)}개의 설명에 대해 임베딩을 생성합니다.")
embeddings = model.encode(sentences, show_progress_bar=True)
print("임베딩 생성이 완료되었습니다.")

# 생성된 embdeddings를 새로운 column으로 추가
df['desc_embedding'] = list(embeddings)

# 변경된 DataFrame을 새로운 CSV 파일로 저장
df.to_csv('articles_with_embeddings.csv', index=False)
print("임베딩이 추가된 파일이 'articles_with_embeddings.csv'로 저장되었습니다.")