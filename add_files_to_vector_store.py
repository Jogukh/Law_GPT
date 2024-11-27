import os
import json
import time
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 데이터가 저장된 폴더 경로
folder_path = "D:\\law_contents"

def extract_text_from_json(obj):
    """
    JSON 객체에서 모든 문자열 값을 재귀적으로 추출하는 함수.
    """
    texts = []
    if isinstance(obj, dict):
        for value in obj.values():
            texts.extend(extract_text_from_json(value))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text_from_json(item))
    elif isinstance(obj, str):
        if obj.strip():  # 빈 문자열 무시
            texts.append(obj.strip())
    return texts

def process_file(file_path):
    """
    JSON 파일에서 텍스트를 추출하는 함수.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return extract_text_from_json(data)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

# 데이터 수집
docs = []
file_paths = []

print("Starting JSON file processing...")

# 폴더 내 모든 JSON 파일 경로 수집
for root, _, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith(".json"):
            file_paths.append(os.path.join(root, file_name))

print(f"Total files to process: {len(file_paths)}")

# 파일 처리
for file_path in file_paths:
    file_result = process_file(file_path)
    if file_result:
        docs.extend(file_result)

print(f"Finished processing {len(file_paths)} files.")
print(f"Total text segments collected: {len(docs)}")

# 데이터 수집 후 종료 조건 확인
if not docs:
    print("No valid text segments collected. Exiting...")
    exit()

# 텍스트 분리
print("Starting text splitting...")
splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=20)
split_docs = [splitter.split_text(doc) for doc in docs]
split_docs = [chunk for sublist in split_docs for chunk in sublist]

if not split_docs:
    print("No documents after text splitting. Exiting...")
    exit()

print(f"Text splitting completed. Total split documents: {len(split_docs)}")

# 배치 크기 설정
batch_size = 5000  # 한 번에 처리할 텍스트 세그먼트 수
batch_count = len(split_docs) // batch_size + (1 if len(split_docs) % batch_size != 0 else 0)

print(f"Total batches to process: {batch_count}")

# 로컬 모델 준비
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Using local embedding model: all-MiniLM-L6-v2")

# FAISS 인덱스 초기화
embedding_dim = 384  # all-MiniLM-L6-v2의 임베딩 차원
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_texts = []

# 배치 처리
for i in range(batch_count):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(split_docs))
    batch = split_docs[start_idx:end_idx]
    if not batch:
        print(f"Batch {i + 1} is empty. Skipping...")
        continue

    print(f"Processing batch {i + 1}/{batch_count}...")

    try:
        # 로컬 임베딩 생성
        batch_embeddings = model.encode(batch, batch_size=64, show_progress_bar=True)
        faiss_index.add(np.array(batch_embeddings))  # FAISS 인덱스에 추가
        faiss_texts.extend(batch)  # 텍스트 저장

    except Exception as e:
        print(f"Error processing batch {i + 1}: {e}")
        continue

if faiss_index.ntotal == 0:
    print("No vectors added to FAISS index. Exiting...")
    exit()

print("All batches processed. Saving vector store...")

# 벡터 스토어 생성 및 저장
vectorstore = FAISS(embedding_model=None, index=faiss_index, texts=faiss_texts)
output_path = os.path.abspath("law_vector")
vectorstore.save_local(output_path)
print(f"Vector store saved successfully to: {output_path}")
