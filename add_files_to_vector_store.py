import os
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# 데이터가 저장된 폴더 경로
folder_path = "C:\\Users\\user\\law_contents"

# 모든 키와 값을 수집하는 함수
def extract_text_from_json(obj, parent_key=""):
    texts = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            texts.extend(extract_text_from_json(value, full_key))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            full_key = f"{parent_key}[{idx}]"
            texts.extend(extract_text_from_json(item, full_key))
    elif isinstance(obj, str):
        if obj.strip():  # 빈 문자열 무시
            texts.append(obj.strip())
    return texts

# 모든 JSON 파일에서 텍스트 추출
docs = []
for root, _, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith(".json"):
            file_path = os.path.join(root, file_name)
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    texts = extract_text_from_json(data)
                    docs.extend(texts)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

print(f"Number of text segments collected: {len(docs)}")

# 텍스트 분리
splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=20)
split_docs = [splitter.split_text(doc) for doc in docs]
split_docs = [chunk for sublist in split_docs for chunk in sublist]

print(f"Number of split documents: {len(split_docs)}")

# 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(split_docs, embeddings)

# 벡터 스토어 저장
vectorstore.save_local("law_vector")
print("Vector store saved to 'law_vector'")
