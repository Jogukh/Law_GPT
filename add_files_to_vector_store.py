import os
import json
import random
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb.config import Settings
from langchain_chroma import Chroma

# 환경 변수 로드
load_dotenv()

# 임베딩 모델 및 경로 설정
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "chroma_vectorstore")
FOLDER_PATH = os.getenv("FOLDER_PATH")

if not FOLDER_PATH:
    raise ValueError(".env 파일에서 FOLDER_PATH를 설정해야 합니다.")

print(f"VECTORSTORE_PATH: {VECTORSTORE_PATH}")
print(f"FOLDER_PATH: {FOLDER_PATH}")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# 텍스트 및 메타데이터 추출 함수
def extract_texts_and_metadata(folder_path, selected_files=None):
    texts, metadatas = [], []
    for root, _, files in os.walk(folder_path):
        if selected_files:
            files = [f for f in files if os.path.join(root, f) in selected_files]
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        extracted_texts = extract_text_from_json(data)
                        for text in extracted_texts:
                            texts.append(text)
                            metadatas.append({"source": file_path})
                    except Exception as e:
                        print(f"파일 처리 중 오류 발생: {file_path}, {e}")
    return texts, metadatas


def extract_text_from_json(data):
    texts = []
    if isinstance(data, dict):
        for key, value in data.items():
            texts.extend(extract_text_from_json(value))
    elif isinstance(data, list):
        for item in data:
            texts.extend(extract_text_from_json(item))
    elif isinstance(data, str) and data.strip():
        texts.append(data.strip())
    return texts


# 텍스트 분할 함수
def split_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    documents = [Document(page_content=text) for text in texts]
    return splitter.split_documents(documents)


# Chroma 초기화 및 데이터 추가
def create_and_add_to_chroma_store(texts, metadatas, collection_name="law_documents"):
    if not texts:
        raise ValueError("유효한 텍스트가 없습니다. 데이터를 확인하세요.")
    
    print(f"컬렉션 '{collection_name}' 생성 및 문서 추가 중...")

    # 로컬 DB 설정
    settings = Settings(
        chroma_db_impl="duckdb+parquet",  # 로컬 DB 설정
        persist_directory=VECTORSTORE_PATH
    )

    # 벡터 스토어 초기화
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding_model,
        client_settings=settings
    )

    # 텍스트를 문서로 변환하여 추가
    documents = [
        Document(page_content=text, metadata=metadata) 
        for text, metadata in zip(texts, metadatas)
    ]
    vectorstore.add_documents(documents=documents)
    vectorstore.persist()
    print(f"벡터스토어가 '{VECTORSTORE_PATH}' 경로에 저장되었습니다.")

# 랜덤 테스트 함수
def select_random_folder_and_files(base_path, max_files=3):
    folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    if not folders:
        raise ValueError("데이터 폴더가 비어 있습니다.")
    random_folder = random.choice(folders)
    files = [os.path.join(random_folder, file) for file in os.listdir(random_folder) if file.endswith(".json")]
    selected_files = random.sample(files, min(len(files), max_files))
    return random_folder, selected_files


# 문서 검색 함수
def search_documents(query, collection_name="law_documents", k=3):
    try:
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=VECTORSTORE_PATH
        )
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embedding_model,
            client_settings=settings
        )
        results = vectorstore.similarity_search(query=query, k=k)
        if not results:
            print("검색 결과가 없습니다.")
        else:
            for result in results:
                print(f"내용: {result.page_content}")
                print(f"메타데이터: {result.metadata}")
                print("="*50)
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")


# 주요 실행 로직
if __name__ == "__main__":
    mode = input("실행 모드 선택 (1: 랜덤 테스트, 2: 전체 처리): ")
    
    if mode == "1":
        print("랜덤 테스트 실행 중...")
        random_folder, selected_files = select_random_folder_and_files(FOLDER_PATH, max_files=3)
        print(f"랜덤 선택 폴더: {random_folder}")
        print("선택된 파일:")
        for file in selected_files:
            print(f"  - {file}")
        texts, metadatas = extract_texts_and_metadata(FOLDER_PATH, selected_files=selected_files)
    elif mode == "2":
        print("전체 데이터 처리 중...")
        texts, metadatas = extract_texts_and_metadata(FOLDER_PATH)
    else:
        raise ValueError("잘못된 모드 선택. 1 또는 2를 입력하세요.")
    
    print(f"총 텍스트 개수: {len(texts)}")
    create_and_add_to_chroma_store(texts, metadatas, collection_name="law_documents")
    
    query = input("검색어를 입력하세요: ")
    search_documents(query=query, collection_name="law_documents")
