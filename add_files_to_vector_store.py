import os
import json
import shutil
import pandas as pd  # CSV 파일 처리용
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
DATA_FOLDER_PATH = os.getenv("FOLDER_PATH")
RESPONSE_FIELDS_PATH = os.getenv("RESPONSE_FIELDS_PATH")
TEST_MODE = os.getenv("TEST_MODE", "on").lower() == "on"
DELETE_STORE = os.getenv("DELETE_STORE", "on").lower() == "on"

if not VECTOR_STORE_PATH:
    raise ValueError("VECTOR_STORE_PATH 환경 변수가 설정되지 않았습니다.")
if not EMBEDDING_MODEL_NAME:
    raise ValueError("EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")
if not DATA_FOLDER_PATH:
    raise ValueError("FOLDER_PATH 환경 변수가 설정되지 않았습니다.")
if not RESPONSE_FIELDS_PATH:
    raise ValueError("RESPONSE_FIELDS_PATH 환경 변수가 설정되지 않았습니다.")

# 기존 벡터스토어 삭제 (옵션에 따라)
if DELETE_STORE and os.path.exists(VECTOR_STORE_PATH):
    print(f"기존 벡터스토어 삭제 중: {VECTOR_STORE_PATH}")
    shutil.rmtree(VECTOR_STORE_PATH)
    print("기존 벡터스토어 삭제 완료")
elif os.path.exists(VECTOR_STORE_PATH):
    print("기존 벡터스토어 유지")

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 초기화
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings,
)
print("Chroma 벡터스토어 초기화 완료")

# response_field.csv 파일 로드
def load_response_fields(csv_path):
    """
    CSV 파일에서 response field를 동적으로 로드.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"응답 필드 CSV 파일이 {csv_path}에 없습니다.")
    response_fields_df = pd.read_csv(csv_path, encoding="utf-8")
    if "필드" not in response_fields_df.columns:
        raise ValueError("CSV 파일에 '필드' 열이 없습니다.")
    return response_fields_df["필드"].tolist()

try:
    RESPONSE_FIELDS = load_response_fields(RESPONSE_FIELDS_PATH)
    print(f"응답 필드 로드 완료: {RESPONSE_FIELDS}")
except Exception as e:
    print(f"응답 필드 로드 중 오류 발생: {e}")
    raise

# 평탄화 함수
def flatten_json(nested_json, parent_key='', separator='.'):
    """
    중첩된 JSON 구조를 평탄화하여 모든 키-값 쌍을 1단계로 정리.
    """
    items = []
    for key, value in nested_json.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_json(value, new_key, separator=separator).items())
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, f"{new_key}[{idx}]", separator=separator).items())
                else:
                    items.append((f"{new_key}[{idx}]", item))
        else:
            items.append((new_key, value))
    return dict(items)

# 메타데이터 생성
def generate_dynamic_mapping(flat_data, response_fields):
    """
    평탄화된 데이터와 출력 필드 기준으로 동적 매핑 테이블 생성.
    """
    metadata = {}
    for field in response_fields:
        matching_keys = [key for key in flat_data if field in key]
        if matching_keys:
            value = flat_data[matching_keys[0]]
            # 리스트 타입일 경우 문자열로 변환
            if isinstance(value, list):
                metadata[field] = " | ".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                metadata[field] = value
            else:
                print(f"[경고] 지원되지 않는 데이터 타입: {field}: {value}")
        else:
            metadata[field] = None
    # None 값을 가진 키는 제거
    return {k: v for k, v in metadata.items() if v is not None}

def process_json_with_dynamic_mapping(file_path, output_fields):
    """
    JSON 파일에서 메타데이터를 동적 매핑으로 추출하고 문서를 생성.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 평탄화
    flat_data = flatten_json(data)

    # 동적 매핑을 통한 메타데이터 추출
    metadata = generate_dynamic_mapping(flat_data, output_fields)
    
    # 파일 이름 추출 및 메타데이터에 추가
    file_name = os.path.basename(file_path)
    metadata['file_name'] = file_name
    
    print(f"처리 중인 파일: {file_name} (메타데이터 필드 수: {len(metadata)}개)")

    # 전체 문서 내용 생성 (제한 없음)
    document_content = "\n".join([f"{k}: {v}" for k, v in flat_data.items()])
    
    return Document(page_content=document_content, metadata=metadata)

def load_all_documents_from_folder(folder_path, output_fields):
    """
    지정된 폴더 내 모든 JSON 파일에서 문서를 로드합니다.
    """
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith(".json")
    ]
    total_files = len(all_files)
    
    documents = []
    for idx, file_path in enumerate(all_files, 1):
        print(f"\r진행률: [{idx}/{total_files}] ({(idx/total_files)*100:.1f}%)", end="")
        doc = process_json_with_dynamic_mapping(file_path, output_fields)
        documents.append(doc)
    print()  # 진행률 표시 후 줄바꿈

    return documents

# 문서 로드 및 벡터스토어에 추가
try:
    print("문서 로드 중...")
    if TEST_MODE:
        print("테스트 모드: 무작위 파일 샘플링")
        import random
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(DATA_FOLDER_PATH)
            for file in files if file.endswith(".json")
        ]
        sampled_files = random.sample(all_files, min(3, len(all_files)))
        documents = [process_json_with_dynamic_mapping(file, RESPONSE_FIELDS) for file in sampled_files]
    else:
        documents = load_all_documents_from_folder(DATA_FOLDER_PATH, RESPONSE_FIELDS)

    print(f"\n로딩된 문서 수: {len(documents)}")

    print("\n벡터스토어에 추가 중...")
    total_docs = len(documents)
    for idx, doc_batch in enumerate(documents):
        vectorstore.add_documents([doc_batch])
        print(f"\r진행률: [{idx+1}/{total_docs}] ({((idx+1)/total_docs)*100:.1f}%)", end="")
    print("\n문서를 벡터스토어에 성공적으로 추가했습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
    raise