import shutil
import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
DATA_FOLDER_PATH = os.getenv("FOLDER_PATH")
TEST_MODE = os.getenv("TEST_MODE", "on").lower() == "on"

if not VECTOR_STORE_PATH:
    raise ValueError("VECTOR_STORE_PATH 환경 변수가 설정되지 않았습니다.")
if not EMBEDDING_MODEL_NAME:
    raise ValueError("EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")
if not DATA_FOLDER_PATH:
    raise ValueError("FOLDER_PATH 환경 변수가 설정되지 않았습니다.")

# 기존 벡터스토어 초기화
if os.path.exists(VECTOR_STORE_PATH):
    print(f"기존 벡터스토어 삭제 중: {VECTOR_STORE_PATH}")
    shutil.rmtree(VECTOR_STORE_PATH)  # 벡터스토어 디렉토리 삭제
    print("기존 벡터스토어 삭제 완료")

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터 스토어 생성 또는 로드
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,  # 저장 디렉토리 설정
    embedding_function=embeddings,
)

# persist() 메서드 호출 없이도 저장 가능
print("문서를 벡터스토어에 성공적으로 추가했습니다.")

# 주요 키워드 정의
KEYWORDS = ['명', '내용', '일자']

def flatten_json(nested_json, parent_key='', separator='.'):
    """
    중첩된 JSON 구조를 평탄화하여 모든 키-값 쌍을 1단계로 정리.
    """
    items = []
    for key, value in nested_json.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            # 재귀적으로 탐색
            items.extend(flatten_json(value, new_key, separator=separator).items())
        elif isinstance(value, list):
            # 리스트 내 객체도 재귀적으로 처리
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, f"{new_key}[{idx}]", separator=separator).items())
                else:
                    items.append((f"{new_key}[{idx}]", item))
        else:
            items.append((new_key, value))
    return dict(items)

def extract_keywords(data, keywords=KEYWORDS):
    """
    JSON 데이터에서 주요 키워드가 포함된 모든 키-값을 추출합니다.
    
    Args:
        data (dict or list): JSON 데이터 (dict 또는 list 형태).
        keywords (list): 주요 키워드 목록.
        
    Returns:
        dict: 주요 키워드가 포함된 키-값 쌍.
    """
    result = {}

    def recursive_search(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{path}.{key}" if path else key
                # 키 이름에 주요 키워드 포함 여부 확인
                if any(keyword in key for keyword in keywords):
                    result[full_key] = value
                # 하위 키 탐색
                recursive_search(value, full_key)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                recursive_search(item, f"{path}[{idx}]")

    recursive_search(data)
    return result

def clean_metadata(metadata):
    """
    메타데이터에서 복잡한 데이터 타입(예: 리스트, 딕셔너리)을 문자열로 변환하거나 제거합니다.
    
    Args:
        metadata (dict): 원본 메타데이터.
    
    Returns:
        dict: 정리된 메타데이터.
    """
    result = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            # 허용된 데이터 타입은 그대로 추가
            result[key] = value
        elif isinstance(value, list):
            # 리스트는 문자열로 변환
            result[key] = " | ".join(map(str, value))
        elif isinstance(value, dict):
            # 딕셔너리는 JSON 문자열로 변환
            import json
            result[key] = json.dumps(value, ensure_ascii=False)
        else:
            # 기타 데이터 타입은 문자열로 변환
            result[key] = str(value)
    return result
def metadata_func(record: dict, metadata: dict) -> dict:
    """
    주요 키워드를 추출하여 메타데이터에 추가하고, 복잡한 데이터를 필터링합니다.
    
    Args:
        record (dict): JSON 데이터의 각 레코드.
        metadata (dict): 기본 메타데이터.
        
    Returns:
        dict: 정리된 메타데이터.
    """
    # JSON을 평탄화하여 모든 키 탐색
    flat_metadata = flatten_json(record)
    metadata.update(flat_metadata)

    # 주요 키워드 추출 및 메타데이터 업데이트
    keyword_matches = extract_keywords(flat_metadata)
    metadata.update({"주요정보": keyword_matches})

    # 복잡한 데이터 타입 클리닝 (이전에 누락된 경우 처리)
    return clean_metadata(metadata)

def load_documents_from_json(file_path):
    """
    JSON 파일에서 문서를 로드하고 주요 키워드 기반 메타데이터를 추가합니다.
    """
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",  # JSON 전체를 처리
        content_key=None,  # 콘텐츠를 추출하지 않고 모든 데이터를 메타데이터로 처리
        text_content=False,  # 콘텐츠가 문자열이 아닌 경우에도 처리
        metadata_func=metadata_func,  # 메타데이터 처리 함수
    )
    return loader.load()

def load_all_documents_from_folder(folder_path):
    """
    지정된 폴더 내 모든 JSON 파일에서 문서를 로드합니다.
    """
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith(".json")
    ]

    all_documents = []
    for file_path in all_files:
        print(f"파일 처리 중: {file_path}")
        documents = load_documents_from_json(file_path)
        all_documents.extend(documents)

    return all_documents


# 문서 로드 및 처리
try:
    if TEST_MODE:
        print("테스트 모드 활성화: 랜덤 샘플 파일 선택")
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(DATA_FOLDER_PATH)
            for file in files if file.endswith(".json")
        ]
        sampled_files = random.sample(all_files, 3)
        documents = []
        for file in sampled_files:
            documents.extend(load_documents_from_json(file))
    else:
        print("전체 데이터 로드 중...")
        documents = load_all_documents_from_folder(DATA_FOLDER_PATH)

    print(f"로딩된 문서 수: {len(documents)}")
except Exception as e:
    print(f"문서 처리 중 오류 발생: {e}")
    raise

# 벡터스토어에 문서 추가
try:
    print("벡터스토어에 문서 추가 중...")
    
    # 문서 메타데이터를 클린 처리한 새로운 Document 객체 생성
    cleaned_documents = [
        Document(
            page_content=doc.page_content,  # 원본 내용 유지
            metadata=clean_metadata(doc.metadata)  # 메타데이터 정리
        )
        for doc in documents
    ]
    
    # 정리된 문서를 벡터스토어에 추가
    vectorstore.add_documents(cleaned_documents)
    print("문서를 벡터스토어에 성공적으로 추가했습니다.")
except Exception as e:
    print(f"벡터스토어 업데이트 중 오류 발생: {e}")
    raise