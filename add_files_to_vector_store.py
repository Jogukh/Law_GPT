import os
import json
import logging
import shutil
import random
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 로그 디렉토리 생성 및 설정
log_directory = "Law_GPT/logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "add_files_to_vector_store.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
dotenv_path = ".env"
load_dotenv(dotenv_path)

# 환경 변수 로드
FOLDER_PATH = os.getenv("FOLDER_PATH")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
RESPONSE_FIELDS_PATH = os.getenv("RESPONSE_FIELDS_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sroberta-nli")
TEST_MODE = os.getenv("TEST_MODE", "off").lower() == "on"
DELETE_STORE = os.getenv("DELETE_STORE", "on").lower() == "on"

# 필수 변수 검증
required_vars = {
    "VECTOR_STORE_PATH": VECTOR_STORE_PATH,
    "RESPONSE_FIELDS_PATH": RESPONSE_FIELDS_PATH,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
    "FOLDER_PATH": FOLDER_PATH
}
for name, val in required_vars.items():
    if not val:
        logger.error(f"{name} 환경 변수가 설정되지 않았습니다.")
        raise ValueError(f"{name} 환경 변수가 설정되지 않았습니다.")

# 필요시 기존 벡터스토어 삭제
if DELETE_STORE and os.path.exists(VECTOR_STORE_PATH):
    logger.info(f"기존 벡터스토어 삭제 중: {VECTOR_STORE_PATH}")
    shutil.rmtree(VECTOR_STORE_PATH)
    logger.info("기존 벡터스토어 삭제 완료")

def load_response_fields(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"응답 필드 CSV 파일이 {csv_path}에 없습니다.")
    response_fields_df = pd.read_csv(csv_path, encoding="utf-8")
    if "필드" not in response_fields_df.columns:
        raise ValueError("CSV 파일에 '필드' 열이 없습니다.")
    return response_fields_df["필드"].tolist()

try:
    RESPONSE_FIELDS = load_response_fields(RESPONSE_FIELDS_PATH)
    logger.info(f"응답 필드 로드 완료: {RESPONSE_FIELDS}")
except Exception as e:
    logger.error(f"응답 필드 로드 중 오류 발생: {e}")
    raise

def flatten_json(nested_json, parent_key='', separator='.'):
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

def generate_dynamic_mapping(flat_data, response_fields):
    metadata = {}
    for field in response_fields:
        matching_keys = [key for key in flat_data if field in key]
        if matching_keys:
            value = flat_data[matching_keys[0]]
            if isinstance(value, list):
                metadata[field] = " | ".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                metadata[field] = value
            else:
                logger.warning(f"지원되지 않는 데이터 타입: {field}: {value}")
        else:
            metadata[field] = None
    return {k: v for k, v in metadata.items() if v is not None}

def process_json_with_dynamic_mapping(file_path, output_fields):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flat_data = flatten_json(data)
    metadata = generate_dynamic_mapping(flat_data, output_fields)
    
    file_name = os.path.basename(file_path)
    metadata['file_name'] = file_name
    
    document_content = "\n".join([f"{k}: {v}" for k, v in flat_data.items()])
    return Document(page_content=document_content, metadata=metadata)

def load_all_documents_from_folder(folder_path, output_fields):
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith(".json")
    ]
    if TEST_MODE:
        sampled_files = random.sample(all_files, min(3, len(all_files)))
        target_files = sampled_files
    else:
        target_files = all_files

    documents = []
    for idx, file_path in enumerate(target_files, 1):
        logger.info(f"문서 처리 중: [{idx}/{len(target_files)}] {file_path}")
        doc = process_json_with_dynamic_mapping(file_path, output_fields)
        documents.append(doc)

    return documents

# 임베딩 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
logger.info(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 초기화
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings,
)
logger.info("Chroma 벡터스토어 초기화 완료")

# 문서 로드 및 벡터스토어 인덱싱
documents = load_all_documents_from_folder(FOLDER_PATH, RESPONSE_FIELDS)
logger.info(f"로딩된 문서 수: {len(documents)}")

if documents:
    vectorstore.add_documents(documents)
    logger.info("문서를 벡터스토어에 성공적으로 추가했습니다.")
else:
    logger.warning("인덱싱할 문서가 없습니다.")
