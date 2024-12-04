import os
import pandas as pd
import torch
import logging  # 로깅 모듈 추가
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# 로그 디렉토리 생성
log_directory = "Law_GPT/logs"
os.makedirs(log_directory, exist_ok=True)  # 디렉토리가 없으면 생성

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정 (INFO 수준 이상의 로그가 출력됨)
    format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 포맷 설정
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "langchain_rag.log")),  # 로그 파일 설정
        logging.StreamHandler()  # 콘솔 출력 핸들러 추가
    ]
)
logger = logging.getLogger(__name__)  # 로거 생성

# .env 파일 로드
load_dotenv()

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"사용 중인 디바이스: {device}")  # print 대신 logger 사용

# 환경 변수 설정
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

if not VECTOR_STORE_PATH:
    logger.error("VECTOR_STORE_PATH 환경 변수가 설정되지 않았습니다.")
    raise ValueError("VECTOR_STORE_PATH 환경 변수가 설정되지 않았습니다.")
if not PROMPT_FILE_PATH:
    logger.error("PROMPT_FILE_PATH 환경 변수가 설정되지 않았습니다.")
    raise ValueError("PROMPT_FILE_PATH 환경 변수가 설정되지 않았습니다.")
if not EMBEDDING_MODEL_NAME:
    logger.error("EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")
    raise ValueError("EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
logger.info(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 로드
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings,
)
logger.info("Chroma 벡터스토어 로드 완료")

# LLM 설정
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

# 질의 수행
query = "착공 이전 확인 해야할 사항?"
logger.info(f"쿼리 실행 중: {query}")

# 검색 수행
retrieved_docs = vectorstore.similarity_search(query, k=3)
if not retrieved_docs:
    logger.error("검색 결과가 없습니다.")
    raise ValueError("검색 결과가 없습니다.")
logger.info(f"검색된 문서 수: {len(retrieved_docs)}")

# 검색된 문서 정보 로깅 추가
for idx, doc in enumerate(retrieved_docs, 1):
    logger.info(f"\n문서 {idx} 정보:")
    # 전체 메타데이터 출력
    logger.info(f"전체 메타데이터: {doc.metadata}")
    
    # 가능한 모든 출처 관련 키 확인
    possible_source_keys = ['source', 'Source', '출처', 'filename', 'file_name', 'path']
    source = next((doc.metadata.get(key) for key in possible_source_keys if key in doc.metadata), '출처 정보 없음')
    
    logger.info(f"출처: {source}")
    logger.info(f"시행일자: {doc.metadata.get('시행일자', '시행일자 없음')}")
    logger.info(f"문서 내용 미리보기: {doc.page_content[:100]}...")  # 내용 미리보기 추가

# 검색된 문서의 컨텍스트 생성
def generate_context(docs):
    """
    검색된 문서에서 컨텍스트를 생성.
    메타데이터와 전체 내용을 포함하여 구성.
    """
    context_list = []
    for doc in docs:
        metadata = doc.metadata
        content_text = doc.page_content

        # 메타데이터에서 출처 정보 추출
        source_info = "\n".join(f"{key}: {value}" for key, value in metadata.items() 
                              if value and key.lower() in ['출처', 'source', '시행일자'])

        # 전체 내용을 포함하여 컨텍스트 구성
        context_list.append(
            f"출처 정보:\n{source_info}\n\n"
            f"전체 내용:\n{content_text}\n"
        )
    
    return "\n\n---\n\n".join(context_list)  # 문서 구분을 위해 구분자 추가

# 컨텍스트 생성
context = generate_context(retrieved_docs)

# 프롬프트 템플릿 로드
if not os.path.exists(PROMPT_FILE_PATH):
    logger.error(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")
    raise FileNotFoundError(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")

with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
    prompt_template = file.read()

# 프롬프트 생성
full_prompt = prompt_template.format(
    context=context,
    question=query
)

# 디버깅: 생성된 프롬프트 확인
logger.debug("\n[디버깅] 생성된 프롬프트:")
logger.debug(full_prompt)

# LLM 호출 및 응답 처리
try:
    response = llm.invoke(full_prompt)  # 'invoke' 사용하여 텍스트만 반환
    # 최종 응답 출력 (content만 추출)
    logger.info("\n[LLM 응답]:")
    logger.info(response)  # response는 이제 문자열로 텍스트만 포함
except FileNotFoundError as e:
    logger.error(f"필수 파일을 찾을 수 없습니다: {e.filename}")
    raise
except Exception as e:
    logger.exception("예상치 못한 오류 발생")
    raise
