import os
import pandas as pd
import torch
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# 로그 디렉토리 생성 및 설정
log_directory = "Law_GPT/logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "langchain_rag.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"사용 중인 디바이스: {device}")

# 환경 변수 로드 및 검증
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

for var, name in [(VECTOR_STORE_PATH, "VECTOR_STORE_PATH"),
                 (PROMPT_FILE_PATH, "PROMPT_FILE_PATH"),
                 (OPENAI_API_KEY, "OPENAI_API_KEY"),
                 (EMBEDDING_MODEL_NAME, "EMBEDDING_MODEL_NAME")]:
    if not var:
        logger.error(f"{name} 환경 변수가 설정되지 않았습니다.")
        raise ValueError(f"{name} 환경 변수가 설정되지 않았습니다.")

# OpenAI 임베딩 설정
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
logger.info(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 로드
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings,
)
logger.info("Chroma 벡터스토어 로드 완료")

# LangChain OpenAI LLM 설정
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=4096
)

# Multi-Query Retriever 설정
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)

def generate_context(docs):
    context_list = []
    for doc in docs:
        metadata = doc.metadata
        content_text = doc.page_content
        source_info = f"출처: {metadata.get('file_name', '알 수 없음')}\n"
        source_info += f"시행일자: {metadata.get('시행일자', '알 수 없음')}"
        context_list.append(
            f"{source_info}\n\n내용:\n{content_text}\n"
        )
    return "\n---\n".join(context_list)

def main():
    # 사용자 입력 또는 다른 방식으로 query 정의
    query = input("질문을 입력하세요: ")
    
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        logger.error("검색 결과가 없습니다.")
        raise ValueError("검색 결과가 없습니다.")
    logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
    
    # 검색된 문서 정보 로깅
    for idx, doc in enumerate(retrieved_docs, 1):
        logger.info(f"\n문서 {idx} 정보:")
        logger.info(f"전체 메타데이터: {doc.metadata}")
        source = next((doc.metadata.get(key) for key in ['source', 'Source', '출처', 'filename', 'file_name', 'path'] if key in doc.metadata), '출처 정보 없음')
        logger.info(f"출처: {source}")
        logger.info(f"시행일자: {doc.metadata.get('시행일자', '시행일자 없음')}")
        logger.info(f"문서 내용 미리보기: {doc.page_content[:100]}...")

    # 검색된 문서 컨텍스트 생성
    context = generate_context(retrieved_docs)

    # 롬프트 템플릿 로드
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
    logger.debug("\n[버깅] 생성된 프롬프트:")
    logger.debug(full_prompt)

    # LLM 호출 및 응답 처리
    try:
        # 메시지 형식으로 변경
        messages = [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": full_prompt}
        ]
        response = llm.generate(messages)
        logger.info("\n[LLM 응답]:")
        logger.info(response)

        answer = response.generations[0].message['content'].strip()  # 응답 처리 방식 수정
        logger.info("\n[생성된 대답]:")
        logger.info(answer)

    except FileNotFoundError as e:
        logger.error(f"필수 파일을 찾을 수 없습니다: {e.filename}")
        raise
    except Exception as e:
        logger.exception("예상치 못한 오류 발생")
        raise

if __name__ == "__main__":
    main()
