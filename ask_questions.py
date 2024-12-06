import os
import logging
from dotenv import load_dotenv
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 로그 디렉토리 생성 및 설정
log_directory = "Law_GPT/logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "ask_questions.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 환경 변수 로드 및 검증
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME")  # 로컬 LLM 모델 이름

for var, name in [(VECTOR_STORE_PATH, "VECTOR_STORE_PATH"),
                (PROMPT_FILE_PATH, "PROMPT_FILE_PATH"),
                (EMBEDDING_MODEL_NAME, "EMBEDDING_MODEL_NAME"),
                (LOCAL_LLM_MODEL_NAME, "LOCAL_LLM_MODEL_NAME")]:
    if not var:
        logger.error(f"{name} 환경 변수가 설정되지 않았습니다.")
        raise ValueError(f"{name} 환경 변수가 설정되지 않았습니다.")

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"사용 중인 디바이스: {device}")

# 임베딩 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
logger.info(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 로드
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings,
)
logger.info("Chroma 벡터스토어 로드 완료")

# 로컬 LLM 설정
tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL_NAME).to(device)
local_llm = HuggingFacePipeline(
    pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1),
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2
)
logger.info(f"로컬 LLM 모델 '{LOCAL_LLM_MODEL_NAME}' 로드 완료")

# Multi-Query Retriever 설정
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=local_llm
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
    query = input("법률 관련 질문을 입력하세요: ")
    
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        logger.error("검색 결과가 없습니다.")
        print("검색 결과가 없습니다.")
        return
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
    
    # 프롬프트 템플릿 로드
    if not os.path.exists(PROMPT_FILE_PATH):
        logger.error(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")
        print(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")
        return
    
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
        response = local_llm(full_prompt)
        logger.info("\n[LLM 응답]:")
        logger.info(response)
        
        answer = response[0]['generated_text'].strip()
        logger.info("\n[생성된 대답]:")
        logger.info(answer)
        
        print("\n[답변]:")
        print(answer)
    
    except Exception as e:
        logger.exception("예상치 못한 오류 발생")
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
