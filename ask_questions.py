import os
import logging
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

# 로그 디렉토리 생성 및 설정
log_directory = "Law_GPT/logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # DEBUG 대신 INFO나 WARNING으로 변경
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "ask_questions.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 환경 변수 로드
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sroberta-nli")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

required_vars = {
    "VECTOR_STORE_PATH": VECTOR_STORE_PATH,
    "PROMPT_FILE_PATH": PROMPT_FILE_PATH,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME
}
for name, val in required_vars.items():
    if not val:
        logger.error(f"{name} 환경 변수가 설정되지 않았습니다.")
        raise ValueError(f"{name} 환경 변수가 설정되지 않았습니다.")

if not LOCAL_LLM_MODEL_NAME:
    raise ValueError("LOCAL_LLM_MODEL_NAME 환경 변수가 설정되지 않았습니다. 로컬 LLM 모델을 지정해야 합니다.")

def get_device(use_gpu: bool):
    if use_gpu and torch.cuda.is_available():
        logger.info("GPU가 사용됩니다.")
        return "cuda"
    else:
        logger.info("GPU가 비활성화되었습니다. CPU를 사용합니다.")
        return "cpu"

device = get_device(USE_GPU)

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
llm_pipeline = pipeline(
    "text-generation",
    model=LOCAL_LLM_MODEL_NAME,
    tokenizer=AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME),
    device=0 if device == "cuda" else -1,
    max_new_tokens=300
)
local_llm = HuggingFacePipeline(pipeline=llm_pipeline)
logger.info(f"로컬 LLM 모델 '{LOCAL_LLM_MODEL_NAME}' 사용")

# Multi-Query Retriever 설정
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=local_llm
)

def generate_context(docs):
    max_chunk_length = 2000
    context_list = []
    for doc in docs:
        metadata = doc.metadata
        content_text = doc.page_content
        for i in range(0, len(content_text), max_chunk_length):
            chunk = content_text[i:i + max_chunk_length]
            source_info = f"출처: {metadata.get('file_name', '알 수 없음')}\n"
            source_info += f"시행일자: {metadata.get('시행일자', '알 수 없음')}"
            context_list.append(f"{source_info}\n\n내용:\n{chunk}\n")
    return "\n---\n".join(context_list)

def main():
    query = input("법률 관련 질문을 입력하세요: ")
    
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        logger.error("검색 결과가 없습니다.")
        print("검색 결과가 없습니다.")
        return

    context = generate_context(retrieved_docs)
    
    if not os.path.exists(PROMPT_FILE_PATH):
        logger.error(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")
        print(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")
        return

    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    full_prompt = prompt_template.format(context=context, question=query)
    
    max_input_length = 8192
    if len(full_prompt) > max_input_length:
        logger.warning("입력 텍스트가 너무 깁니다. 최대 길이에 맞게 자릅니다.")
        full_prompt = full_prompt[:max_input_length]

    try:
        response = local_llm.pipeline(full_prompt)
        answer = response[0]["generated_text"].strip()

        logger.info("\n[생성된 대답]:")
        logger.info(answer)
        print("\n[답변]:")
        print(answer)
    except Exception as e:
        logger.exception("예상치 못한 오류 발생")
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
