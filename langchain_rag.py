import os
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# .env 파일 로드
load_dotenv()

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 환경 변수 설정
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

if not VECTOR_STORE_PATH:
    raise ValueError("VECTOR_STORE_PATH 환경 변수가 설정되지 않았습니다.")
if not PROMPT_FILE_PATH:
    raise ValueError("PROMPT_FILE_PATH 환경 변수가 설정되지 않았습니다.")
if not EMBEDDING_MODEL_NAME:
    raise ValueError("EMBEDDING_MODEL_NAME 환경 변수가 설정되지 않았습니다.")

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 로드
vectorstore = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embeddings,
)
print("Chroma 벡터스토어 로드 완료")

# LLM 설정
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

# 질의 수행
query = "폐수 배출 기준?"
print(f"쿼리 실행 중: {query}")

# 검색 수행
retrieved_docs = vectorstore.similarity_search(query, k=20)
if not retrieved_docs:
    raise ValueError("검색 결과가 없습니다.")
print(f"검색된 문서 수: {len(retrieved_docs)}")

# 검색된 문서의 컨텍스트 생성
def generate_context(docs):
    """
    검색된 문서에서 컨텍스트를 생성.
    메타데이터를 그대로 포함하여 구성.
    """
    context_list = []
    for doc in docs:
        metadata = doc.metadata

        # 메타데이터를 키-값 쌍으로 정리
        metadata_text = "\n".join(f"{key}: {value}" for key, value in metadata.items() if value != "정보 없음")

        # 문서의 원본 내용 포함
        content_text = doc.page_content

        # 컨텍스트 구성
        context_list.append(
            f"메타데이터:\n{metadata_text if metadata_text else '메타데이터 정보 없음'}\n\n"
            f"문서 원본:\n{content_text}\n"
        )
    return "\n\n".join(context_list)

# 컨텍스트 생성
context = generate_context(retrieved_docs)

# 프롬프트 템플릿 로드
if not os.path.exists(PROMPT_FILE_PATH):
    raise FileNotFoundError(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")

with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
    prompt_template = file.read()

# 프롬프트 생성
full_prompt = prompt_template.format(
    context=context,
    question=query
)

# 디버깅: 생성된 프롬프트 확인
print("\n[디버깅] 생성된 프롬프트:")
print(full_prompt)

# 검색된 문서의 메타데이터 확인
for idx, doc in enumerate(retrieved_docs):
    print(f"문서 {idx + 1} 메타데이터: {doc.metadata}")

# LLM 호출 및 응답 처리
response = llm.invoke(full_prompt)

# 최종 응답 출력
print("\n[LLM 응답]:")
print(response)
