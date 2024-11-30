import os
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
query = "가장 최근에 시행된 법령을 가져오세요."
print(f"쿼리 실행 중: {query}")

# 검색 수행
retrieved_docs = vectorstore.similarity_search(query, k=200)
if not retrieved_docs:
    raise ValueError("검색 결과가 없습니다.")
print(f"검색된 문서 수: {len(retrieved_docs)}")

# 디버깅: 검색된 문서의 메타데이터 확인
for idx, doc in enumerate(retrieved_docs[:3]):
    print(f"문서 {idx + 1} 메타데이터: {doc.metadata}")

# 컨텍스트 생성 함수
def generate_context(retrieved_docs):
    context_list = []
    for doc in retrieved_docs:
        metadata = doc.metadata
        content_summary = doc.page_content[:200]  # 내용 요약
        # 메타데이터 기본값 처리
        law_name = metadata.get('법령명', '법령명 정보 없음')
        effective_date = metadata.get('시행일자', '시행일자 정보 없음')
        department = metadata.get('소관부처명', '소관 부처 정보 없음')
        contact_info = metadata.get('부처연락처', '연락처 정보 없음')

        context_list.append(
            f"Law Name: {law_name}\n"
            f"Effective Date: {effective_date}\n"
            f"Responsible Department: {department}\n"
            f"Contact Info: {contact_info}\n"
            f"Content: {content_summary}\n"
        )
    if not context_list:
        raise ValueError("컨텍스트 생성에 실패했습니다. 검색된 문서가 유효하지 않습니다.")
    return "\n\n".join(context_list[:5])  # 상위 5개 문서 사용

# 프롬프트 템플릿 로드 및 검증
if not os.path.exists(PROMPT_FILE_PATH):
    raise FileNotFoundError(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")

with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
    prompt_template = file.read()

# 프롬프트 템플릿 수정
full_prompt = prompt_template.format(
    top_metadata={
        "법령명": retrieved_docs[0].metadata.get("법령명", "법령명 정보 없음"),
        "시행일자": retrieved_docs[0].metadata.get("시행일자", "시행일자 정보 없음"),
        "소관부처명": retrieved_docs[0].metadata.get("소관부처명", "소관 부처 정보 없음"),
        "부처연락처": retrieved_docs[0].metadata.get("부처연락처", "연락처 정보 없음"),
    },
    context=generate_context(retrieved_docs),
    question=query
)

# 디버깅: 생성된 프롬프트 확인
print("\n[디버깅] LLM 호출용 프롬프트:")
print(full_prompt)

# LLM 호출
response = llm.invoke(full_prompt)

# 디버깅: LLM 응답 확인
print("\n[응답]\n", response)