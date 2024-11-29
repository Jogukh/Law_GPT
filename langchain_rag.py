import os
import json
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
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "chroma_vectorstore")
HIERARCHY_FOLDER = os.getenv("HIERARCHY_FOLDER", "law_hierarchy")
PROMPT_FILE_PATH = os.getenv("PROMPT_FILE_PATH", "prompt.txt")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print(f"임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 완료")

# 벡터스토어 로드
try:
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )
    print("Chroma 벡터스토어 로드 완료")
except Exception as e:
    raise RuntimeError(f"Chroma 벡터스토어 로드 실패: {e}")

# 법령체계도 로드 함수
def load_legal_hierarchy(folder_path):
    hierarchy_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(" 법령체계도.json"):
            law_name = file_name.replace(" 법령체계도.json", "")
            with open(os.path.join(folder_path, file_name), encoding="utf-8") as f:
                data = json.load(f)
                hierarchy = data.get("법령체계도", {}).get("상하위법", {})
                hierarchy_data[law_name] = {
                    "상위법": hierarchy.get("법률", {}).get("상위법", []),
                    "관련법령": hierarchy.get("법률", {}).get("관련법령", [])
                }
    return hierarchy_data

# 법령체계도 로드
hierarchy_data = load_legal_hierarchy(HIERARCHY_FOLDER)
print(f"법령체계도 로드 완료. 총 {len(hierarchy_data)}개의 법령 체계 정보를 로드했습니다.")

# LLM 설정
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

# 질의 수행
query = "가장 최근에 시행된 법령을 가져오세요."
print(f"쿼리 실행 중: {query}")

# 검색 수행
retrieved_docs = vectorstore.similarity_search(query, k=500)
if not retrieved_docs:
    raise ValueError("검색 결과가 없습니다.")
print(f"검색된 문서 수: {len(retrieved_docs)}")

# 검색 결과에서 첫 번째 문서의 메타데이터 및 주요 내용
top_doc = retrieved_docs[0]
top_metadata = top_doc.metadata

# 디버깅용 검색된 문서의 메타데이터 출력
print("검색된 문서의 메타데이터:", top_metadata)

# 안전하게 메타데이터 값 가져오기 (기본값 설정)
법령명 = top_metadata.get("법령명", "알 수 없음")
시행일자 = top_metadata.get("시행일자", "알 수 없음")
소관부처명 = top_metadata.get("소관부처명", "알 수 없음")
부처연락처 = top_metadata.get("부처연락처", "알 수 없음")
주요_법령_내용_및_해석 = top_doc.page_content

# 디버깅용 메타데이터 확인
print(f"디버깅: 법령명={법령명}, 시행일자={시행일자}, 소관부처명={소관부처명}, 부처연락처={부처연락처}")

# 상하위 문서 검색
related_docs = []
if 법령명 in hierarchy_data:
    hierarchy = hierarchy_data[법령명]
    related_keys = hierarchy.get("상위법", []) + hierarchy.get("관련법령", [])
    for key in related_keys:
        related_docs.extend(vectorstore.similarity_search(key, k=50))

# 관련 문서들의 메타데이터 수집
related_metadata_queries = [doc.metadata for doc in related_docs]

# 관련 문서 메타데이터 생성
related_metadata = "\n".join(
    [f"{doc.get('법령명', '알 수 없음')} - {doc.get('시행일자', '알 수 없음')}" for doc in related_metadata_queries]
)

# 컨텍스트 생성
context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])  # 상위 5개 문서만 포함
context += "\n\n" + "\n\n".join([doc.page_content for doc in related_docs[:5]])  # 상위 5개 관련 문서 포함
context = context[:3000]  # 컨텍스트 길이 제한

# 프롬프트 템플릿 로드
if not os.path.exists(PROMPT_FILE_PATH):
    raise FileNotFoundError(f"프롬프트 파일이 {PROMPT_FILE_PATH}에 없습니다.")

with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
    prompt_template = file.read()

# 프롬프트 생성
try:
    full_prompt = prompt_template.format(
        related_metadata_queries=related_metadata_queries,
        top_metadata={
            "법령명": 법령명,
            "시행일자": 시행일자,
            "소관부처명": 소관부처명,
            "부처연락처": 부처연락처
        },
        related_metadata=related_metadata,
        context=context,
        question=query
    )
except KeyError as e:
    raise KeyError(f"프롬프트 템플릿에서 '{e.args[0]}' 변수에 해당하는 값이 없습니다. 템플릿과 코드 변수를 확인하세요.")

# LLM 호출
response = llm.invoke(full_prompt)
print("\n[응답]\n", response)
