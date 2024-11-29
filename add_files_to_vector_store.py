import os
from glob import glob
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

FOLDER_PATH = os.getenv("FOLDER_PATH", "law_contents")  # 기본 폴더 경로
VECTOR_STORE_PATH = os.getenv("VECTORSTORE_PATH", "law_vector")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# JSON 데이터 로드 함수 (하위 폴더 포함)
def load_json_documents(folder_path):
    """하위 폴더의 모든 JSON 파일을 Document로 로드합니다."""
    documents = []
    for root, dirs, files in os.walk(folder_path):  # 폴더 및 하위 폴더 탐색
        for file in files:
            if file.endswith(".json"):  # JSON 파일만 처리
                file_path = os.path.join(root, file)
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.["본문"]["법령"]["조문내용"][]',  # 특수 문자를 처리하도록 수정
                    metadata_func=lambda record, metadata: {
                        "법령명": record.get("법령명", "알 수 없음"),
                        "조문번호": record.get("조문번호", "알 수 없음"),
                        "source": file_path,
                    }
                )
                documents.extend(loader.load())
    return documents

# 데이터 로드
try:
    docs = load_json_documents(FOLDER_PATH)
    print(f"Loaded {len(docs)} documents from JSON files.")
except Exception as e:
    print(f"Error loading documents: {e}")
    raise

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} text chunks.")

# 유효한 텍스트만 필터링
all_splits = [doc for doc in all_splits if doc.page_content.strip()]
if not all_splits:
    raise ValueError("No valid documents found for vector store.")
print(f"Filtered down to {len(all_splits)} valid text chunks.")

# 벡터 스토어 설정
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_store = Chroma(embedding_function=embeddings, persist_directory=VECTOR_STORE_PATH)
vector_store.add_documents(all_splits)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# 프롬프트 정의
prompt_template = """You are a legal assistant specializing in Korean building laws. Answer the user's query based on the provided context.
If the answer cannot be found, respond with "정보를 찾을 수 없습니다."

[법령명]: {law_name}
[문맥]:
{context}

[질문]:
{question}

[답변]:
"""
prompt = PromptTemplate.from_template(prompt_template)

# 상태(State) 정의
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 검색 및 생성 단계 정의
def retrieve(state: State):
    """사용자의 질문을 기반으로 벡터 스토어에서 관련 문서를 검색합니다."""
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    """검색된 문서를 바탕으로 LLM을 사용해 답변을 생성합니다."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    if state["context"]:
        metadata = state["context"][0].metadata
        formatted_prompt = prompt.format(
            question=state["question"],
            context=docs_content,
            law_name=metadata.get("법령명", "알 수 없음"),
        )
        response = llm.invoke({"text": formatted_prompt})
        return {"answer": response.content}
    return {"answer": "정보를 찾을 수 없습니다."}

# LangGraph로 워크플로우 정의
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# 테스트 실행
def run_legal_qa(question):
    """질문을 입력받아 답변을 생성합니다."""
    response = graph.invoke({"question": question})
    return response["answer"]

# 테스트
if __name__ == "__main__":
    question = "다중생활시설에 대한 건축 기준은 무엇인가요?"
    answer = run_legal_qa(question)
    print(f"질문: {question}\n답변: {answer}")
