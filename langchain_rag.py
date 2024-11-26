from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# 저장된 벡터 스토어 로드
vectorstore_path = "law_vector"
vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
print("Vector store loaded successfully.")

# 프롬프트 파일에서 내용 불러오기
with open("prompt.txt", "r", encoding="utf-8") as file:
    prompt_template = file.read()

# 예제 쿼리
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
query = "안전진단이 필요한 경우는 무엇인가요?"

# 유사 문서 검색
retrieved_docs = vectorstore.similarity_search(query, k=50)

# 검색된 문서를 디버깅 출력
print("Retrieved Docs:")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Document {i}:\n{doc.page_content}\n")

# 검색된 문서를 프롬프트로 변환
context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])  # 상위 5개 문서만 사용
full_prompt = prompt_template.format(context=context, question=query)

# 프롬프트 디버깅 출력
print("Full Prompt:")
print(full_prompt)

# LLM 호출
response = llm.predict(full_prompt)
print("Response:", response)
