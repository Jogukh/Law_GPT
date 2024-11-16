import os
import csv
import random
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# OpenAI Client 초기화
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gpt_model = os.getenv("GPT_MODEL", "gpt-4")
db_name = os.getenv("DB_NAME", "law_data.db")

def extract_law_name_from_question(prompt):
    """
    OpenAI GPT를 사용하여 질문에서 관련 법령명을 추출.
    """
    messages = [
        {"role": "system", "content": "당신은 한국의 법률 전문가입니다."},
        {"role": "user", "content": f"다음 질문에 관련된 법령명을 쉼표로 구분하여 알려주세요, 다른 수식어 없이 글머리 기호도 필요 없이 그냥 쉼표로 구분해서 대답 하세요 : \"{prompt}\""}
    ]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=gpt_model,
            max_tokens=4096,
            temperature=0.1,
        )
        # 반환된 객체에서 명시적으로 content를 추출
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return None


def search_law_content(question, db_name=db_name):
    """
    질문에서 관련된 법령명을 추출하고, 데이터베이스에서 해당 법령 내용을 검색.
    """
    law_names = extract_law_name_from_question(question)
    if not law_names:
        return "관련 법령명을 추출하지 못했습니다."

    law_name_list = [name.strip() for name in law_names.split(",")]
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        results = []
        for law_name in law_name_list:
            cursor.execute("SELECT law_name, xml_content FROM law_xml WHERE law_name LIKE ?", (f"%{law_name}%",))
            rows = cursor.fetchall()

            if not rows:
                continue

            for row in rows:
                law_name, xml_content = row
                results.append({"law_name": law_name, "snippet": xml_content[:500]})

        if not results:
            return "관련 법령 내용을 찾을 수 없습니다."

        return results
    except Exception as e:
        print(f"데이터베이스 쿼리 중 오류 발생: {e}")
        return "데이터베이스 검색 중 오류가 발생했습니다."
    finally:
        conn.close()

def generate_answer_with_gpt(prompt, question, law_content):
    """
    OpenAI GPT를 사용하여 질문과 법령 내용을 기반으로 답변 생성.
    """
    full_prompt = f"""
    {prompt}

    질문: "{question}"

    다음은 관련 법령 내용입니다:
    {law_content}
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "당신은 법률 전문가입니다."},
                {"role": "user", "content": full_prompt}
            ],
            model=gpt_model,
            max_tokens=4096,
            temperature=0.1,
        )
        # 반환된 객체에서 명시적으로 content를 추출
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return "GPT 응답 생성 중 오류 발생."

def process_question(question, prompt):
    """
    질문에 대해 데이터베이스 검색 및 GPT로 답변 생성.
    """
    search_results = search_law_content(question)
    if not isinstance(search_results, list):  # 에러 메시지 반환
        return search_results

    law_content = "\n\n".join(
        [f"{result['law_name']}:\n{result['snippet']}" for result in search_results]
    )
    return generate_answer_with_gpt(prompt, question, law_content)


if __name__ == "__main__":
    # 프롬프트 파일 읽기
    with open("prompt.txt", "r", encoding="UTF-8") as file:
        prompt_content = file.read().strip()

    # 입력 CSV 파일 경로 및 출력 CSV 파일 경로
    input_csv = "법규 질문 답변 생성.csv"
    output_csv = "법규 질문 답변 결과.csv"

    # 입력 CSV 파일 읽기
    with open(input_csv, "r", encoding="UTF-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 헤더 읽기
        data = [row for row in reader]

    # 질문이 없는 경우 예외 처리
    if not data:
        print("질문이 없습니다. CSV 파일을 확인하세요.")
    else:
        print(f"총 {len(data)}개의 질문에 대해 처리합니다.")

    # 각 질문 처리
    for index, row in enumerate(data):
        question = row[0]  # 질문 내용
        if len(row) < 2:
            row.append("")  # 결과 열 추가

        print(f"Processing ({index + 1}/{len(data)}): {question}")
        result = process_question(question, prompt_content)
        row[1] = result  # 결과 저장

    # 출력 CSV 파일에 저장
    with open(output_csv, "w", newline="", encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 헤더 작성
        writer.writerows(data)  # 데이터 작성

    print(f"결과가 {output_csv}에 저장되었습니다.")