import os
import csv
import logging
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from get_law_xml import extract_key_xml_data

# 환경 변수 로드
load_dotenv()

# OpenAI 설정
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gpt_model = os.getenv("GPT_MODEL", "GPT_MODEL")
db_name = os.getenv("DB_NAME", "law_data.db")

def extract_law_name_from_question(prompt):
    """
    질문에서 관련 법령명을 추출.
    """
    messages = [
        {"role": "system", "content": "당신은 한국의 법률 전문가입니다."},
        {"role": "user", "content": f"다음 질문에 관련된 법령명을 쉼표로 구분하여 알려주세요. 수식어는 필요 없습니다. 법령면만 알려주세요.: \"{prompt}\""}
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=gpt_model,
        max_tokens=4096,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def search_law_content(law_names, db_name="law_data.db"):
    """
    데이터베이스에서 법령 이름으로 법령 내용을 검색합니다.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        results = []
        for law_name in law_names:
            cursor.execute("""
                SELECT law_name, xml_data 
                FROM laws 
                WHERE law_name LIKE ?
            """, (f"%{law_name}%",))
            rows = cursor.fetchall()

            for row in rows:
                law_name, xml_data = row
                # 핵심 XML 데이터 추출
                key_content = extract_key_xml_data(xml_data)
                results.append({
                    "law_name": law_name,
                    "snippet": key_content
                })

        if not results:
            return "관련 법령 내용을 찾을 수 없습니다."
        return results
    except Exception as e:
        logging.error(f"데이터베이스 쿼리 중 오류 발생: {e}")
        return "데이터베이스 검색 중 오류가 발생했습니다."
    finally:
        conn.close()

def generate_answer_with_gpt(prompt, question, law_content):
    """
    질문과 법령 내용을 기반으로 GPT를 사용해 답변 생성.
    """
    full_prompt = f"""
    {prompt}

    질문: "{question}"

    관련 법령 내용:
    {law_content}
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "당신은 법률 전문가입니다."},
            {"role": "user", "content": full_prompt}
        ],
        model=gpt_model,
        max_tokens=4096,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def process_question(question, prompt):
    """
    질문에 대해 데이터베이스 검색 및 GPT로 답변 생성.
    """
    law_names = extract_law_name_from_question(question).split(", ")
    search_results = search_law_content(law_names)

    if isinstance(search_results, str):
        return search_results

    law_content = "\n\n".join(
        [f"{result['law_name']}:\n{result['snippet']}" for result in search_results]
    )
    return generate_answer_with_gpt(prompt, question, law_content)

if __name__ == "__main__":
    # 프롬프트 파일 읽기
    with open("prompt.txt", "r", encoding="UTF-8") as file:
        prompt_content = file.read().strip()

    # 입력 CSV 파일 읽기
    input_csv = "법규 질문 답변 생성.csv"
    output_csv = "법규 질문 답변 결과.csv"

    with open(input_csv, "r", encoding="UTF-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = [row for row in reader]

    for row in data:
        question = row[0]
        row.append(process_question(question, prompt_content))

    with open(output_csv, "w", newline="", encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header + ["답변"])
        writer.writerows(data)

    print(f"결과가 {output_csv}에 저장되었습니다.")
