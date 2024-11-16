import os
import logging
from dotenv import load_dotenv
import requests
import sqlite3

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
LAW_API_KEY = os.getenv("LAW_API_KEY")
BASE_URL = "https://www.law.go.kr/DRF/lawService.do"

# SQLite 데이터베이스 파일 이름
DB_NAME = "law_data.db"

def initialize_db():
    """
    SQLite 데이터베이스 초기화.
    """
    logging.debug("데이터베이스 초기화 시작.")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            law_id TEXT NOT NULL,
            law_name TEXT NOT NULL,
            xml_data TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logging.debug("데이터베이스 초기화 완료.")

def fetch_law_xml(law_name):
    """
    API를 통해 법령 XML 데이터를 가져옵니다.
    """
    logging.info(f"API 요청 시작 - 법령명: {law_name}")
    params = {
        "OC": LAW_API_KEY,
        "target": "law",
        "LM": law_name  # 'law_id' 대신 'law_name'을 LM 파라미터로 사용
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        logging.info(f"API 요청 성공 - 법령명: {law_name}")
        return response.text
    else:
        logging.error(f"API 요청 실패: {response.status_code} - {response.text}")
        raise Exception(f"API 요청 실패: {response.status_code} - {response.text}")

def save_law_to_db(law_id, law_name, xml_data):
    """
    법령 XML 데이터를 데이터베이스에 저장합니다.
    """
    logging.debug(f"데이터베이스 저장 시작 - 법령명: {law_name}")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO laws (law_id, law_name, xml_data)
        VALUES (?, ?, ?)
    """, (law_id, law_name, xml_data))
    conn.commit()
    conn.close()
    logging.debug(f"데이터베이스 저장 완료 - 법령명: {law_name}")

def process_law(law_name):
    """
    법령명으로 XML 데이터를 가져오고 시행령 및 시행규칙 데이터를 포함하여 저장합니다.
    """
    try:
        logging.info(f"법령 처리 시작: {law_name}")
        
        # 기본 법령 데이터 가져오기
        xml_data = fetch_law_xml(law_name)
        save_law_to_db(law_name, f"{law_name} (법령)", xml_data)

        # 시행령 및 시행규칙 추출 후 저장
        related_laws = ["시행령", "시행규칙"]
        for related in related_laws:
            related_name = f"{law_name} {related}"
            logging.info(f"관련 법령 처리 시작: {related_name}")
            xml_data_related = fetch_law_xml(related_name)
            save_law_to_db(related_name, f"{law_name} ({related})", xml_data_related)
            logging.info(f"관련 법령 저장 완료: {related_name}")
    except Exception as e:
        logging.error(f"에러 발생: {law_name} - {e}")

def load_laws_from_file(file_path):
    """
    파일에서 법령명을 로드합니다.
    """
    logging.debug(f"법령 목록 파일 로드: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        laws = [line.strip() for line in file if line.strip().endswith("법")]
    logging.debug(f"법령 목록 로드 완료: {laws}")
    return laws

if __name__ == "__main__":
    # 데이터베이스 초기화
    initialize_db()

    # 법규 목록 로드
    file_path = "법규 목록.txt"
    laws = load_laws_from_file(file_path)

    # 각 법령 처리
    for law_name in laws:
        process_law(law_name)