import os
import logging
import sqlite3
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

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
DB_NAME = "law_data.db"

def initialize_db():
    """
    SQLite 데이터베이스 초기화.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            law_id TEXT NOT NULL,
            law_name TEXT NOT NULL,
            promulgation_date TEXT,
            enforcement_date TEXT,
            issuing_agency TEXT,
            xml_data TEXT
        )
    """)

    conn.commit()
    conn.close()
    logging.debug("데이터베이스 초기화 완료.")

def fetch_law_xml(law_name):
    """
    API를 통해 법령 XML 데이터를 가져옵니다.
    """
    logging.info(f"API 요청 시작: {law_name}")
    params = {
        "OC": LAW_API_KEY,
        "target": "law",
        "LM": law_name
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        logging.info(f"API 요청 성공: {law_name}")
        return response.text
    else:
        logging.error(f"API 요청 실패: {response.status_code} - {response.text}")
        raise Exception(f"API 요청 실패: {response.status_code} - {response.text}")

def save_law_to_db(law_name, xml_data):
    """
    XML 데이터를 데이터베이스에 저장합니다.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 기본정보 추출
    root = ET.ElementTree(ET.fromstring(xml_data)).getroot()
    basic_info = root.find("기본정보")

    law_id = basic_info.find("법령ID").text if basic_info.find("법령ID") is not None else "N/A"
    promulgation_date = basic_info.find("공포일자").text if basic_info.find("공포일자") is not None else "N/A"
    enforcement_date = basic_info.find("시행일자").text if basic_info.find("시행일자") is not None else "N/A"
    issuing_agency = basic_info.find("소관부처").text if basic_info.find("소관부처") is not None else "N/A"

    cursor.execute("""
        INSERT INTO laws (law_id, law_name, promulgation_date, enforcement_date, issuing_agency, xml_data)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (law_id, law_name, promulgation_date, enforcement_date, issuing_agency, xml_data))

    conn.commit()
    conn.close()
    logging.info(f"법령 저장 완료: {law_name}")

def get_related_law_names(law_name):
    """
    법령명에 대해 시행령 및 시행규칙 이름 생성.
    """
    return [
        law_name,                # 법
        f"{law_name} 시행령",    # 시행령
        f"{law_name} 시행규칙"   # 시행규칙
    ]

def process_law_list(file_path):
    """
    법령 목록 파일을 읽어 모든 법령과 관련된 시행령 및 시행규칙을 처리합니다.
    """
    logging.info(f"법령 목록 처리 시작: {file_path}")

    # 법령 목록 파일 읽기
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            law_names = [line.strip() for line in file if line.strip()]
    except Exception as e:
        logging.error(f"법령 목록 파일 읽기 실패: {e}")
        return

    # 법령 및 관련 법령 처리
    for law_name in law_names:
        related_laws = get_related_law_names(law_name)
        for related_name in related_laws:
            try:
                logging.info(f"법령 처리 중: {related_name}")
                xml_data = fetch_law_xml(related_name)
                save_law_to_db(related_name, xml_data)
            except Exception as e:
                logging.error(f"법령 처리 실패: {related_name} - {e}")

    logging.info("법령 목록 처리 완료.")

def extract_key_xml_data(xml_data):
    """
    XML 데이터에서 조문시행일자, 조문내용, 호내용, 목내용을 추출.
    """
    try:
        root = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        extracted_data = []

        for article in root.findall("조문/조문단위"):
            article_content = article.find("조문내용").text if article.find("조문내용") is not None else "조문내용 없음"
            article_date = article.find("조문시행일자").text if article.find("조문시행일자") is not None else "시행일자 없음"
            extracted_data.append(f"조문시행일자: {article_date}\n조문내용: {article_content}")

            for clause in article.findall("항"):
                for item in clause.findall("호"):
                    item_content = item.find("호내용").text if item.find("호내용") is not None else "호내용 없음"
                    extracted_data.append(f"호내용: {item_content}")

                for sub_item in clause.findall("목"):
                    sub_item_content = sub_item.text if sub_item is not None else "목내용 없음"
                    extracted_data.append(f"목내용: {sub_item_content}")

        return "\n\n".join(extracted_data[:5])  # 최대 5개 데이터 반환
    except Exception as e:
        logging.error(f"XML 파싱 중 오류 발생: {e}")
        return "XML 데이터 파싱 중 오류 발생."


if __name__ == "__main__":
    # 데이터베이스 초기화
    initialize_db()

    # 법령 목록 파일 경로
    law_list_file = "법령목록.txt"

    # 법령 목록 처리
    process_law_list(law_list_file)
