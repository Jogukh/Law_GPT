import os
import json
import logging
import requests
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 키 가져오기
API_KEY = os.getenv("LAW_API_KEY")

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_law_structure(law_name):
    """
    법령 체계도를 JSON 형식으로 가져옵니다.
    """
    url = f"https://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": API_KEY,
        "target": "lsStmd",
        "type": "JSON",
        "LM": law_name
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, str):  # 문자열인 경우 JSON으로 디코딩
            data = json.loads(data)

        # 디버그 메시지 추가
        logging.debug(f"{law_name} 체계도 데이터 타입: {type(data)}")

        return data
    except Exception as e:
        logging.error(f"{law_name} 체계도 가져오기 실패: {e}")
        return None

def extract_all_sub_laws(law_structure, exclusion_keywords=None):
    """
    JSON 구조를 재귀적으로 탐색하여 모든 하위 법령 및 행정규칙(훈령, 예규, 고시)을 추출합니다.
    """
    if exclusion_keywords is None:
        exclusion_keywords = ["조례"]  # 제외할 키워드 (기본값)

    sub_laws = []

    def traverse(node):
        try:
            # 노드가 딕셔너리인 경우
            if isinstance(node, dict):
                # 법종구분-content 확인
                if "기본정보" in node:
                    law_type = node["기본정보"].get("법종구분", {}).get("content", "")
                    law_name = node["기본정보"].get("법령명", "") or node["기본정보"].get("행정규칙명", "")
                    if law_name and law_type not in exclusion_keywords:
                        sub_laws.append((law_name, law_type))  # 이름과 종류를 함께 저장

                # 하위 항목 탐색
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        traverse(value)

            # 노드가 리스트인 경우
            elif isinstance(node, list):
                for item in node:
                    traverse(item)
        except Exception as e:
            logging.error(f"노드 탐색 중 오류 발생: {e}")

    # 탐색 시작
    if "법령체계도" in law_structure and "상하위법" in law_structure["법령체계도"]:
        traverse(law_structure["법령체계도"]["상하위법"])
    else:
        logging.warning("법령체계도 또는 상하위법 필드가 없습니다.")

    return list(set(sub_laws))  # 중복 제거 후 반환

def fetch_law_content(law_name, law_type):
    """
    법령 본문 내용을 JSON 형식으로 가져옵니다.
    하위법 종류에 따라 API 호출의 target 값을 다르게 설정합니다.
    """
    # target 값 설정
    target_mapping = {
        "법률": "law",
        "대통령령": "law",
        "국토교통부령": "law",
        "훈령": "admrul",
        "예규": "admrul",
        "고시": "admrul"
    }
    target = target_mapping.get(law_type, "law")  # 기본값은 "law"

    url = f"https://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": API_KEY,
        "target": target,
        "type": "JSON",
        "LM": law_name
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # 디버깅 로그 추가
        logging.debug(f"'{law_name}' ({law_type}) API 응답 완료")

        return data
    except Exception as e:
        logging.error(f"'{law_name}' ({law_type}) 본문 가져오기 실패: {e}")
        return None

def save_json_to_folder(folder_name, file_name, data):
    """
    데이터를 지정된 폴더에 JSON 파일로 저장합니다.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, f"{file_name}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"파일 저장 완료: {file_path}")
    except Exception as e:
        logging.error(f"파일 저장 실패: {file_path}, 이유: {e}")


def process_law_structure_and_content(law_name):
    """
    법령 체계도와 모든 하위 법령 본문을 처리하고 저장합니다.
    """
    logging.info(f"법령 처리 시작: {law_name}")
    # 법령 체계도 가져오기
    law_structure = fetch_law_structure(law_name)
    if not law_structure:
        logging.error(f"법령 체계도 가져오기 실패: {law_name}")
        return

    # 체계도 저장
    save_json_to_folder(law_name, "법령체계도", law_structure)

    # 하위 법령 추출
    sub_laws = extract_all_sub_laws(law_structure)
    logging.info(f"추출된 하위 법령: {sub_laws}")
    for sub_law in sub_laws:
        # 하위 법령 본문 가져오기
        content = fetch_law_content(sub_law)
        if content:
            save_json_to_folder(law_name, sub_law, content)

