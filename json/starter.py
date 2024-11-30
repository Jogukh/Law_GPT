import os
import logging
from main import fetch_law_structure, extract_all_sub_laws, fetch_law_content, save_json_to_folder

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_PATH = "/Users/joguk/Library/CloudStorage/OneDrive-우미건설/01. 문서/★ LLM 도입 프로젝트/자동화/법령 json"

def create_directory(base_path, folder_name):
    """
    폴더 생성. 이미 존재하면 생성하지 않음.
    """
    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def process_single_law(law_name):
    """
    단일 법령의 체계도 및 하위법 본문 처리.
    """
    logging.info(f"'{law_name}' 처리 시작")
    try:
        # 체계도 가져오기
        law_structure = fetch_law_structure(law_name)
        if not law_structure:
            logging.warning(f"'{law_name}'의 체계도를 가져오지 못했습니다.")
            return

        # 저장 디렉토리 생성
        law_directory = create_directory(BASE_PATH, law_name)

        # 체계도 저장
        save_json_to_folder(law_directory, "법령체계도", law_structure)

        # 하위법 추출
        sub_laws = extract_all_sub_laws(law_structure)
        logging.info(f"'{law_name}'에서 {len(sub_laws)}개의 하위법이 발견되었습니다.")

        # 하위법 본문 처리
        for sub_law, law_type in sub_laws:
            logging.info(f"하위법 '{sub_law}' (종류: {law_type}) 본문 가져오기 시작")
            content = fetch_law_content(sub_law, law_type)  # 하위법 이름과 종류 전달
            if content:
                save_json_to_folder(law_directory, sub_law, {"본문": content})
                logging.info(f"하위법 '{sub_law}' 본문 저장 완료")
            else:
                logging.warning(f"하위법 '{sub_law}' (종류: {law_type}) 본문 데이터가 없습니다.")

        logging.info(f"'{law_name}' 처리 완료")
    except Exception as e:
        logging.error(f"'{law_name}' 처리 중 오류 발생: {e}")

def process_all_laws(file_path):
    """
    법령목록 파일을 읽고 모든 법령 처리.
    """
    if not os.path.exists(file_path):
        logging.error(f"법령목록 파일이 존재하지 않습니다: {file_path}")
        return

    try:
        # 법령목록 읽기
        with open(file_path, "r", encoding="utf-8") as file:
            law_names = [line.strip() for line in file if line.strip()]

        logging.info(f"총 {len(law_names)}개의 법령을 처리합니다.")
        for law_name in law_names:
            process_single_law(law_name)

        logging.info("모든 법령 처리 완료")
    except Exception as e:
        logging.error(f"법령목록 파일 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    logging.info("=== 법령 처리 작업 시작 ===")
    process_all_laws("법령목록.txt")
    logging.info("=== 법령 처리 작업 종료 ===")
