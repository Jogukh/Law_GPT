#%%
import csv
import random
from v1_db구축방식.main import process_question, extract_law_name_from_question  # 필요한 함수 가져오기

def random_test(input_csv, prompt_file, test_count=2):
    """
    랜덤으로 질문을 선택하여 테스트. 디버깅 정보 출력 포함.
    """
    print("[INFO] 랜덤 테스트 시작")

    # 프롬프트 파일 읽기
    try:
        print("[DEBUG] 프롬프트 파일 읽기 시작")
        with open(prompt_file, "r", encoding="UTF-8") as file:
            prompt_content = file.read().strip()
        print("[DEBUG] 프롬프트 파일 읽기 완료")
    except Exception as e:
        print(f"[ERROR] 프롬프트 파일 읽기 실패: {e}")
        return

    # 입력 CSV 파일 읽기
    try:
        print("[DEBUG] 입력 CSV 파일 읽기 시작")
        with open(input_csv, "r", encoding="UTF-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # 헤더 읽기
            questions = [row[0] for row in reader]  # 첫 번째 열(질문)만 가져옴
        print("[DEBUG] 입력 CSV 파일 읽기 완료")
    except Exception as e:
        print(f"[ERROR] 입력 CSV 파일 읽기 실패: {e}")
        return

    # 질문이 없을 경우 처리
    if not questions:
        print("[WARNING] CSV 파일에 질문이 없습니다.")
        return

    # 랜덤으로 테스트할 질문 선택
    try:
        print("[DEBUG] 랜덤 질문 선택 시작")
        selected_questions = random.sample(questions, min(test_count, len(questions)))
        print(f"[DEBUG] 선택된 질문: {selected_questions}")
    except Exception as e:
        print(f"[ERROR] 랜덤 질문 선택 실패: {e}")
        return

    # 각 질문에 대해 테스트 실행
    for index, question in enumerate(selected_questions, start=1):
        print(f"\n[INFO] 테스트 ({index}/{test_count}) 시작: {question}")
        
        # 추출된 법령명 출력
        try:
            law_names = extract_law_name_from_question(question)
            print(f"[DEBUG] 추출된 법령명: {law_names}")
        except Exception as e:
            print(f"[ERROR] 법령명 추출 실패 - 질문: {question}, 오류: {e}")
            continue
        
        # GPT 응답 생성
        try:
            result = process_question(question, prompt_content)
            print(f"[DEBUG] GPT 응답 생성 완료: {result[:200]}...")  # 응답 앞부분만 출력
            print(f"답변: {result}")
        except Exception as e:
            print(f"[ERROR] 테스트 실패 - 질문: {question}, 오류: {e}")

    print("[INFO] 랜덤 테스트 종료")

# 실행 코드
if __name__ == "__main__":
    input_csv = "법규 질문 답변 생성.csv"
    prompt_file = "prompt.txt"

    # 랜덤 테스트 실행
    random_test(input_csv, prompt_file, test_count=2)

# %%
