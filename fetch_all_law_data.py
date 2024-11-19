import os
import requests
import xml.etree.ElementTree as ET
import json
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
API_KEY = os.getenv("LAW_API_KEY")
if not API_KEY:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

def read_law_list(file_path):
    """법령목록.txt 파일에서 법령명을 읽어 리스트로 반환."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def fetch_all_law_data(law_name):
    """법, 시행령, 시행규칙 데이터를 한꺼번에 가져옵니다."""
    targets = ["law", "ordinance", "rule"]
    combined_data = {"법령명": law_name, "법": None, "시행령": None, "시행규칙": None}

    for target in targets:
        url = f"https://www.law.go.kr/DRF/lawService.do"
        params = {
            "OC": API_KEY,
            "target": target,
            "LM": law_name,
            "type": "XML"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            combined_data[target] = response.content
        else:
            print(f"Failed to fetch {target} data for {law_name}. Status code: {response.status_code}")
    
    return combined_data

def parse_law_xml(xml_data):
    """법령 XML 데이터를 파싱하여 JSON 데이터로 변환."""
    root = ET.fromstring(xml_data)

    # 법령 기본 정보 추출
    basic_info = root.find("기본정보")
    if basic_info is None:
        print("기본정보를 찾을 수 없습니다.")
        return None

    law_info = {
        "법령ID": basic_info.findtext("법령ID"),
        "법령명": basic_info.findtext("법령명_한글"),
        "공포일자": basic_info.findtext("공포일자"),
        "법종구분": basic_info.find("법종구분").text if basic_info.find("법종구분") is not None else None,
        "소관부처": basic_info.find("소관부처").text if basic_info.find("소관부처") is not None else None,
        "시행일자": basic_info.findtext("시행일자"),
    }

    # 조문 정보 추출
    articles = []
    for article in root.findall("조문/조문단위"):
        article_data = {
            "조문번호": article.findtext("조문번호"),
            "조문제목": article.findtext("조문제목"),
            "조문내용": article.findtext("조문내용"),
            "시행일자": article.findtext("조문시행일자"),
            "항목록": []
        }

        # 항 및 호 정보 추출
        for paragraph in article.findall("항"):
            paragraph_data = {
                "항번호": paragraph.findtext("항번호"),
                "항내용": paragraph.findtext("항내용"),
                "호목록": []
            }

            # 호와 목 정보 추출
            for sub_item in paragraph.findall("호"):
                sub_item_data = {
                    "호번호": sub_item.findtext("호번호"),
                    "호내용": sub_item.findtext("호내용"),
                    "목목록": []
                }
                for sub_sub_item in sub_item.findall("목"):
                    sub_sub_item_data = {
                        "목번호": sub_sub_item.findtext("목번호"),
                        "목내용": sub_sub_item.findtext("목내용"),
                    }
                    sub_item_data["목목록"].append(sub_sub_item_data)
                paragraph_data["호목록"].append(sub_item_data)
            article_data["항목록"].append(paragraph_data)

        articles.append(article_data)

    return {"법령정보": law_info, "조문목록": articles}
