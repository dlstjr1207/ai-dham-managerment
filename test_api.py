import requests
import os
from dotenv import load_dotenv

# 실행 파일 위치를 기준으로 현재 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("KWATER_API_KEY", "")

params = {
    'serviceKey': api_key,
    'pageNo': '1',
    'numOfRows': '50',
    '_type': 'json'
}
r = requests.get(url, params=params)
print(r.status_code)
try:
    print(r.json())
except:
    print(r.text[:500])
