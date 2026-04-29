import requests
import os
from dotenv import load_dotenv

load_dotenv()
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
