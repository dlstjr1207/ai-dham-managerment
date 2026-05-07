import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
api_key = os.getenv("KWATER_API_KEY", "")

# 1. K-water API
print("Testing K-water API...")
try:
    url = 'http://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist'
    now = datetime.now()
    stdt = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    eddt = now.strftime("%Y-%m-%d")
    # To prevent double encoding, sometimes we need to unquote the key
    import urllib.parse
    decoded_key = urllib.parse.unquote(api_key)
    
    params = {'serviceKey': decoded_key, 'pageNo': '1', 'numOfRows': '24', 'damcode': '1012110', 'stdt': stdt, 'eddt': eddt, '_type': 'json'}
    r = requests.get(url, params=params)
    print(f"Status Code: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"K-water Exception: {e}")

# 2. KMA API
print("\nTesting KMA API...")
try:
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    now = datetime.now()
    times = [2, 5, 8, 11, 14, 17, 20, 23]
    base_time = "2300"
    base_date = (now - timedelta(days=1)).strftime("%Y%m%d")
    for t in reversed(times):
        if now.hour >= t:
            base_time = f"{t:02d}00"
            base_date = now.strftime("%Y%m%d")
            break
            
    params = {'serviceKey': decoded_key, 'pageNo': '1', 'numOfRows': '10', 'dataType': 'JSON', 'base_date': base_date, 'base_time': base_time, 'nx': 76, 'ny': 114}
    r = requests.get(url, params=params)
    print(f"Status Code: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"KMA Exception: {e}")
