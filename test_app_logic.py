import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("KWATER_API_KEY", "")

def fetch_kwater_data(key, code):
    url = 'http://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist'
    now = datetime.now()
    stdt = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    eddt = now.strftime("%Y-%m-%d")
    import urllib.parse
    decoded_key = urllib.parse.unquote(key)
    params = {'serviceKey': decoded_key, 'pageNo': '1', 'numOfRows': '24', 'damcode': code, 'stdt': stdt, 'eddt': eddt, '_type': 'json'}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    items = r.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
    if not items: raise ValueError(f"No items in response: {r.text[:200]}")
    df = pd.DataFrame(items)
    df.rename(columns={'obsrdt': '시간', 'inflowqy': '유입량(m³/s)', 'lowlevel': '저수위(EL.m)', 'rf': '강우량(mm)', 'totdcwtrqy': '총방류량(m³/s)'}, inplace=True)
    for col in ['유입량(m³/s)', '저수위(EL.m)', '강우량(mm)', '총방류량(m³/s)']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    year = str(now.year)
    clean_time = df['시간'].astype(str).str.extract(r'(\d{2}-\d{2}\s*\d{2})')[0].str.replace(r'\s+', ' ', regex=True)
    df['시간'] = pd.to_datetime(year + "-" + clean_time, format="%Y-%m-%d %H")
    return df.iloc[::-1].reset_index(drop=True)

def fetch_kma_weather(key, nx, ny):
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
            
    import urllib.parse
    decoded_key = urllib.parse.unquote(key)
    params = {'serviceKey': decoded_key, 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON', 'base_date': base_date, 'base_time': base_time, 'nx': nx, 'ny': ny}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    items = r.json().get('response', {}).get('body', {}).get('items', {}).get('item', [])
    if not items: raise ValueError(f"No KMA data in response: {r.text[:200]}")
    
    w_data = []
    for item in items:
        if item['category'] == 'PCP':
            val = 0.0 if item['fcstValue'] == "강수없음" else float(item['fcstValue'].replace("mm", "").replace("범위", "").split("~")[0].strip())
            dt = pd.to_datetime(f"{item['fcstDate']} {item['fcstTime'][:2]}:00")
            w_data.append({"시간": dt, "예상강수량(mm)": val})
            
    df = pd.DataFrame(w_data).sort_values("시간").reset_index(drop=True)
    return df

print("Fetching K-water data...")
try:
    df1 = fetch_kwater_data(api_key, "1012110")
    print(df1.head())
except Exception as e:
    print("K-water error:", e)

print("\nFetching KMA data...")
try:
    df2 = fetch_kma_weather(api_key, 76, 114)
    print(df2.head())
except Exception as e:
    print("KMA error:", e)

