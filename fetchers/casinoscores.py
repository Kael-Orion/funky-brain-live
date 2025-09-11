import re, datetime as dt
import requests
from bs4 import BeautifulSoup
import pandas as pd

# رابط الصفحة
URL = "https://casinoscores.com/funky-time/"

# هيدر عشان نتجنب الحجب من السيرفر
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}

# خريطة الأيقونات للأحرف والبونصات
ICON_MAP = {
    "1.png": "1", "p.png": "P", "l.png": "L", "a.png": "A", "y.png": "Y",
    "f.png": "F", "u.png": "U", "n.png": "N", "k.png": "K",
    "t.png": "T", "i.png": "I", "m.png": "M", "e.png": "E",
    "bar": "BAR", "disco": "DISCO", "vip": "VIP", "stayinalive": "STAYINALIVE"
}

def _icon_to_segment(src: str) -> str:
    src = src.lower()
    m = re.search(r"funky-time/([^/?#]+)", src)
    key = m.group(1) if m else src
    for k, v in ICON_MAP.items():
        if k in key:
            return v
    return None

def _to_multiplier(txt: str) -> int:
    # يلتقط مثلا "25X" أو "3x"
    m = re.search(r"(\d+)\s*x", txt.strip().lower())
    return int(m.group(1)) if m else 1

def fetch_latest(limit: int = 300) -> pd.DataFrame:
    """يسحب آخر الرميات من CasinoScores"""
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    # نبحث في العناصر اللي فيها الوقت + صورة + مضاعف
    cards = soup.find_all("li")
    for el in cards:
        time_el = el.find(string=re.compile(r"\d{1,2}:\d{2}"))
        mult_el = el.find(string=re.compile(r"\d+\s*[xX]"))
        img = el.find("img", src=re.compile(r"funky-time/"))

        if not time_el or not mult_el or not img:
            continue

        today = dt.date.today().strftime("%Y-%m-%d")
        ts = f"{today} {time_el.strip()}"

        seg = _icon_to_segment(img.get("src", ""))
        mult = _to_multiplier(str(mult_el))

        if seg:
            rows.append([ts, seg, mult])

    df = pd.DataFrame(rows, columns=["ts", "segment", "multiplier"])
    df = df.drop_duplicates().iloc[:limit]
    df["multiplier"] = df["multiplier"].astype(int)
    return df.sort_values("ts").reset_index(drop=True)
