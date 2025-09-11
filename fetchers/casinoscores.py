# fetchers/casinoscores.py
import re, time, datetime as dt
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

URL = "https://casinoscores.com/funky-time/"  # صفحة النتائج

# خرائط الأيقونات -> القطع
ICON_MAP = {
    "1.png": "1",
    "p.png": "P","l.png":"L","a.png":"A","y.png":"Y",
    "f.png":"F","u.png":"U","n.png":"N","k.png":"K",
    "t.png":"T","i.png":"I","m.png":"M","e.png":"E",
    "bar": "BAR",
    "disco": "DISCO",
    "vip": "VIP",
    "stayinalive": "STAYINALIVE",  # انتبه لكتابة الموقع
}

def _icon_to_segment(src: str) -> str:
    src = src.lower()
    # نلتقط اسم الملف/الأيقونة فقط
    m = re.search(r"funky-time/([^/?#]+)", src)
    key = m.group(1) if m else src
    for k,v in ICON_MAP.items():
        if k in key:
            return v
    return None  # UNKNOWN

def _to_multiplier(txt: str) -> int:
    # أمثلة: "25X", "1X", "3X", "100X"
    m = re.search(r"(\d+)\s*x", txt.strip().lower())
    return int(m.group(1)) if m else 1

def fetch_latest(limit: int = 300) -> pd.DataFrame:
    """يرجع DataFrame بأعمدة ts, segment, multiplier"""
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    # الكروت في الصفحة تحتوي وقت + أيقونة + المضاعف
    cards = soup.select("div.card, li, article, div")  # نأخذ عينة عريضة ثم نرشّح
    for el in cards:
        # وقت الرمية (يحصل أحيانًا داخل سبان/ديف)
        time_el = el.find(string=re.compile(r"\d{1,2}:\d{2}"))
        mult_el = el.find(string=re.compile(r"\d+\s*[xX]"))
        img = el.find("img", src=re.compile(r"funky-time/"))

        if not time_el or not mult_el or not img:
            continue

        ts_str = str(time_el).strip()
        # التاريخ: نضيف تاريخ اليوم (الموقع يعرض الساعة فقط غالبًا)
        today = dt.date.today().strftime("%Y-%m-%d")
        ts = f"{today} {ts_str}"

        seg = _icon_to_segment(img.get("src",""))
        mult = _to_multiplier(str(mult_el))

        if seg:
            rows.append([ts, seg, mult])

    df = pd.DataFrame(rows, columns=["ts", "segment", "multiplier"])
    # أحدث النتائج أولًا في الموقع؛ نخليها تصاعدي زمنيًا
    df = df.drop_duplicates().iloc[:limit].copy()
    df["multiplier"] = df["multiplier"].astype(int)
    # فرز حسب الوقت
    df = df.sort_values("ts").reset_index(drop=True)
    return df
