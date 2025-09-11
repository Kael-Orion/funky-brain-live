# fetchers/casinoscores.py
# -*- coding: utf-8 -*-
import re
import datetime as dt
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup

URL = "https://casinoscores.com/funky-time/"  # صفحة اللعبة

# خريطة تحويل اسم أيقونة الصورة إلى اسم الخانة
ICON_MAP = {
    "1.png": "1",
    "bar.png": "BAR",
    "disco.png": "DISCO",
    "vip.png": "VIP",
    "stayingalive.png": "STAYINALIVE",
    "stayinalive.png": "STAYINALIVE",  # احتياط لتسمية بديلة

    "p.png": "P", "l.png": "L", "a.png": "A", "y.png": "Y",
    "f.png": "F", "u.png": "U", "n.png": "N", "k.png": "K",
    "t.png": "T", "i.png": "I", "m.png": "M", "e.png": "E",
}

# أنماط استخراج مرنة
RE_ICON = re.compile(r"/funky-time/([^/]+)\.png", re.I)
RE_MULT = re.compile(r"(\d+)\s*[x×X]")          # 25x أو 25×
RE_TIME = re.compile(r"\b(\d{1,2}:\d{2})\b")     # HH:MM

# نصنع سكرابر يشبه كروم (يساعد ضد Cloudflare)
_scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

def _icon_to_segment(src: str) -> str:
    src = (src or "").lower()
    m = RE_ICON.search(src)
    key = (m.group(1) + ".png") if m else src.split("/")[-1]
    return ICON_MAP.get(key, "UNKNOWN")

def _clean_text(s: str) -> str:
    return (s or "").strip().replace("\u200b", "")

def fetch_latest(limit: int = 300) -> pd.DataFrame:
    """
    يجلب آخر الرميات من Casinoscores ويعيد DataFrame بالأعمدة:
    ts (HH:MM), segment, multiplier
    """
    r = _scraper.get(URL, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Fetch blocked or failed. HTTP={r.status_code}")

    soup = BeautifulSoup(r.text, "lxml")

    # نختار عناصر كثيرة لأن الـDOM قد يتغيّر
    candidates = soup.select("li, .result, .list-item, .card, .row, .flex") or []

    rows = []
    for el in candidates:
        # صورة الخانة
        img = el.find("img")
        src = (img.get("src") or img.get("data-src") or "") if img else ""
        seg = _icon_to_segment(src)
        if seg == "UNKNOWN":
            continue

        # نص البطاقة
        text = _clean_text(el.get_text(" ", strip=True))
        if not text:
            text = _clean_text(" ".join(t.get_text(" ", strip=True) for t in el.find_all(True)))

        # المضاعِف (إن وُجد)
        m_mult = RE_MULT.search(text)
        mult = int(m_mult.group(1)) if m_mult else (1 if seg == "1" else None)

        # الوقت
        m_time = RE_TIME.search(text)
        if m_time:
            today = dt.datetime.utcnow().date()
            hh, mm = map(int, m_time.group(1).split(":"))
            ts = dt.datetime(today.year, today.month, today.day, hh, mm).strftime("%H:%M")
        else:
            ts = None

        rows.append({"ts": ts, "segment": seg, "multiplier": mult})
        if len(rows) >= limit:
            break

    df = pd.DataFrame(rows).dropna(subset=["segment"]).reset_index(drop=True)

    # ضبط القيم الافتراضية
    letters = list("PLAYFUNKTIME")
    mask_letters = df["segment"].isin(letters)
    df.loc[mask_letters & df["multiplier"].isna(), "multiplier"] = 25
    df.loc[(df["segment"] == "1") & df["multiplier"].isna(), "multiplier"] = 1

    return df
