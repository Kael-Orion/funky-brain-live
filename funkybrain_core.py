# -*- coding: utf-8 -*-
"""
Core helpers for Funky Brain LIVE
- normalize_df: يحوّل أي CSV شائع إلى شكل موحّد: [ts, segment, multiplier]
- compute_probs: يحسب احتمالات الظهور المتوقعة في 10 و 15 رمية
- board_matrix: يجهّز مصفوفة للعرض اللوحي
"""

import re
import math
import pandas as pd

# الخرائط الممكنة لاستخراج الـ segment من صور casinoscores (cloudinary)
ICON_MAP = {
    # letters
    "p": "P", "l": "L", "a": "A", "y": "Y",
    "f": "F", "u": "U", "n": "N", "k": "K",
    "t": "T", "i": "I", "m": "M", "e": "E",
    # specials
    "1": "1",
    "bar": "BAR",
    "disco": "DISCO",
    "vip": "VIP",
    "stayin": "STAYINALIVE",  # stayin-alive / stayinalive
}

ALL_SEGMENTS = [
    "P","L","A","Y",
    "F","U","N","K",
    "T","I","M","E",
    "1","BAR","DISCO","VIP","STAYINALIVE"
]

def _guess_segment_from_text(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s_low = s.lower()
    # الترتيب مهم: حاول الحالات الخاصة أولاً
    if "vip" in s_low:
        return "VIP"
    if "disco" in s_low:
        return "DISCO"
    if "bar" in s_low:
        return "BAR"
    if "stay" in s_low or "stayin" in s_low:
        return "STAYINALIVE"
    if "number1" in s_low or s_low.strip() in {"1", "one"}:
        return "1"
    # حرف منفرد
    m = re.search(r"/([plafunktime])\.png", s_low)  # من مسار الصور
    if m:
        return m.group(1).upper()
    # حرف مكتوب نصياً مثل LetterK أو K فقط
    m2 = re.search(r"letter\s*([plafunktime])", s_low)
    if m2:
        return m2.group(1).upper()
    if len(s_low.strip()) == 1 and s_low.strip() in "playfunktime":
        return s_low.strip().upper()
    return None


def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    يحاول اكتشاف الأعمدة تلقائياً ويُرجع DataFrame بالأعمدة:
    ts (string/datetime-like), segment (one of ALL_SEGMENTS), multiplier (int)
    يدعم صيغ casinoscores الشائعة (روابط صور) وصيغنا اليدوية السابقة.
    """
    df = raw.copy()

    # --------------------------
    # 1) أعمدة التوقيت
    # --------------------------
    ts_col = None
    for c in df.columns:
        c_low = str(c).lower()
        if c_low in {"ts", "time", "datetime", "date", "heure"}:
            ts_col = c
            break
    if ts_col is None:
        # لو مفيش وقت، اصنع واحداً على التسلسل
        df["ts"] = range(len(df), 0, -1)
        ts_col = "ts"

    # --------------------------
    # 2) أعمدة تحديد الخانة
    # --------------------------
    seg_col_candidates = [c for c in df.columns if str(c).lower() in {"segment", "tile", "result", "icon", "img", "image"}]
    seg_col = seg_col_candidates[0] if seg_col_candidates else None

    if seg_col is None:
        # لو ما لقيناش، جرّب نقرأ من أول عمود نصّي
        for c in df.columns:
            if df[c].dtype == "object":
                seg_col = c
                break

    # استخرج segment
    def extract_segment(val):
        # إن كان أصلاً قيمة صالحة
        if isinstance(val, str):
            v = val.strip().upper()
            if v in ALL_SEGMENTS:
                return v
        # جرّب من النص/الرابط
        s = str(val)
        # Cloudinary path مثل .../funky-time/k.png
        m = re.search(r"funky[-_]?time/([a-z0-9\-]+)\.png", s.lower())
        if m:
            key = m.group(1)
            # مفاتيح خاصة
            if key in {"1", "one"}:
                return "1"
            if "bar" in key:
                return "BAR"
            if "disco" in key and "vip" in key:
                return "VIP"  # أحياناً vipdisco*
            if "vip" in key:
                return "VIP"
            if "disco" in key:
                return "DISCO"
            if "stay" in key:
                return "STAYINALIVE"
            if key in ICON_MAP:
                return ICON_MAP[key].upper()
        # نصوص مثل LetterK / Number1 / Bar
        g = _guess_segment_from_text(s)
        if g:
            return g
        return None

    df["segment"] = df[seg_col].apply(extract_segment) if seg_col else None

    # --------------------------
    # 3) أعمدة المضاعف (multiplier)
    # --------------------------
    mult_col = None
    for c in df.columns:
        c_low = str(c).lower()
        if c_low in {"multiplier", "multi", "x", "pay", "payout"}:
            mult_col = c
            break
    if mult_col is None:
        # حاول استخراجه من نص مثل "25X" أو "x25"
        maybe_text_col = None
        for c in df.columns:
            if df[c].dtype == "object":
                maybe_text_col = c
                break

        def extract_mult(val):
            if pd.isna(val): 
                return 1
            s = str(val).upper()
            m = re.search(r"(\d+)\s*[X×]", s)
            if m:
                return int(m.group(1))
            # أحياناً أرقام نقية
            m2 = re.search(r"\d+", s)
            if m2:
                return int(m2.group(0))
            return 1

        if maybe_text_col:
            df["multiplier"] = df[maybe_text_col].apply(extract_mult)
        else:
            df["multiplier"] = 1
    else:
        # نظّف إلى أرقام
        def to_int(x):
            try:
                s = str(x).upper()
                s = s.replace("X", "").replace("×", "")
                return int(float(s))
            except Exception:
                return 1
        df["multiplier"] = df[mult_col].apply(to_int)

    # صفّي الصفوف اللي ما قدرناش نحدّد segment لها
    df = df[df["segment"].isin(ALL_SEGMENTS)].copy()
    df = df[["ts", "segment", "multiplier"]].reset_index(drop=True)
    return df


def _prob_ge1_in_n(p: float, n: int) -> float:
    """احتمال ظهور الخانة مرة واحدة على الأقل خلال n رميات."""
    try:
        return 1.0 - (1.0 - p) ** n
    except Exception:
        return 0.0


def compute_probs(df: pd.DataFrame, window: int = 100):
    """
    يحسب:
      - P(next) لكل خانة = تكرارها / مجموع آخر window
      - Exp in 10, 15
      - P(≥1 in 10), P(≥1 in 15)
    """
    if df.empty:
        tiles = pd.DataFrame(columns=["Tile","P(next)","Exp in 10","P(≥1 in 10)","Exp in 15","P(≥1 in 15)"])
        return tiles, pd.DataFrame(), window

    # خُذ آخر window صف
    last = df.tail(window).copy()
    total = len(last)

    counts = last["segment"].value_counts().reindex(ALL_SEGMENTS, fill_value=0)
    probs = counts / max(total, 1)

    tiles = pd.DataFrame({
        "Tile": counts.index,
        "P(next)": probs.values,
        "Exp in 10": (probs * 10).round(2),
        "P(≥1 in 10)": probs.apply(lambda p: _prob_ge1_in_n(p, 10)).values,
        "Exp in 15": (probs * 15).round(2),
        "P(≥1 in 15)": probs.apply(lambda p: _prob_ge1_in_n(p, 15)).values,
    })

    # ترتيب بسيط: الأكثر احتمالاً أولاً
    tiles = tiles.sort_values("P(next)", ascending=False).reset_index(drop=True)

    # عيون الصقر البسيطة (إشارات)
    eyes = pd.DataFrame({
        "Metric": ["Chance of 50x+", "Chance of 100x+", "If 1 dominates (≥50%)", "Bonus (≥40% chance)"],
        "Value": [
            float((last["multiplier"] >= 50).mean()),
            float((last["multiplier"] >= 100).mean()),
            float((counts.get("1", 0) / max(total, 1)) >= 0.5),
            float((probs[["DISCO","VIP","STAYINALIVE","BAR"]].sum()))
        ]
    })

    return tiles, eyes, window


def board_matrix(tiles: pd.DataFrame) -> pd.DataFrame:
    """
    يُعيد فقط عمودين لتغذية عرض اللوح.
    """
    return tiles[["Tile", "P(≥1 in 10)"]].copy()
