# funkybrain_core.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

SEGMENTS = [
    "1", "BAR",
    "P","L","A","Y",
    "F","U","N","K",
    "T","I","M","E",
    "DISCO","STAYINALIVE","VIP"
]

GROUPS = {
    "P":"Orange (PLAY)","L":"Orange (PLAY)","A":"Orange (PLAY)","Y":"Orange (PLAY)",
    "F":"Pink (FUNK)","U":"Pink (FUNK)","N":"Pink (FUNK)","K":"Pink (FUNK)",
    "T":"Violet (TIME)","I":"Violet (TIME)","M":"Violet (TIME)","E":"Violet (TIME)",
    "1":"One","BAR":"BAR",
    "DISCO":"Bonus","STAYINALIVE":"Bonus","VIP":"Bonus"
}

# خرائط أيقونات casinoscores -> قطاعات
ICON_MAP = {
    "1.png":"1",
    "bar":"BAR",
    "disco":"DISCO",
    "vip":"VIP",
    "stayinalive":"STAYINALIVE",
    # أحرف
    "/p.png":"P","/l.png":"L","/a.png":"A","/y.png":"Y",
    "/f.png":"F","/u.png":"U","/n.png":"N","/k.png":"K",
    "/t.png":"T","/i.png":"I","/m.png":"M","/e.png":"E",
}

def _icon_to_segment(s: str) -> str:
    s = str(s).lower()
    # جرّب استخراج آخر جزء من مسار funky-time/<key>.png
    m = re.search(r"funky-time/([a-z0-9_-]+)\.png", s)
    if m:
        key = m.group(1)
        # مفاتيح معروفة بالاسم
        if key in ("1","bar","vip","disco","stayinalive"):
            return key.upper() if key != "1" else "1"
        # أحرف مفردة
        if len(key) == 1 and key.isalpha():
            return key.upper()
    # بحث سريع بالخرائط الثابتة
    for k,v in ICON_MAP.items():
        if k in s:
            return v
    # إن تعذّر، أرجِع القيمة كما هي (قد تكون حرفًا أصلاً)
    return s.upper()

def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    يتوقّع أعمدة: ts, segment, multiplier
    أو: ts, icon/src, multiplier  (من casinoscores الخام)
    """
    df = raw.copy()

    # اسماء اعمدة محتملة
    cols = {c.lower():c for c in df.columns}
    # الطابع الزمني
    ts_col = cols.get("ts") or list(df.columns)[0]
    df.rename(columns={ts_col:"ts"}, inplace=True)

    # المضاعِف
    mult_col = cols.get("multiplier")
    if not mult_col:
        # حاول استنتاجه من نص مثل "25X" أو "1X"
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\d+\s*[xX]$").any():
                mult_col = c
                break
    if mult_col:
        df["multiplier"] = (
            df[mult_col].astype(str)
            .str.extract(r"(\d+)")[0]
            .astype("Int64")
            .fillna(1)
            .astype(int)
        )
    else:
        df["multiplier"] = 1

    # القطاع (segment) أو الأيقونة
    seg_col = cols.get("segment")
    if seg_col:
        df["segment"] = df[seg_col].astype(str).str.upper()
    else:
        # ابحث عن عمود رابط الصورة
        icon_col = None
        for c in df.columns:
            if df[c].astype(str).str.contains(r"funky-time/").any():
                icon_col = c
                break
        if icon_col:
            df["segment"] = df[icon_col].map(_icon_to_segment)
        else:
            # لو ما لقينا شيء، اعتبر العمود الثاني هو القطاع
            fallback = list(df.columns)[1] if len(df.columns) > 1 else "segment"
            df["segment"] = df[fallback].astype(str).str.upper()

    # حصر القيم ضمن لائحة القطاعات المعتمدة
    df["segment"] = df["segment"].apply(lambda s: s if s in SEGMENTS else s)

    # نظّف الطابع الزمني لصيغة واحدة (لا نستعمله حسابيًا هنا)
    df["ts"] = df["ts"].astype(str)

    # أعمدة نهائية بالترتيب
    out = df[["ts","segment","multiplier"]].copy()
    return out

def _probabilities_from_counts(counts: Dict[str,int]) -> Dict[str,float]:
    total = max(1, sum(counts.values()))
    return {k: counts.get(k,0)/total for k in SEGMENTS}

def compute_probs(df: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, Dict[str,float], int]:
    """
    يُرجع:
      tiles_df: DataFrame بكل القطاعات واحتمالاتها
      eyes: إشارات/ملخّصات سريعة (يمكن عرضها في Eyes Eagle)
      win: حجم النافذة المستخدم
    """
    if len(df) == 0:
        empty = pd.DataFrame({"Title":SEGMENTS,"Group":[GROUPS.get(s,"") for s in SEGMENTS],
                              "P(next)":0.0,"Exp in 10":0.0,"P(≥1 in 10)":0.0,
                              "Exp in 15":0.0,"P(≥1 in 15)":0.0})
        return empty, {}, window

    last = df.tail(window)
    counts = last["segment"].value_counts().to_dict()
    p = _probabilities_from_counts(counts)

    def p_at_least_once(p_single: float, n: int) -> float:
        return 1.0 - (1.0 - p_single)**n

    rows = []
    for s in SEGMENTS:
        ps = p.get(s, 0.0)
        rows.append({
            "Title": s,
            "Group": GROUPS.get(s, ""),
            "P(next)": ps,
            "Exp in 10": ps * 10.0,
            "P(≥1 in 10)": p_at_least_once(ps, 10),
            "Exp in 15": ps * 15.0,
            "P(≥1 in 15)": p_at_least_once(ps, 15),
        })
    tiles = pd.DataFrame(rows)

    # إشارات مبسطة (عيـن الصقر)
    # تقدير احتمالات ≥50x في النافذة (تقريب سريع)
    bonus_mask = last["multiplier"] >= 50
    p50 = bonus_mask.mean() if len(last) else 0.0
    eyes = {
        "p50_in15": 1 - (1 - p50)**15 if p50 > 0 else 0.0,
        "ones_ratio": (last["segment"] == "1").mean(),
    }
    return tiles, eyes, window

def board_matrix(tiles_df: pd.DataFrame) -> pd.DataFrame:
    """يُجهّز جدول بسيط للعرض اللوحي مع الترتيب اللوني."""
    order = SEGMENTS
    df = tiles_df.set_index("Title").reindex(order).reset_index()
    return df
