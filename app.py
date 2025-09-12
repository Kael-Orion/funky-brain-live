# -*- coding: utf-8 -*-
import io
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# إعداد الصفحة
# --------------------------
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠🎡 Funky Brain – LIVE")

# --------------------------
# أدوات مساعدة
# --------------------------
TILE_ORDER = [
    "1", "BAR",
    "P", "L", "A", "Y",
    "F", "U", "N", "K",
    "VIP", "DISCO", "STAYINALIVE"
]

GROUP_MAP = {
    # مفاتيح أحرف/أيقونات إلى اسم التجميع
    "1": "One",
    "BAR": "BAR",
    "P": "Orange (PLAY)",
    "L": "Orange (PLAY)",
    "A": "Orange (PLAY)",
    "Y": "Orange (PLAY)",
    "F": "Pink (FUNK)",
    "U": "Pink (FUNK)",
    "N": "Pink (FUNK)",
    "K": "Pink (FUNK)",
    "VIP": "VIP",
    "DISCO": "DISCO",
    "STAYINALIVE": "STAYINALIVE",
}

DISPLAY_NAME = {
    "1": "1",
    "BAR": "BAR",
    "P": "P",
    "L": "L",
    "A": "A",
    "Y": "Y",
    "F": "F",
    "U": "U",
    "N": "N",
    "K": "K",
    "VIP": "VIP",
    "DISCO": "Disco",
    "STAYINALIVE": "Stayin’ Alive",
}

def nice_percent(x, digits=2):
    return f"{x*100:.{digits}f}%"

def expected_in_n(p, n):
    # التوقع = n*p
    return n * p

def prob_at_least_one(p, n):
    # P(≥1) = 1 - (1-p)^n
    return 1 - (1 - p) ** n

def style_table(df, header_color="#222", header_font="#fff"):
    return (
        df.style
        .format(precision=2)
        .set_table_styles(
            [
                {"selector": "thead th",
                 "props": [("background-color", header_color),
                           ("color", header_font),
                           ("font-weight", "600"),
                           ("text-align", "center")]},
                {"selector": "tbody td",
                 "props": [("text-align", "center")]}
            ]
        )
        .hide(axis="index")
    )

# --------------------------
# الشريط الجانبي
# --------------------------
st.sidebar.header("⚙️ الإعدادات")

# نافذة التحليل (عدد اللفات الأخيرة)
window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

# تحديث تلقائي (اختياري)
auto = st.sidebar.checkbox("التحديث التلقائي")
every = st.sidebar.slider("كل كم ثانية؟", 10, 120, 45, step=5) if auto else None

# مصدر البيانات
st.sidebar.subheader("مصدر البيانات")
src = st.sidebar.radio(
    "اختر المصدر",
    ["رفع ملف Excel/CSV", "رابط Google Sheets (CSV)"],
    horizontal=False
)

uploaded = None
gsheet_csv_url = None

if src == "رفع ملف Excel/CSV":
    uploaded = st.sidebar.file_uploader(
        "اختر ملفًا (Excel .xlsx أو CSV)",
        type=["xlsx", "csv"]
    )
else:
    gsheet_csv_url = st.sidebar.text_input(
        "ضع رابط CSV من Google Sheets (ملف > مشاركة > نشر للويب > CSV)",
        value=""
    )

# اسم الورقة داخل Excel (إن وُجد)
sheet_name = st.sidebar.text_input("اسم الورقة (Excel فقط)", value="sample_spins")

# --------------------------
# تحميل البيانات
# --------------------------
@st.cache_data(show_spinner=False)
def load_data_from_excel(file_bytes: bytes, sheet: str):
    try:
        # لو CSV
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df
        except Exception:
            pass
        # Excel
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
        return df
    except Exception as e:
        raise RuntimeError(f"تعذّر قراءة الملف: {e}")

@st.cache_data(show_spinner=False)
def load_data_from_csv_url(url: str):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise RuntimeError(f"تعذّر تحميل CSV من الرابط: {e}")

def validate_columns(df: pd.DataFrame):
    needed = {"ts", "segment", "multiplier"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"أعمدة مفقودة في الجدول: {', '.join(missing)}")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف الأعمدة وضبط الأنواع وترميز القطاعات."""
    out = df.copy()
    # تحويل الطابع الزمني إن كان نصًا
    if "ts" in out.columns:
        with pd.option_context("mode.chained_assignment", None):
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce")

    # توحيد أسماء القطاعات (segment)
    # نقبل أحرف صغيرة/كبيرة وبعض الصيغ الشائعة
    repl = {
        "stayinalive": "STAYINALIVE",
        "stayinalive ": "STAYINALIVE",
        "disco": "DISCO",
        "vip": "VIP",
        "bar": "BAR",
        "p": "P", "l": "L", "a": "A", "y": "Y",
        "f": "F", "u": "U", "n": "N", "k": "K",
        "1": "1"
    }
    with pd.option_context("mode.chained_assignment", None):
        out["segment"] = out["segment"].astype(str).str.strip()
        out["segment"] = out["segment"].str.upper().replace(repl)

        # المضاعف: نحذف X إن وُجد (مثل 25X)
        out["multiplier"] = (
            out["multiplier"].astype(str).str.upper().str.replace("X", "", regex=False)
        )
        # أي قيمة غير عددية نجعلها NaN ثم نحول إلى float
        out["multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce")

    # إبقاء الصفوف ذات قطاع معروف فقط
    out = out[out["segment"].isin(TILE_ORDER)]
    out = out.sort_values("ts", ascending=True).reset_index(drop=True)
    return out

def compute_tiles_table(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """يبني جدول Tiles: الاحتمالات الحالية وفق آخر نافذة."""
    if len(df) == 0:
        return pd.DataFrame(columns=["Title", "Group", "P(next)", "Exp in 10", "P(≥1 in 10)", "Exp in 15", "P(≥1 in 15)"])

    # آخر نافذة
    wdf = df.tail(win)
    total = len(wdf)

    rows = []
    for key in TILE_ORDER:
        cnt = (wdf["segment"] == key).sum()
        p = cnt / total if total > 0 else 0.0
        row = {
            "Title": DISPLAY_NAME.get(key, key),
            "Group": GROUP_MAP.get(key, "—"),
            "P(next)": p,
            "Exp in 10": expected_in_n(p, 10),
            "P(≥1 in 10)": prob_at_least_one(p, 10),
            "Exp in 15": expected_in_n(p, 15),
            "P(≥1 in 15)": prob_at_least_one(p, 15),
        }
        rows.append(row)

    tdf = pd.DataFrame(rows)

    # تنسيقات العرض
    show = tdf.copy()
    show["P(next)"] = show["P(next)"].apply(nice_percent)
    show["Exp in 10"] = show["Exp in 10"].map(lambda x: f"{x:.1f}")
    show["P(≥1 in 10)"] = show["P(≥1 in 10)"].apply(nice_percent)
    show["Exp in 15"] = show["Exp in 15"].map(lambda x: f"{x:.1f}")
    show["P(≥1 in 15)"] = show["P(≥1 in 15)"].apply(nice_percent)

    return show

# --------------------------
# تحميل المصدر المختار
# --------------------------
df_raw = None
load_err = None

if src == "رفع ملف Excel/CSV":
    if uploaded is not None:
        try:
            df_raw = load_data_from_excel(uploaded.getvalue(), sheet_name)
        except Exception as e:
            load_err = str(e)
else:
    if gsheet_csv_url:
        try:
            df_raw = load_data_from_csv_url(gsheet_csv_url)
        except Exception as e:
            load_err = str(e)

if load_err:
    st.error(load_err)

if df_raw is None:
    st.info("⬆️ ارفع ملفك (أو ضع رابط CSV من Google Sheets) لبدء التحليل.\n\n"
            "يجب أن يحتوي الجدول على الأعمدة: **ts**, **segment**, **multiplier**.")
    st.stop()

# تأكيد الأعمدة المطلوبة
try:
    validate_columns(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

# تنظيف وتطبيع البيانات
df = normalize_df(df_raw)

# --------------------------
# عرض جداول/ملخص
# --------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tiles – احتمالات وتوقعات")
    tiles_df = compute_tiles_table(df, window)
    st.dataframe(style_table(tiles_df, header_color="#101828"), use_container_width=True)

with col2:
    st.subheader("ملخص النافذة")
    wdf = df.tail(window).copy()
    total = len(wdf)
    by_group = (
        wdf.assign(Group=wdf["segment"].map(GROUP_MAP))
           .groupby("Group", dropna=False)["segment"]
           .count()
           .rename("count")
           .sort_values(ascending=False)
    )
    summary = pd.DataFrame({
        "نافذة": [window],
        "عدد اللفات (نافذة)": [total],
        "آخر تحديث": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    st.dataframe(style_table(summary, header_color="#512DA8"), use_container_width=True)

    st.write("**العدّ حسب المجموعة (داخل النافذة):**")
    grp_df = by_group.reset_index()
    st.dataframe(style_table(grp_df, header_color="#512DA8"), use_container_width=True)

# سجلّ آخر اللفات (للمراجعة)
st.subheader("آخر اللفات (raw)")
tail_show = df.tail(min(200, len(df))).copy()
tail_show["multiplier"] = tail_show["multiplier"].map(lambda x: f"{x:.0f}X" if pd.notnull(x) else "")
st.dataframe(style_table(tail_show, header_color="#0B7285"), use_container_width=True)

# التحديث التلقائي
if auto and every:
    st.caption(f"سيتم إعادة التحميل كل **{every}** ثانية.")
    # طريقة بسيطة لإجبار إعادة التحميل بدون حزم إضافية:
    time.sleep(every)
    st.experimental_rerun()
