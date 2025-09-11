# app.py
# -*- coding: utf-8 -*-
import time
import pandas as pd
import streamlit as st

# هذه الدوال موجودة عندك من قبل في مشروع الإكسل/بايثون
# إن اختلفت التواقيع، لا تقلق: وضعنا try/except تحت لعدم كسر التطبيق.
from funkybrain_core import normalize_df, compute_probs, board_matrix

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ===== Sidebar =====
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 200, step=10)

st.sidebar.subheader("جلب آخر الرميات (تجريبي)")
auto = st.sidebar.toggle("Auto-refresh", value=False, help="تحديث تلقائي كل 60 ثانية")
colA, colB = st.sidebar.columns([1, 1])
with colA:
    fetch_btn = st.button("سحب من casinoscores.py", use_container_width=True)
with colB:
    refresh_btn = st.button("Force Reload", use_container_width=True)

status_box = st.sidebar.empty()

@st.cache_data(ttl=60)
def _cached_fetch_latest():
    from fetchers.casinoscores import fetch_latest
    df_fetched = fetch_latest(limit=300)
    return df_fetched

# رفع CSV يدويًا (مسار بديل آمن)
st.sidebar.subheader("ارفع CSV من casinoscores")
uploads = st.sidebar.file_uploader("Drag & drop", type=["csv"], accept_multiple_files=True)

# ===== Data source selection =====
use_uploaded = True

if auto or fetch_btn or refresh_btn:
    try:
        raw = _cached_fetch_latest()
        status_box.info(f"✔ تم الجلب الآلي: {len(raw)} رمية")
        use_uploaded = False
    except Exception as e:
        status_box.error(f"فشل الجلب الآلي: {e}")
        use_uploaded = True

if use_uploaded:
    if not uploads:
        st.info("ابدأ التحليل برفع CSV من casinoscores أو استخدم زر الجلب في الشريط الجانبي.")
        st.stop()
    dfs = [pd.read_csv(f) for f in uploads]
    raw = pd.concat(dfs, ignore_index=True)

# ===== Normalize & compute =====
try:
    df = normalize_df(raw)
except Exception as e:
    st.error(f"normalize_df فشل: {e}")
    st.dataframe(raw.head())
    st.stop()

try:
    tiles, eyes, win = compute_probs(df, window)
except Exception as e:
    st.error(f"compute_probs فشل: {e}")
    st.dataframe(df.head())
    st.stop()

# ===== Tiles Table =====
st.subheader("Tiles – احتمالات وتوقعات")
st.dataframe(tiles, use_container_width=True)

# ===== Board (احتمال ≥1 خلال 10) =====
try:
    st.subheader("Board – P(≥1 in 10)")
    board_df = board_matrix(tiles)  # إن كانت الدالة تتوقع بيانات أخرى عدّلها لديك
    st.dataframe(board_df, use_container_width=True)
except Exception:
    # لو التوقيع مختلف، تخطَّ العرض البصري واكتفِ بعمود الاحتمالات
    st.warning("تعذر بناء لوحة Board بالوظيفة الحالية. يتم عرض الاحتمالات من الجدول فقط.")

# ===== Eyes Eagle (نفس منطق إصداراتك السابقة إن كانت موجودة في tiles/eyes) =====
if "Exp in 15" in tiles.columns and "P(≥1 in 15)" in tiles.columns:
    st.subheader("Eyes Eagle – مؤشرات سريعة")
    ee = tiles.loc[:, ["Tile", "Exp in 15", "P(≥1 in 15)"]].copy() if "Tile" in tiles.columns else tiles.loc[:, ["Exp in 15", "P(≥1 in 15)"]]
    st.dataframe(ee, use_container_width=True)

# ===== Auto refresh (واجهة فقط) =====
if auto:
    st.sidebar.caption("التحديث يعمل… (كل ~60 ثانية)")
    # Streamlit يعيد التنفيذ عند انتهاء TTL للكاش؛ لا نحتاج sleep هنا.
