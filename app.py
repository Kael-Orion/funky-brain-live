# -*- coding: utf-8 -*-
import time
import pandas as pd
import streamlit as st

from funkybrain_core import normalize_df, compute_probs, board_matrix

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ===== Sidebar =====
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

st.sidebar.subheader("رفع ملف CSV من casinoscores")
uploads = st.sidebar.file_uploader("يمكن رفع أكثر من ملف (CSV)", type=["csv"], accept_multiple_files=True)

auto = st.sidebar.checkbox("Auto-refresh", value=False)
interval = st.sidebar.slider("كل كم ثانية؟", 10, 90, 45, step=5)

# ===== Main =====
placeholder = st.empty()

def render(_dfs: list[pd.DataFrame]):
    raw = pd.concat(_dfs, ignore_index=True) if len(_dfs) > 1 else _dfs[0]
    df = normalize_df(raw)
    tiles, eyes, win = compute_probs(df, window)

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Tiles – احتمالات وتوقعات")
        st.dataframe(
            tiles.style.format({
                "P(next)": "{:.1%}",
                "P(≥1 in 10)": "{:.1%}",
                "P(≥1 in 15)": "{:.1%}"
            })
        )
    with col2:
        st.subheader("Eyes Eagle – تنبيهات (بسيطة)")
        st.dataframe(
            eyes.style.format({"Value": "{:.1%}"})
        )

    # Board
    st.subheader(f"Board – P(≥1 in 10) • Window={win}")
    board = board_matrix(tiles)
    # ترتيب يشبه لوح اللعبة (تقريب مبسّط)
    desired = ["1","BAR",
               "P","L","A","Y",
               "F","U","N","K",
               "T","I","M","E",
               "DISCO","STAYINALIVE","VIP"]
    board["order"] = board["Tile"].apply(lambda t: desired.index(t) if t in desired else 999)
    board = board.sort_values("order").drop(columns="order")

    # عرض بسيط كجدول (اللوح الرسومي الكامل لاحقاً)
    st.dataframe(board.style.format({"P(≥1 in 10)": "{:.1%}"}))

if uploads:
    dfs = [pd.read_csv(f) for f in uploads]
    render(dfs)
else:
    st.info("ابدأ برفع ملف/ملفات CSV من casinoscores.")

# تحديث تلقائي اختياري (بدون جلب من الإنترنت)
if auto:
    st.sidebar.write("⏳ سيتم التحديث الذاتي…")
    st.experimental_rerun()
