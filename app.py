# app.py
import time
import pandas as pd
import streamlit as st

from funkybrain_core import normalize_df, compute_probs, board_matrix

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ===== Sidebar =====
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

st.sidebar.divider()
st.sidebar.subheader("رفع ملف CSV من casinoscores")
uploads = st.sidebar.file_uploader("Drag and drop files here", type=["csv"], accept_multiple_files=True)

st.sidebar.divider()
st.sidebar.subheader("Auto-refresh")
auto = st.sidebar.checkbox("تشغيل تلقائي", value=False)
interval = st.sidebar.slider("كل كم ثانية؟", 10, 90, 45)

# ====== Caching ======
@st.cache_data(show_spinner=False, ttl=60)
def _read_many(files):
    dfs = []
    for f in files:
        # قراءة سريعة: أعمدة المحتملة فقط
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            # إعادة الفتح لأن Streamlit يغلق المؤشر بعد الفشل
            f.seek(0)
            df = pd.read_csv(f, engine="python", low_memory=False)
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return normalize_df(raw)

@st.cache_data(show_spinner=False, ttl=30)
def _compute(df: pd.DataFrame, window: int):
    return compute_probs(df, window)

# ====== UI Body ======
placeholder = st.empty()

if not uploads:
    st.info("ابدأ التحليل برفع ملف CSV من casinoscores.")
else:
    with st.spinner("⏳ Processing data..."):
        df = _read_many(uploads)
        tiles, eyes, win = _compute(df, window)
        board = board_matrix(tiles)

    # ===== Tiles Table =====
    st.subheader("Tiles – احتمالات وتوقعات")
    nice = tiles.copy()
    # تنسيقات عرض فقط
    for c in ["P(next)","P(≥1 in 10)","P(≥1 in 15)"]:
        nice[c] = (nice[c]*100).round(1).astype(str) + "%"
    for c in ["Exp in 10","Exp in 15"]:
        nice[c] = nice[c].round(1)
    st.dataframe(nice, use_container_width=True, hide_index=True)

    # ===== Board View (مبسّط) =====
    st.subheader("Board – P(≥1 in 10)")
    colA, colB, colC, colD = st.columns(4)
    cols = [colA,colB,colC,colD]
    layout = [
        ["1","P","F","T","DISCO"],
        ["BAR","L","U","I"],
        ["A","N","M","STAYINALIVE"],
        ["Y","K","E","VIP"],
    ]
    for c, line in zip(cols, layout):
        with c:
            for seg in line:
                row = board[board["Title"] == seg].iloc[0]
                pct = float(row["P(≥1 in 10)"]) * 100.0
                st.markdown(
                    f"""
<div style="background:{'#f2a23a' if seg in ['P','L','A','Y'] else '#d04f93' if seg in ['F','U','N','K'] else '#4d2ddb' if seg in ['T','I','M','E'] else '#3aa35e' if seg=='BAR' else '#ff6b6b' if seg=='VIP' else '#2b8cff' if seg in ['DISCO','STAYINALIVE'] else '#e0ad60'}; 
            color:{'yellow' if seg=='BAR' else 'white'};
            padding:8px; border-radius:8px; margin:6px 0; text-align:center; font-weight:700">
  {seg} — {pct:.0f}%
</div>
""",
                    unsafe_allow_html=True,
                )

    # ===== Eyes Eagle (مختصر) =====
    st.subheader("Eyes Eagle – تنبيهات (على 15 رمية)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chance of 50x+ (est.)", f"{eyes.get('p50_in15',0)*100:.1f}%")
    with col2:
        st.metric("نزول 1 (domination)", f"{eyes.get('ones_ratio',0)*100:.1f}%")

# ===== Auto-refresh (آمن) =====
if auto and uploads:
    # لا نستخدم st.experimental_rerun بدون مهلة
    time.sleep(interval)
    st.cache_data.clear()  # تأكد أن النتائج تُحدّث
    st.rerun()
