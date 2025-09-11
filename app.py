# app.py
import time, io, requests
import pandas as pd
import streamlit as st

from funkybrain_core import normalize_df, compute_probs, board_matrix

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ============ Sidebar ============
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

st.sidebar.divider()
st.sidebar.subheader("مصدر البيانات")
remote_url = st.sidebar.text_input("Remote CSV URL (اختياري)")
fetch_now = st.sidebar.button("Fetch remote now")

st.sidebar.subheader("أو ارفع CSV يدويًا")
uploads = st.sidebar.file_uploader("Drag & drop CSV", type=["csv"], accept_multiple_files=True)

st.sidebar.divider()
st.sidebar.subheader("Auto-refresh")
auto = st.sidebar.checkbox("تشغيل تلقائي", value=False)
interval = st.sidebar.slider("كل كم ثانية؟", 10, 90, 45)

# ============ Caching ============
@st.cache_data(show_spinner=False, ttl=60)
def _read_many(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, engine="python", low_memory=False)
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return normalize_df(raw)

@st.cache_data(show_spinner=False, ttl=30)
def _fetch_remote_csv(url: str) -> pd.DataFrame:
    """يجلب CSV من رابط عام (غوغل شيتس/جيتهب/raw ...) ثم يطبّع الحقول."""
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    data = io.StringIO(r.text)
    raw = pd.read_csv(data, low_memory=False)
    return normalize_df(raw)

@st.cache_data(show_spinner=False, ttl=30)
def _compute(df: pd.DataFrame, window: int):
    return compute_probs(df, window)

# ============ اختيار المصدر ============
df = None
source = None

if uploads:
    df = _read_many(uploads)
    source = "uploads"
elif remote_url:
    try:
        if fetch_now or auto:
            st.sidebar.write("⏳ fetching remote …")
        df = _fetch_remote_csv(remote_url)
        source = "remote"
    except Exception as e:
        st.error(f"فشل جلب CSV من الرابط: {e}")

# ============ العرض ============
if df is None or len(df) == 0:
    st.info("أدخل رابط CSV في الحقل الجانبي أو ارفع ملف CSV لبدء التحليل.")
else:
    with st.spinner("⏳ Processing data..."):
        tiles, eyes, win = _compute(df, window)
        board = board_matrix(tiles)

    st.caption(f"Source: {source} • Rows: {len(df)} • Window={win}")

    # جدول الاحتمالات
    st.subheader("Tiles – احتمالات وتوقعات")
    nice = tiles.copy()
    for c in ["P(next)", "P(≥1 in 10)", "P(≥1 in 15)"]:
        nice[c] = (nice[c]*100).round(1).astype(str) + "%"
    for c in ["Exp in 10", "Exp in 15"]:
        nice[c] = nice[c].round(1)
    st.dataframe(nice, use_container_width=True, hide_index=True)

    # Board مبسّط
    st.subheader("Board – P(≥1 in 10)")
    colA, colB, colC, colD = st.columns(4)
    cols = [colA, colB, colC, colD]
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
                bg = (
                    "#f2a23a" if seg in ["P","L","A","Y"] else
                    "#d04f93" if seg in ["F","U","N","K"] else
                    "#4d2ddb" if seg in ["T","I","M","E"] else
                    "#3aa35e" if seg == "BAR" else
                    "#ff6b6b" if seg == "VIP" else
                    "#2b8cff" if seg in ["DISCO","STAYINALIVE"] else
                    "#e0ad60"
                )
                fg = "yellow" if seg == "BAR" else "white"
                st.markdown(
                    f"""
<div style="background:{bg};color:{fg};padding:8px;border-radius:8px;
     margin:6px 0;text-align:center;font-weight:700">
  {seg} — {pct:.0f}%
</div>
""", unsafe_allow_html=True)

    # Eyes Eagle مختصر
    st.subheader("Eyes Eagle – تنبيهات (15 رمية)")
    c1, c2 = st.columns(2)
    with c1: st.metric("Chance of 50x+ (est.)", f"{eyes.get('p50_in15',0)*100:.1f}%")
    with c2: st.metric("Dominance of 1", f"{eyes.get('ones_ratio',0)*100:.1f}%")

# Auto-refresh يعمل للمصدرين
if auto and (uploads or remote_url):
    time.sleep(interval)
    st.cache_data.clear()
    st.rerun()
