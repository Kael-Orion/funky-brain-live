# app.py
import streamlit as st
import pandas as pd
import time

# دوالنا الأساسية من الملف المساعد
from funkybrain_core import normalize_df, compute_probs, board_matrix

# ================== إعداد الصفحة ==================
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ================== إعداد رابط Google Sheets ==================
# رابطك الذي أرسلته:
# https://docs.google.com/spreadsheets/d/1z15_Wc6mEWFbsrQduq1UB4bh-oy-bJdp952p9OyACCk/edit?usp=sharing
SHEET_ID = "1z15_Wc6mEWFbsrQduq1UB4bh-oy-bJdp952p9OyACCk"
SHEET_NAME = "sample_spins"  # غيّرها إذا كان اسم الورقة مختلفًا
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

# ================== الشريط الجانبي ==================
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

st.sidebar.subheader("تحديث تلقائي")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
period = st.sidebar.slider("كل كم ثانية؟", 10, 120, 45, step=5)

st.sidebar.caption("مصدر البيانات: Google Sheets")
st.sidebar.code(CSV_URL, language="text")

# ================== تحميل وتنظيف البيانات ==================
@st.cache_data(ttl=60)
def load_csv(url: str) -> pd.DataFrame:
    # قراءة CSV مباشرة من Google Sheets
    df = pd.read_csv(url)

    # التأكد من الأعمدة اللازمة: ts, segment, multiplier
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for need in ["ts", "segment", "multiplier"]:
        if need not in cols_lower:
            raise ValueError(f"عمود مفقود في الجدول: {need}")

    # إعادة ترتيب وإعادة تسمية للأمان
    df = df[[cols_lower["ts"], cols_lower["segment"], cols_lower["multiplier"]]].copy()
    df.columns = ["ts", "segment", "multiplier"]

    # تنظيف المضاعِف مثل '25X' -> 25
    df["multiplier"] = (
        df["multiplier"].astype(str)
        .str.replace("x", "", case=False)
        .str.replace(r"\D+", "", regex=True)
    )
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1).astype(int)

    # توحيد أسماء القطاعات
    df["segment"] = df["segment"].astype(str).str.strip().str.upper()

    # إزالة التكرارات على (ts, segment, multiplier)
    df = df.drop_duplicates(subset=["ts", "segment", "multiplier"])

    return df

def render(df: pd.DataFrame):
    # التطبيع + الحسابات
    df_norm = normalize_df(df)
    tiles, eyes, win = compute_probs(df_norm, window)

    # جدول التوقعات
    st.subheader("Tiles – احتمالات وتوقعات")
    st.dataframe(
        tiles.style.format({
            "P(next)": "{:.2%}",
            "Exp in 10": "{:.1f}",
            "P(≥1 in 10)": "{:.2%}",
            "Exp in 15": "{:.1f}",
            "P(≥1 in 15)": "{:.2%}",
        }),
        use_container_width=True
    )

    # لوحة الاحتمالات
    st.subheader("Board – P(≥1 in 10)")
    board = board_matrix(tiles)
    st.dataframe(board, use_container_width=True)

    # تنبيه سريع لفرص قويّة (Eyes Eagle)
    hot = tiles.loc[tiles["P(≥1 in 10)"] >= 0.65].sort_values("P(≥1 in 10)", ascending=False)
    if not hot.empty:
        st.success("🦅 Eyes Eagle – فرص ساخنة:")
        st.write(
            hot[["Title", "P(≥1 in 10)", "P(next)"]].assign(
                **{
                    "P(≥1 in 10)": hot["P(≥1 in 10)"].map(lambda x: f"{x:.0%}"),
                    "P(next)": hot["P(next)"].map(lambda x: f"{x:.0%}"),
                }
            )
        )

# ================== حلقة التحديث ==================
placeholder = st.empty()

while True:
    try:
        df = load_csv(CSV_URL)
        with placeholder.container():
            render(df)
    except Exception as e:
        st.error(f"تعذّر تحميل جدول Google Sheets: {e}")

    if not auto_refresh:
        break

    time.sleep(period)
    st.rerun()
