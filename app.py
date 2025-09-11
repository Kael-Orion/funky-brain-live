import time
import pandas as pd
import streamlit as st

# دوال المعالجة التي عندك من قبل
from funkybrain_core import normalize_df, compute_probs, board_matrix

# ---- (جديد) جالب بيانات من CasinoScores ----
try:
    from fetchers.casinoscores import fetch_latest   # تأكد أن الملف موجود: fetchers/casinoscores.py
    FETCH_AVAILABLE = True
except Exception:
    FETCH_AVAILABLE = False

# ---------------- UI ----------------
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# حافظ جلسة للبيانات
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["ts", "segment", "multiplier"])

# --------- Sidebar ---------
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 300, 200, step=10)

st.sidebar.write("---")
st.sidebar.caption("جلب تلقائي من casinoscores (تجريبي)")

colA, colB = st.sidebar.columns(2)
auto_refresh = colA.toggle("Auto-refresh كل 60s", value=False)
fetch_click = colB.button("Fetch latest (beta)", disabled=not FETCH_AVAILABLE)

if not FETCH_AVAILABLE:
    st.sidebar.info("لتفعيل الجلب التلقائي أنشئ الملف: fetchers/casinoscores.py")

st.sidebar.write("---")
st.sidebar.subheader("أو ارفع CSV من casinoscores")
uploads = st.sidebar.file_uploader("ارفع ملف/ملفات (CSV)", type=["csv"], accept_multiple_files=True)

# --------- مصادر البيانات ---------
def read_uploaded_csvs(files) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=["ts","segment","multiplier"])
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            # بعض مرات casinoscores يصدر بـ ;
            dfs.append(pd.read_csv(f, sep=";"))
    out = pd.concat(dfs, ignore_index=True)
    return out

def fetch_and_store():
    with st.spinner("Fetching latest spins..."):
        df_new = fetch_latest(limit=300)
        if df_new.empty:
            st.warning("لم أجد بيانات صالحة.")
        else:
            st.session_state.df = df_new
            st.success(f"تم جلب {len(df_new)} رمية من casinoscores.")

# زر الجلب اليدوي
if fetch_click and FETCH_AVAILABLE:
    fetch_and_store()

# التحديث التلقائي كل 60 ثانية (إن مُفعّل)
if auto_refresh and FETCH_AVAILABLE:
    # نجلب عند الفتح ثم نضبط مؤقّت بسيط
    if st.session_state.get("_last_fetch_ts") is None or (time.time() - st.session_state["_last_fetch_ts"] > 55):
        try:
            fetch_and_store()
            st.session_state["_last_fetch_ts"] = time.time()
        except Exception as e:
            st.warning(f"فشل الجلب التلقائي: {e}")

# رفع CSV يطغى على الجلب
if uploads:
    df_up = read_uploaded_csvs(uploads)
    if not df_up.empty:
        st.session_state.df = df_up

# df النهائي الذي سنحلله
raw = st.session_state.df.copy()

if raw.empty:
    st.info("ابدأ برفع CSV من casinoscores أو استخدم زر Fetch latest.")
    st.stop()

# --------- المعالجة كما في نسختك السابقة ---------
try:
    df = normalize_df(raw)  # توحيد الأعمدة والقيم
except Exception as e:
    st.error(f"خطأ في normalize_df: {e}")
    st.stop()

try:
    # حسب نسختك السابقة: كانت ترجع tiles, eyes, win
    tiles, eyes, win = compute_probs(df, window)
except Exception as e:
    st.error(f"خطأ في compute_probs: {e}")
    st.stop()

# --------- Tiles Table ---------
st.subheader("Tiles – احتمالات وتوقعات")
try:
    st.dataframe(
        tiles.style.format({
            "P(next)": "{:.1%}",
            "Exp in 10": "{:.1f}",
            "P(≥1 in 10)": "{:.1%}",
            "Exp in 15": "{:.1f}",
            "P(≥1 in 15)": "{:.1%}",
        }),
        use_container_width=True
    )
except Exception:
    st.dataframe(tiles, use_container_width=True)

# --------- Board View ---------
st.write("---")
st.subheader(f"Board – P(≥1 in 10) • Window={window}")
try:
    board = board_matrix(tiles)  # نفسها التي كنت تستعملها للرسم/الألوان
    st.dataframe(board, use_container_width=True)
except Exception:
    st.info("تعذّر إنشاء Board بالوظيفة board_matrix، اعرضنا الجدول فقط.")

# --------- Eyes Eagle ---------
st.write("---")
st.subheader("Eyes Eagle – Alerts (next 15 spins)")
try:
    st.dataframe(
        eyes.style.format({
            "Value": "{:.1%}",
            "Exp in 15": "{:.1f}",
        }),
        use_container_width=True
    )
except Exception:
    st.dataframe(eyes, use_container_width=True)

st.caption("v2 • Live fetch + CSV • إذا لاحظت اختلافًا في الصفحة المصدر، سنعدّل جالب البيانات بسرعة.")
