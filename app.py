import io
import re
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests

# =========================
# إعدادات عامة + ألوان
# =========================
APP_TITLE = "Funky Brain LIVE"
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1z15_Wc6mEWFbsrQduq1UB4bh-oy-bJdp952p9OyACCk/edit?usp=sharing"
DATA_SHEET_NAME = "Data"   # يجب أن يكون تبويب البيانات بهذا الاسم ويحتوي الأعمدة: ts, segment, multiplier

# ألوان رؤوس الأقسام (مطابقة قدر الإمكان)
COLOR_TILES     = "#222222"
COLOR_BOARD     = "#222222"
COLOR_EYESEAGLE = "#8B4513"   # من ملف Excel (EyesEagle)

# خريطة المجموعات لأسماء التايلات (يمكنك تعديلها لاحقاً إن رغبت)
SEGMENT_GROUP = {
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
    # لو عندك قطع أخرى مثل VIP / DISCO / StayinAlive أضفها هنا:
    "VIP": "VIP",
    "Disco": "DISCO",
    "StayinAlive": "STAYINALIVE",
}

# ترتيب العرض في جدول Tiles (نفس ترتيب ملفك V2)
TILE_ORDER = ["1", "BAR", "P", "L", "A", "Y", "F", "U", "N", "K", "VIP", "Disco", "StayinAlive"]


# =========================
# أدوات تحميل البيانات
# =========================
def gsheet_to_csv_url(sheet_url: str, sheet_name: str = DATA_SHEET_NAME) -> str:
    """
    يحوّل رابط Google Sheets إلى رابط CSV قابل للقراءة مباشرة.
    يفضّل أن تكون الورقة مشتركة "Anyone with the link".
    نجرّب طريقتين: export بالgid إن وُجد، أو gviz بالاسم.
    """
    # حاول استخراج الـ id و gid من الرابط
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    sheet_id = m.group(1) if m else None
    gid = None
    mg = re.search(r"[?&]gid=(\d+)", sheet_url)
    if mg:
        gid = mg.group(1)

    if sheet_id and gid:
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&id={sheet_id}&gid={gid}"

    if sheet_id:
        # gviz عبر اسم الورقة
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    # fallback: نرجع الرابط كما هو (قد يكون CSV مباشر)
    return sheet_url


@st.cache_data(show_spinner=False)
def load_from_google_sheets(sheet_url: str, sheet_name: str = DATA_SHEET_NAME) -> pd.DataFrame:
    csv_url = gsheet_to_csv_url(sheet_url, sheet_name)
    r = requests.get(csv_url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    return df


def load_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        # Excel
        return pd.read_excel(uploaded_file, sheet_name=DATA_SHEET_NAME)


# =========================
# تجهيز واحتساب المؤشرات
# =========================
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # نتأكد من الأعمدة المطلوبة
    required = ["ts", "segment", "multiplier"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"الأعمدة الناقصة: {missing}. يجب أن تكون الأعمدة موجودة: {required}")

    out = df.copy()
    # تواريخ
    try:
        out["ts"] = pd.to_datetime(out["ts"])
    except Exception:
        pass

    # تنظيف نصوص
    out["segment"] = out["segment"].astype(str).str.strip()
    out["multiplier"] = out["multiplier"].astype(str).str.upper().str.replace("X", "", regex=False)
    out["multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce").fillna(1).astype(int)

    # رتب من الأحدث إلى الأقدم
    if np.issubdtype(out["ts"].dtype, np.datetime64):
        out = out.sort_values("ts", ascending=False)
    else:
        out = out.iloc[::-1]  # fallback

    return out.reset_index(drop=True)


def compute_probs(df: pd.DataFrame, window: int) -> pd.DataFrame:
    last = df.head(window).copy()
    total = len(last)
    freq = last["segment"].value_counts().rename("count").to_frame()
    freq["P(next)"] = freq["count"] / total
    freq["Exp in 10"] = 10 * freq["P(next)"]
    freq["P(≥1 in 10)"] = 1 - (1 - freq["P(next)"]) ** 10
    freq["Exp in 15"] = 15 * freq["P(next)"]
    freq["P(≥1 in 15)"] = 1 - (1 - freq["P(next)"]) ** 15

    # ترتيب الأعمدة النهائية
    out = freq.reset_index().rename(columns={"index": "Title"})
    out["Group"] = out["Title"].map(SEGMENT_GROUP).fillna("—")
    # حافظ على الترتيب المطلوب
    out["order"] = out["Title"].apply(lambda t: TILE_ORDER.index(t) if t in TILE_ORDER else 10_000)
    out = out.sort_values(["order", "Title"]).drop(columns=["order"])
    # تنسيق النسب للعرض
    for c in ["P(next)", "P(≥1 in 10)", "P(≥1 in 15)"]:
        out[c] = (out[c] * 100).round(2).astype(str) + "%"
    out["Exp in 10"] = out["Exp in 10"].round(2)
    out["Exp in 15"] = out["Exp in 15"].round(2)
    return out[["Title", "Group", "P(next)", "Exp in 10", "P(≥1 in 10)", "Exp in 15", "P(≥1 in 15)"]]


def board_summary(df_probs: pd.DataFrame) -> pd.DataFrame:
    # ملخص "Board – P(≥1 in 10)" حسب Group
    tmp = df_probs.copy()
    # خذ متوسط P(≥1 in 10) داخل كل Group (كمؤشر بسيط)
    tmp["p10"] = tmp["P(≥1 in 10)"].str.rstrip("%").astype(float) / 100.0
    agg = tmp.groupby("Group", as_index=False)["p10"].mean().sort_values("p10", ascending=False)
    agg["P(≥1 in 10)"] = (agg["p10"] * 100).round(2).astype(str) + "%"
    return agg[["Group", "P(≥1 in 10)"]]


def eyes_eagle(df_probs: pd.DataFrame, next_spins: int = 15, top_k: int = 8) -> pd.DataFrame:
    tmp = df_probs.copy()
    tmp["pN"] = tmp[f"P(≥1 in {next_spins})"].str.rstrip("%").astype(float)
    tmp = tmp.sort_values("pN", ascending=False).head(top_k)
    tmp = tmp.rename(columns={f"P(≥1 in {next_spins})": f"Alert P(≥1 in {next_spins})"})
    return tmp[["Title", "Group", f"Alert P(≥1 in {next_spins})"]]


def style_header(df: pd.DataFrame, color_hex: str) -> pd.io.formats.style.Styler:
    return df.style.set_table_styles(
        [{"selector": "th", "props": [("background-color", color_hex), ("color", "white")]}]
    )


# =========================
# واجهة Streamlit
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🧠 {APP_TITLE}")

# الشريط الجانبي
st.sidebar.header("⚙️ الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 300, 120, step=10)
auto = st.sidebar.checkbox("تحديث تلقائي (Auto-refresh)", value=False)
every_sec = st.sidebar.slider("كل كم ثانية؟", 10, 120, 45, step=5)

st.sidebar.subheader("📄 مصدر البيانات: Google Sheets")
sheet_url = st.sidebar.text_input("ضع رابط Google Sheets (تبويب Data)", value=DEFAULT_SHEET_URL)

st.sidebar.divider()
st.sidebar.subheader("بديل: ارفع ملف CSV/Excel (اختياري)")
uploaded = st.sidebar.file_uploader("اختر ملفًا", type=["csv", "xlsx"])

# تحميل البيانات
def load_data():
    if uploaded is not None:
        df0 = load_from_upload(uploaded)
    else:
        df0 = load_from_google_sheets(sheet_url, DATA_SHEET_NAME)
    return normalize_df(df0)

# Loop للتحديث التلقائي
place_tiles = st.empty()
place_board = st.empty()
place_eagle = st.empty()

def render_once():
    df = load_data()
    if df.empty:
        st.warning("لا توجد بيانات في تبويب Data.")
        return

    probs = compute_probs(df, window)

    with place_tiles.container():
        st.subheader("Tiles – احتمالات وتوقعات")
        st.dataframe(style_header(probs, COLOR_TILES), use_container_width=True, hide_index=True)

    with place_board.container():
        st.subheader("Board – P(≥1 in 10)")
        board = board_summary(probs)
        st.dataframe(style_header(board, COLOR_BOARD), use_container_width=True, hide_index=True)

    with place_eagle.container():
        st.subheader("EyesEagle – Alerts (next 15 spins)")
        eagle = eyes_eagle(probs, next_spins=15, top_k=8)
        st.dataframe(style_header(eagle, COLOR_EYESEAGLE), use_container_width=True, hide_index=True)

render_once()

if auto:
    # تحديث كل عدة ثوانٍ
    while True:
        time.sleep(every_sec)
        render_once()
