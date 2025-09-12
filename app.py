import streamlit as st
import pandas as pd
from pandas.io.formats.style import Styler  # استيراد صحيح لو حابب تستخدم Styler

# ================= إعداد الصفحة =================
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ================= سايد بار =================
st.sidebar.header("⚙️ الإعدادات")

window = st.sidebar.slider("Window size (spins)", 50, 200, 120, 5)
refresh_rate = st.sidebar.slider("كل كم ثانية؟", 10, 120, 45, 5)
auto_refresh = st.sidebar.checkbox("تحديث تلقائي", True)

# ================= تحميل البيانات =================
@st.cache_data
def load_data(file_path="Funky_Brain_V2_7_4_LIVE-1.xlsx"):
    return pd.read_excel(file_path, sheet_name=None)

data_dict = load_data()
sheet_name = list(data_dict.keys())[0]
df = data_dict[sheet_name]

# ================= دوال تنسيق =================
def style_header(df: pd.DataFrame, color_hex: str):  # بدون -> pd.io...
    styled = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("background-color", color_hex),
                          ("color", "white"),
                          ("font-weight", "bold"),
                          ("text-align", "center")]
            }
        ]
    ).set_properties(**{"text-align": "center"})
    return styled

def style_table(df: pd.DataFrame):
    return (
        df.style
        .highlight_max(axis=0, color="lightgreen")
        .highlight_min(axis=0, color="lightcoral")
        .set_properties(**{"text-align": "center"})
    )

# ================= عرض الجداول =================
st.subheader("📊 جدول الاحتمالات")
styled_df = style_header(df, "#2E86C1")
st.dataframe(styled_df, use_container_width=True)

# لو عندك Sheets ثانية
if len(data_dict) > 1:
    st.subheader("📑 أوراق إضافية")
    for sheet, sheet_df in data_dict.items():
        st.markdown(f"### {sheet}")
        styled_sheet = style_table(sheet_df)
        st.dataframe(styled_sheet, use_container_width=True)
