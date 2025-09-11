import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ===== Sidebar =====
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

st.sidebar.subheader("مصدر البيانات")
auto = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_rate = st.sidebar.slider("⏳ كل كم ثانية", 10, 90, 45)

# رابط Google Sheets (ثابت)
CSV_URL = "https://docs.google.com/spreadsheets/d/1z15_Wc6mEWFbsrQduq1UB4bh-oy-bJdp952p9OyACCk/export?format=csv"

def load_data():
    try:
        response = requests.get(CSV_URL)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except Exception as e:
        st.error(f"⚠️ لم أتمكن من تحميل البيانات: {e}")
        return None

# زر لجلب البيانات
if st.sidebar.button("Fetch latest"):
    df = load_data()
else:
    df = None

# تحديث تلقائي
if auto:
    st.sidebar.write("⏳ سيتم التحديث الذاتي…")
    st.rerun()

# ===== Main Content =====
if df is not None:
    st.subheader("📊 آخر البيانات")
    st.dataframe(df.tail(20))  # عرض آخر 20 رمية
else:
    st.info("⬅️ اضغط على *Fetch latest* أو فعّل التحديث التلقائي")
