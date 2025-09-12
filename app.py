import streamlit as st
import pandas as pd

# ==== إعداد الصفحة ====
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0b0f19; color: white; }
    .title { font-size: 20px; font-weight: bold; }
    .bonus { color: #f7b731; }
    .hawk-eye { border: 2px solid red; padding: 5px; margin: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==== الشريط الجانبي ====
st.sidebar.header("⚙️ الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 120)

st.sidebar.subheader("👁️ Hawk-Eye thresholds")
stop_bonus = st.sidebar.slider("STOP: حد البونص", 0.0, 1.0, 0.55)
stop_play = st.sidebar.slider("STOP: PLAY", 0.0, 1.0, 0.25)
go_signal = st.sidebar.slider("GO: إشارة دخول", 0.0, 1.0, 0.22)

# ==== رفع CSV ====
st.sidebar.subheader("📂 رفع CSV")
file = st.sidebar.file_uploader("اختر ملف CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.tail(20))  # عرض آخر 20 رمية

    # ==== عين الصقر ====
    st.markdown("### 👁️ Hawk-Eye Analysis")
    for _, row in df.tail(window).iterrows():
        seg, mul = row['segment'], row['multiplier']
        if "X" in str(mul) and int(mul.replace("X","")) >= 50:
            st.markdown(f"<div class='hawk-eye'>🎯 عين الصقر: {seg} {mul}</div>", unsafe_allow_html=True)

