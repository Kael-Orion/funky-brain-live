import streamlit as st, pandas as pd
from funkybrain_core import normalize_df, compute_probs, board_matrix

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ===== Sidebar =====
st.sidebar.header("الإعدادات")
window = st.sidebar.slider("Window size (spins)", 50, 200, 200, step=10)

st.sidebar.subheader("ارفع ملفات CSV من casinoscores")
uploads = st.sidebar.file_uploader(
    "اختر ملفًا أو أكثر (CSV)", type=["csv"], accept_multiple_files=True
)

if not uploads:
    st.info("ارفع ملف/ملفين CSV من casinoscores لبدء التحليل.")
    st.stop()

# ===== Read & normalize =====
dfs = [pd.read_csv(f) for f in uploads]
raw = pd.concat(dfs, ignore_index=True)
df = normalize_df(raw)

tiles, eyes, win = compute_probs(df, window)

# ===== Tiles =====
st.subheader("Tiles – احتمالات وتوقعات")
st.dataframe(
    tiles.style.format({
        "P(next)":"{:.2%}",
        "Exp in 10":"{:.1f}",
        "P(≥1 in 10)":"{:.2%}",
        "Exp in 15":"{:.1f}",
        "P(≥1 in 15)":"{:.2%}",
    }),
    use_container_width=True
)

# ===== Board =====
st.subheader("Board – P(≥1 in 10)")
probs10 = dict(zip(tiles["Tile"], tiles["P(≥1 in 10)"]))
rows = board_matrix(probs10)

def board_html(rows):
    # ألوان قريبة من اللعبة
    color = {
        "1":"#E9C46A", "BAR":"#2A9D8F",
        "P":"#F4A742","L":"#F4A742","A":"#F4A742","Y":"#F4A742",
        "F":"#E06AA3","U":"#E06AA3","N":"#E06AA3","K":"#E06AA3",
        "T":"#8E63D9","I":"#8E63D9","M":"#8E63D9","E":"#8E63D9",
        "DISCO":"#1D4ED8","VIP":"#E63946","STAYINALIVE":"#2DD4BF"
    }
    text = {
        "BAR":"#FFD700","DISCO":"#FFFFFF","VIP":"#FFFFFF","STAYINALIVE":"#FFFFFF",
        "T":"#FFFFFF","I":"#FFFFFF","M":"#FFFFFF","E":"#FFFFFF"
    }
    html = '<div style="display:flex;flex-direction:column;gap:8px">'
    for r in rows:
        html += '<div style="display:flex;gap:8px">'
        for s,p in r:
            html += (
                f'<div style="width:120px;padding:10px;border-radius:10px;'
                f'background:{color.get(s,"#ddd")};color:{text.get(s,"#000")};'
                f'text-align:center;font-weight:700">'
                f'{s}<br><span style="font-weight:500">{p}</span></div>'
            )
        html += '</div>'
    html += '</div>'
    return html

st.markdown(board_html(rows), unsafe_allow_html=True)

# ===== Eyes Eagle =====
st.subheader("Eyes Eagle – تنبيهات 15 رمية")
def color_signal(s):
    if s=="STOP":   return '<span style="background:#E63946;color:#fff;padding:3px 8px;border-radius:6px">STOP</span>'
    if s=="BIG":    return '<span style="background:#2A9D8F;color:#fff;padding:3px 8px;border-radius:6px">BIG</span>'
    if s=="MEDIUM": return '<span style="background:#F4A742;color:#000;padding:3px 8px;border-radius:6px">MEDIUM</span>'
    return ""

eyes2 = eyes.copy()
eyes2["Value %"] = eyes2["Value"].apply(lambda v: "" if pd.isna(v) else f"{v*100:.1f}%")
eyes2 = eyes2.drop(columns=["Value"]).rename(columns={"Value %":"Value"})
eyes2["Signal"] = eyes2["Signal"].apply(color_signal)
st.markdown(eyes2.to_html(index=False, escape=False), unsafe_allow_html=True)

# ===== Raw Data =====
st.subheader("Data (آخر الرميات داخل النافذة)")
st.dataframe(win.tail(100), use_container_width=True)
