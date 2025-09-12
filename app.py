# app.py — Funky Brain LIVE (Design + Recency/Temperature + Table)

import math
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ===== محاولات استيراد دوالك الأصلية إن وُجدت (لن نكسر الأساس) =====
_HAS_CORE = False
try:
    from funkybrain_core import normalize_df, compute_probs, board_model
    _HAS_CORE = True
except Exception:
    _HAS_CORE = False

# ====================== إعدادات عامة ======================
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain — LIVE")

# ألوان البلاطات
COLORS = {
    "ONE": "#F4D36B", "BAR": "#5AA64F",
    "ORANGE": "#E7903C", "PINK": "#C85C8E", "PURPLE": "#9A5BC2",
    "STAYINALIVE": "#4FC3D9", "DISCO": "#314E96", "DISCO_VIP": "#B03232",
}
LETTER_GROUP = {
    "P":"ORANGE","L":"ORANGE","A":"ORANGE","Y":"ORANGE",
    "F":"PINK","U":"PINK","N":"PINK","K":"PINK","Y2":"PINK",
    "T":"PURPLE","I":"PURPLE","M":"PURPLE","E":"PURPLE",
}
GRID_LETTERS = [
    ["1","BAR"],
    ["P","L","A","Y"],
    ["F","U","N","K","Y2"],
    ["T","I","M","E"],
    ["DISCO","STAYINALIVE","DISCO_VIP"]
]

BONUS_SEGMENTS = {"DISCO","STAYINALIVE","DISCO_VIP","BAR"}
ALL_SEGMENTS = {
    "1","BAR",
    "P","L","A","Y","F","U","N","K","Y","T","I","M","E",
    "DISCO","STAYINALIVE","DISCO_VIP"
}
# ترتيب العرض في الجدول
ORDER = ["1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]

# ========== أحجام البلاطات ==========
TILE_H=96; TILE_TXT=38; TILE_SUB=13
TILE_H_SMALL=84; TILE_TXT_SMALL=32; TILE_SUB_SMALL=12
TILE_TXT_BONUS=20

# ====================== وظائف مساعدة ======================

@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window):
    df=None
    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df=pd.read_csv(file)
            else:
                df=pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    if df is None and sheet_url:
        url=sheet_url.strip()
        if "docs.google.com/spreadsheets" in url and "export?format=csv" not in url:
            try: gid=url.split("gid=")[-1]
            except Exception: gid="0"
            doc_id=url.split("/d/")[1].split("/")[0]
            url=f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"
        try:
            df=pd.read_csv(url)
        except Exception as e:
            st.error(f"تعذّر تحميل Google Sheets: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    if df is None: return pd.DataFrame(columns=["ts","segment","multiplier"])

    wanted=["ts","segment","multiplier"]
    for c in wanted:
        if c not in df.columns:
            st.error(f"❗ عمود مفقود في الجدول: {c}")
            return pd.DataFrame(columns=wanted)

    try: df["ts"]=pd.to_datetime(df["ts"])
    except Exception: pass

    df["multiplier"]=(df["multiplier"].astype(str)
                      .str.extract(r"(\d+)\s*[xX]?",expand=False).fillna("1")
                      .astype(int).astype(str)+"X")

    if len(df)>window: df=df.tail(window).copy()
    df["segment"]=df["segment"].astype(str).str.upper()
    return df[["ts","segment","multiplier"]].reset_index(drop=True)


def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    dfx=df[~df["segment"].eq("UNKNOWN")].copy()
    if dfx.empty: dfx=df.copy()
    segs=list(ALL_SEGMENTS)
    n=len(dfx)
    if n==0:
        vec=np.ones(len(segs),dtype=float)
    else:
        ages=np.arange(n,0,-1)
        half=max(int(decay_half_life),1)
        w=np.power(0.5,(ages-1)/half); w=w/w.sum()
        counts={s:0.0 for s in segs}
        for seg,wt in zip(dfx["segment"],w):
            if seg in counts: counts[seg]+=wt
        vec=np.array([counts[s] for s in segs],dtype=float)
    for i,s in enumerate(segs):
        if s in BONUS_SEGMENTS: vec[i]*=float(bonus_boost)
    if vec.sum()<=0: vec[:]=1.0
    x=vec/(vec.std()+1e-9); x=x/max(float(temperature),1e-6)
    z=np.exp(x-x.max()); p_next=z/z.sum()
    probs=dict(zip(segs,p_next))
    p_in10={s:1.0-(1.0-probs[s])**horizon for s in segs}
    return probs,p_in10

def fallback_naive(df,horizon=10):
    counts=df["segment"].value_counts()
    segs=list(ALL_SEGMENTS)
    vec=np.array([counts.get(s,0) for s in segs],dtype=float)
    if vec.sum()==0: vec[:]=1.0
    z=np.exp((vec-vec.mean())/(vec.std()+1e-6)); p=z/z.sum()
    probs=dict(zip(segs,p))
    prob_in10={s:1.0-(1.0-probs[s])**horizon for s in segs}
    return probs,prob_in10

def get_probs(df,horizon=10,temperature=1.6,decay_half_life=60,bonus_boost=1.15):
    if _HAS_CORE:
        try:
            dfn=normalize_df(df)
            comp=compute_probs(dfn,horizon=horizon)
            p_next=comp.get("p_next",{}); p_in10=comp.get("p_in10",{})
            if len(p_next)==0 or len(p_in10)==0: raise ValueError
            return p_next,p_in10
        except Exception:
            pass
    try:
        return recency_softmax_probs(df,horizon,temperature,decay_half_life,bonus_boost)
    except Exception:
        return fallback_naive(df,horizon=horizon)

def pct(x): return f"{x*100:.1f}%"

def letter_color(letter):
    if letter in {"1","ONE"}: return COLORS["ONE"]
    if letter=="BAR": return COLORS["BAR"]
    if letter in {"P","L","A","Y"}: return COLORS[LETTER_GROUP[letter]]
    if letter in {"F","U","N","K","Y","Y2"}: return COLORS["PINK"]
    if letter in {"T","I","M","E"}: return COLORS["PURPLE"]
    if letter=="STAYINALIVE": return COLORS["STAYINALIVE"]
    if letter=="DISCO": return COLORS["DISCO"]
    if letter=="DISCO_VIP": return COLORS["DISCO_VIP"]
    return "#444"

def display_tile(label, subtext, bg, height=TILE_H, radius=16, txt_size=TILE_TXT, sub_size=TILE_SUB):
    st.markdown(
        f"""
        <div style="background:{bg};color:white;border-radius:{radius}px;height:{height}px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:700;">
            <div style="font-size:{txt_size}px;line-height:1">{label if label!='Y2' else 'Y'}</div>
            <div style="font-size:{sub_size}px;opacity:.95;margin-top:2px">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)

def section_header(title):
    st.markdown(f"<div style='font-size:20px;font-weight:700;margin:6px 0 10px'>{title}</div>", unsafe_allow_html=True)

# ====================== الواجهة ======================

with st.sidebar:
    st.subheader("⚙️ الإعدادات")
    window = st.slider("Window size (spins)", 50, 300, 120, step=10)
    horizon = st.slider("توقع على كم جولة؟", 5, 20, 10, step=1)
    st.write("---")
    st.subheader("🎛️ معلمات التنبؤ (Recency/Softmax)")
    temperature = st.slider("Temperature (تركيز السوفت-ماكس)", 1.0, 2.5, 1.6, 0.1)
    decay_half_life = st.slider("Half-life (ترجيح الحداثة)", 20, 120, 60, 5)
    bonus_boost = st.slider("تعزيز البونص", 1.00, 1.40, 1.15, 0.05)
    st.write("---")
    st.subheader("📥 مصدر البيانات")
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

# تحميل الداتا
df = load_data(upload, sheet_url, window)
if df.empty:
    st.info("أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")
    st.stop()

p_next, p_in10 = get_probs(df, horizon=horizon, temperature=temperature,
                           decay_half_life=decay_half_life, bonus_boost=bonus_boost)

# أدوات رياضيات بسيطة
def p_at_least_once(p, n): return 1.0 - (1.0 - p)**n
def exp_count(p, n): return n * p

# تبويبات
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(
    ["🎛️ Tiles", "🎯 Board + 10 Spins", "📊 Table", "🦅 Falcon Eye"]
)

# ========== تبويب البلاطات ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")
    c1,c2,_,_=st.columns([1.2,1.2,0.1,0.1])
    with c1: display_tile("1", f"P(next) {pct(p_next.get('1',0))}", letter_color("1"))
    with c2: display_tile("BAR", f"P(next) {pct(p_next.get('BAR',0))}", letter_color("BAR"), txt_size=34)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["P","L","A","Y"]):
        with cols[i]: display_tile(L, f"P(next) {pct(p_next.get(L,0))}", letter_color(L))
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(5)
    for i,L in enumerate(["F","U","N","K","Y2"]):
        key="Y" if L=="Y2" else L
        with cols[i]: display_tile(key, f"P(next) {pct(p_next.get(key,0))}", letter_color(L))
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["T","I","M","E"]):
        with cols[i]: display_tile(L, f"P(next) {pct(p_next.get(L,0))}", letter_color(L))
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(3)
    for i,B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        with cols[i]:
            display_tile("VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO"),
                         f"P(next) {pct(p_next.get(B,0))}", letter_color(B),
                         height=TILE_H, txt_size=TILE_TXT_BONUS)

# ========== تبويب اللوحة + 10 ==========
with tab_board:
    section_header("لوحة الرهان + توقع الظهور خلال 10 جولات")
    st.caption("النسبة هي احتمال الظهور مرة واحدة على الأقل خلال الجولات العشر القادمة.")
    def prob10(seg): return pct(p_at_least_once(p_next.get(seg,0), 10))
    c1,c2=st.columns(2)
    with c1: display_tile("1", f"≥1 in 10: {prob10('1')}", letter_color("1"),
                          height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    with c2: display_tile("BAR", f"≥1 in 10: {prob10('BAR')}", letter_color("BAR"),
                          height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["P","L","A","Y"]):
        with cols[i]: display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                                   height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(5)
    for i,L in enumerate(["F","U","N","K","Y"]):
        with cols[i]: display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L if L!="Y" else "Y2"),
                                   height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["T","I","M","E"]):
        with cols[i]: display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                                   height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(3)
    for i,B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label="VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]: display_tile(label, f"≥1 in 10: {prob10(B)}", letter_color(B),
                                   height=TILE_H_SMALL, txt_size=TILE_TXT_BONUS, sub_size=TILE_SUB_SMALL)

# ========== تبويب الجدول ==========
with tab_table:
    section_header("📊 جدول التكهّنات")
    rows=[]
    for s in ORDER:
        p=p_next.get(s,0.0)
        rows.append({
            "Segment": "VIP DISCO" if s=="DISCO_VIP" else ("STAYIN'ALIVE" if s=="STAYINALIVE" else s),
            "≥1 in 10": p_at_least_once(p,10),
            "≥1 in 15": p_at_least_once(p,15),
            "≥1 in 25": p_at_least_once(p,25),
            "Exp in 15": exp_count(p,15),
            "_color": letter_color("Y2" if s=="Y" else s),
        })
    tdf=pd.DataFrame(rows)

    # تنسيق ألوان و النِّسَب
    def _fmt(v, col):
        return f"{v*100:.1f}%" if col in {"≥1 in 10","≥1 in 15","≥1 in 25"} else (f"{v:.2f}" if col=="Exp in 15" else v)

    styled = (tdf.drop(columns=["_color"])
              .style.format({c:(lambda v, c=c: _fmt(v,c)) for c in ["≥1 in 10","≥1 in 15","≥1 in 25","Exp in 15"]})
              .apply(lambda s: [f"background-color: {tdf.loc[i,'_color']}; color: white; font-weight:700"
                                if s.name=="Segment" else "" for i in range(len(s))], axis=0))
    st.dataframe(styled, use_container_width=True)

# ========== تبويب عين الصقر ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")

    # احتمالات ظهور أي بونص مرة واحدة على الأقل (10/15/25)
    p_any_bonus_10 = 1.0
    p_any_bonus_15 = 1.0
    p_any_bonus_25 = 1.0
    for b in BONUS_SEGMENTS:
        p = p_next.get(b,0.0)
        p_any_bonus_10 *= (1.0 - p)**10
        p_any_bonus_15 *= (1.0 - p)**15
        p_any_bonus_25 *= (1.0 - p)**25
    p_any_bonus_10 = 1.0 - p_any_bonus_10
    p_any_bonus_15 = 1.0 - p_any_bonus_15
    p_any_bonus_25 = 1.0 - p_any_bonus_25

    c0,c1,c2 = st.columns(3)
    with c0:
        st.markdown(
            f"<div style='background:#1565C0;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 10: <span style='float:right'>{pct(p_any_bonus_10)}</span></div>",
            unsafe_allow_html=True
        )
    with c1:
        st.markdown(
            f"<div style='background:#00897B;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 15: <span style='float:right'>{pct(p_any_bonus_15)}</span></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div style='background:#6A1B9A;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 25: <span style='float:right'>{pct(p_any_bonus_25)}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # تقديرات ≥×50 / ≥×100 / أسطوري كما هي (تقريب)
    bonus10 = {b: (1.0 - (1.0 - p_next.get(b,0.0))**10) for b in BONUS_SEGMENTS}
    p50 = sum(bonus10.values()) * 0.25
    p100 = sum(bonus10.values()) * 0.10
    pLegend = sum(bonus10.values()) * 0.04

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background:#F8E16C;padding:14px;border-radius:14px;font-weight:700'>"
            f"🎁 بونص ≥ ×50 في 10: <span style='float:right'>{pct(p50)}</span></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div style='background:#61C16D;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"💎 بونص ≥ ×100 في 10: <span style='float:right'>{pct(p100)}</span></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div style='background:#7C4DFF;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🚀 بونص أسطوري (+100) في 10: <span style='float:right'>{pct(pLegend)}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # تغيُّر ديناميكي و High Risk كما كان
    Wmini=min(30,len(df))
    if Wmini>=10:
        tail=df.tail(Wmini)
        counts=tail["segment"].value_counts(normalize=True)
        meanp=counts.mean(); varp=((counts-meanp)**2).mean()
        if varp>0.005:
            change_label="High change"; badge="<span style='color:#D32F2F;font-weight:700'>HIGH</span>"
        elif varp>0.002:
            change_label="Medium change"; badge="<span style='color:#FB8C00;font-weight:700'>MEDIUM</span>"
        else:
            change_label="Low change"; badge="<span style='color:#2E7D32;font-weight:700'>LOW</span>"
    else:
        change_label="Not enough data"; badge="<span style='color:#999'>N/A</span>"
    st.markdown(
        f"<div style='background:#1E1E1E;color:#fff;padding:14px;border-radius:12px'>"
        f"🔎 التقلب العام: {change_label} — {badge}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    p1_next=p_next.get("1",0.0); p1_in15=p_at_least_once(p1_next,15)
    high_risk=p1_in15>0.85; color="#D32F2F" if high_risk else "#37474F"
    st.markdown(
        f"<div style='background:{color};color:#fff;padding:14px;border-radius:12px'>"
        f"⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>",
        unsafe_allow_html=True
    )
    st.caption("🔧 ملاحظة: عندما تفعّل نماذجك الخاصة سيُستبدل ما سبق بتقديراتك الدقيقة.")

# ========== أسفل الصفحة: معاينة الداتا ==========
with st.expander("عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(50), use_container_width=True)
