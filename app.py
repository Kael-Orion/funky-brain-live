# app.py — Funky Brain LIVE (V3.1)
import os, math, pickle
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain — LIVE")

DATA_DIR = "data"
REPO_COMBINED_PATH = os.path.join(DATA_DIR, "combined_spins.csv")

COLORS = {
    "ONE": "#F4D36B", "BAR": "#5AA64F",
    "ORANGE": "#E7903C", "PINK": "#C85C8E", "PURPLE": "#9A5BC2",
    "STAYINALIVE": "#4FC3D9", "DISCO": "#314E96", "DISCO_VIP": "#B03232",
}
LETTER_GROUP = {
    "P":"ORANGE","L":"ORANGE","A":"ORANGE","Y":"ORANGE",
    "F":"PINK","U":"PINK","N":"PINK","K":"PINK",
    "T":"PURPLE","I":"PURPLE","M":"PURPLE","E":"PURPLE",
}
BONUS_SEGMENTS = {"DISCO","STAYINALIVE","DISCO_VIP","BAR"}
ALL_SEGMENTS = {"1","BAR","P","L","A","Y","F","U","N","K","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"}
ORDER = ["1","BAR","P","L","A","Y","F","U","N","K","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]

TILE_H=96; TILE_TXT=38; TILE_SUB=13
TILE_H_SMALL=84; TILE_TXT_SMALL=32; TILE_SUB_SMALL=12
TILE_TXT_BONUS=20

def pct(x): 
    try: return f"{float(x)*100:.1f}%"
    except: return "0.0%"

def p_at_least_once(p,n): return 1-(1-float(p))**int(n)
def exp_count(p,n): return float(n)*float(p)

def letter_color(s):
    if s in ("1","ONE"): return COLORS["ONE"]
    if s=="BAR": return COLORS["BAR"]
    if s in {"P","L","A","Y"}: return COLORS["ORANGE"]
    if s in {"F","U","N","K"}: return COLORS["PINK"]
    if s in {"T","I","M","E"}: return COLORS["PURPLE"]
    if s=="STAYINALIVE": return COLORS["STAYINALIVE"]
    if s=="DISCO": return COLORS["DISCO"]
    if s=="DISCO_VIP": return COLORS["DISCO_VIP"]
    return "#444"

def display_tile(label, subtext, bg, height=TILE_H, radius=16, txt_size=TILE_TXT, sub_size=TILE_SUB):
    st.markdown(
        f"""
        <div style="background:{bg};color:white;border-radius:{radius}px;height:{height}px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:700;">
            <div style="font-size:{txt_size}px;line-height:1">{label}</div>
            <div style="font-size:{sub_size}px;opacity:.95;margin-top:2px">{subtext}</div>
        </div>""", unsafe_allow_html=True)

def section_header(t):
    st.markdown(f"<div style='font-size:20px;font-weight:700;margin:6px 0 10px'>{t}</div>", unsafe_allow_html=True)

# ---------- التنظيف ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    # ts
    guess_ts=None
    for k in ["ts","time","timestamp","date","datetime"]:
        if k in cols_lower: guess_ts=cols_lower[k]; break
    if guess_ts is None: 
        raise ValueError("Column missing: ts")
    # segment
    guess_seg=None
    for k in ["segment","result","tile","symbol","letter","outcome"]:
        if k in cols_lower: guess_seg=cols_lower[k]; break
    if guess_seg is None:
        for c in df.columns:
            if df[c].dtype==object:
                m = df[c].astype(str).str.extract(r"/([A-Za-z0-9_]+)\.png", expand=False)
                if m.notna().any():
                    df["segment"]=m.str.upper().str.replace("DISCOVIP","DISCO_VIP").str.replace("VIPDISCO","DISCO_VIP")
                    guess_seg="segment"; break
        if guess_seg is None: raise ValueError("Column missing: segment")
    # multiplier
    guess_mul=None
    for k in ["multiplier","multi","x","payout","winmult","factor"]:
        if k in cols_lower: guess_mul=cols_lower[k]; break
    if guess_mul is None:
        for c in df.columns:
            if df[c].dtype==object:
                m=df[c].astype(str).str.extract(r"(\d+)\s*[xX]", expand=False)
                if m.notna().any():
                    df["multiplier"]=m.fillna("1"); guess_mul="multiplier"; break
        if guess_mul is None: raise ValueError("Column missing: multiplier")

    df["ts"]=pd.to_datetime(df[guess_ts], errors="coerce")
    seg=df[guess_seg].astype(str).str.strip().str.upper()
    seg=seg.replace({"VIP DISCO":"DISCO_VIP","DISCO VIP":"DISCO_VIP","DISCOVIP":"DISCO_VIP",
                     "STAYIN'ALIVE":"STAYINALIVE","STAYIN_ALIVE":"STAYINALIVE","STAYIN-ALIVE":"STAYINALIVE"})
    seg=seg.str.replace(r"[^A-Z0-9_]","",regex=True)
    mul=(df[guess_mul].astype(str).str.extract(r"(\d+)\s*[xX]?", expand=False).fillna("1").astype(int).astype(str)+"X")
    out=pd.DataFrame({"ts":df["ts"],"segment":seg,"multiplier":mul}).dropna(subset=["ts"]).reset_index(drop=True)
    is_unknown=~out["segment"].isin(list(ALL_SEGMENTS))
    out.loc[is_unknown & out["multiplier"].eq("1X"),"segment"]="1"
    out.loc[~out["segment"].isin(list(ALL_SEGMENTS)),"segment"]="UNKNOWN"
    out=out.sort_values("ts")
    return out[["ts","segment","multiplier"]]

# ---------- دمج ملفات data/spins_cleaned*.{csv,xlsx} ----------
def combine_inside_streamlit():
    os.makedirs(DATA_DIR, exist_ok=True)
    paths=[]
    for name in os.listdir(DATA_DIR):
        l=name.lower()
        if l.startswith("spins_cleaned") and (l.endswith(".csv") or l.endswith(".xlsx") or l.endswith(".xls")):
            paths.append(os.path.join(DATA_DIR,name))
    if not paths: return 0,"لم يتم العثور على أي ملفات تبدأ بـ spins_cleaned داخل data/."
    frames=[]
    for p in sorted(paths):
        try:
            df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
            frames.append(clean_df(df))
        except Exception as e:
            st.warning(f"تجاوز {os.path.basename(p)}: {e}")
    if not frames: return 0,"لم يُحمَّل أي ملف صالح."
    big=pd.concat(frames, ignore_index=True)
    big=big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
    big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return len(big), f"تم الدمج في {REPO_COMBINED_PATH} — إجمالي الصفوف: {len(big):,}"

# ---------- تحميل البيانات ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    df=None
    if use_repo_file and os.path.exists(repo_path):
        try: df=pd.read_csv(repo_path)
        except Exception as e: st.warning(f"تعذر قراءة {repo_path}: {e}")
    if df is None and file is not None:
        try:
            df=pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])
    if df is None and sheet_url:
        url=sheet_url.strip()
        if "docs.google.com/spreadsheets" in url and "export?format=csv" not in url:
            try: gid=url.split("gid=")[-1]
            except: gid="0"
            doc_id=url.split("/d/")[1].split("/")[0]
            url=f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"
        try: df=pd.read_csv(url)
        except Exception as e:
            st.error(f"تعذّر تحميل Google Sheets: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])
    if df is None:
        return pd.DataFrame(columns=["ts","segment","multiplier"])
    try: df=clean_df(df)
    except Exception as e:
        st.error(f"تنسيق الجدول غير صالح: {e}")
        return pd.DataFrame(columns=["ts","segment","multiplier"])
    if len(df)>window: df=df.tail(window).copy()
    return df.reset_index(drop=True)

# ---------- نموذج Recency/Softmax ----------
def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    dfx=df[~df["segment"].eq("UNKNOWN")].copy()
    if dfx.empty: dfx=df.copy()
    segs=list(ALL_SEGMENTS); n=len(dfx)
    if n==0:
        vec=np.ones(len(segs))
    else:
        ages=np.arange(n,0,-1)
        half=max(int(decay_half_life),1)
        w=np.power(0.5,(ages-1)/half); w=w/w.sum()
        counts={s:0.0 for s in segs}
        for seg,wt in zip(dfx["segment"], w):
            if seg in counts: counts[seg]+=wt
        vec=np.array([counts[s] for s in segs], dtype=float)
    for i,s in enumerate(segs):
        if s in BONUS_SEGMENTS: vec[i]*=float(bonus_boost)
    if vec.sum()<=0: vec[:]=1.0
    x=vec/(vec.std()+1e-9); x=x/max(float(temperature),1e-6)
    z=np.exp(x-x.max()); p_next=z/z.sum()
    probs=dict(zip(segs,p_next))
    p_in10={s:p_at_least_once(probs[s], horizon) for s in segs}
    return probs,p_in10

def get_probs(df,horizon=10,temperature=1.6,decay_half_life=60,bonus_boost=1.15):
    return recency_softmax_probs(df,horizon,temperature,decay_half_life,bonus_boost)

# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("⚙️ الإعدادات")
    window=st.slider("Window size (spins)",50,5000,120,step=10)
    horizon=st.slider("توقع على كم جولة؟",5,20,10,1)
    st.write("---")
    st.subheader("🎛️ معلمات التنبؤ (Recency/Softmax)")
    temperature=st.slider("Temperature (تركيز السوفت-ماكس)",1.0,2.5,1.6,0.1)
    decay_half_life=st.slider("Half-life (ترجيح الحداثة)",20,120,60,5)
    bonus_boost=st.slider("تعزيز البونص",1.00,1.40,1.15,0.05)
    st.write("---")

    st.subheader("🧩 إدارة البيانات")
    if st.button("🔁 دمج ملفات data/spins_cleaned*.csv(xlsx) → combined_spins.csv"):
        rows,msg=combine_inside_streamlit()
        if rows>0:
            st.success(msg); load_data.clear(); st.experimental_rerun()
        else: st.warning(msg)

    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH,"rb") as f:
            st.download_button("⬇️ تنزيل combined_spins.csv", f.read(), file_name="combined_spins.csv", mime="text/csv")

    st.write("---")
    st.subheader("📥 مصدر البيانات")
    use_repo_combined=st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url=st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload=st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

    # 🧹 زر التنظيف المباشر للملف المرفوع
    if upload is not None:
        if st.button("🧹 تنظيف + إضافة إلى combined_spins.csv", use_container_width=True):
            try:
                raw = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
                cleaned = clean_df(raw)
                st.success(f"تم تنظيف الملف — صفوف صالحة: {len(cleaned)}")
                st.dataframe(cleaned.tail(15), use_container_width=True)
                os.makedirs(DATA_DIR, exist_ok=True)
                if os.path.exists(REPO_COMBINED_PATH):
                    base = pd.read_csv(REPO_COMBINED_PATH)
                    base = pd.concat([base, cleaned], ignore_index=True)
                else:
                    base = cleaned.copy()
                base = (base.drop_duplicates(subset=["ts","segment","multiplier"])
                             .sort_values("ts").reset_index(drop=True))
                base.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
                st.success(f"أُضيف إلى {REPO_COMBINED_PATH}. الإجمالي الآن: {len(base):,}")
                load_data.clear()
            except Exception as e:
                st.error(f"فشل التنظيف/الإضافة: {e}")

# تحميل البيانات النهائية للاستخدام
df = load_data(upload, sheet_url, window, use_repo_file=use_repo_combined, repo_path=REPO_COMBINED_PATH)
if df.empty:
    st.info("أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")
    st.stop()

# -------- نموذج متعلّم (اختياري) --------
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 نموذج متعلّم (اختياري)")
    use_learned = st.toggle("استخدم النموذج المتعلَّم إن وُجد", value=False)
    model_path_to_use = st.text_input("مسار ملف النموذج", value="models/pattern_model.pkl", key="use_model_path")
    learned_pnext=None; model_meta=None
    if use_learned:
        try:
            with open(model_path_to_use,"rb") as f:
                model_obj=pickle.load(f)
            learned_pnext=model_obj.get("p_next",None)
            model_meta=model_obj.get("meta",{})
            if not isinstance(learned_pnext,dict) or len(learned_pnext)==0:
                st.error("النموذج المُحمّل لا يحتوي p_next صالح."); learned_pnext=None
        except Exception as e:
            st.error(f"تعذّر تحميل النموذج: {e}")
        if model_meta:
            with st.expander("إعدادات النموذج (meta)"):
                st.json(model_meta)

if use_learned and learned_pnext:
    p_next={s:float(learned_pnext.get(s,0.0)) for s in ALL_SEGMENTS}
else:
    p_next,_=recency_softmax_probs(df, horizon=10, temperature=temperature, decay_half_life=decay_half_life, bonus_boost=bonus_boost)

# -------- Tabs --------
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(["🎛️ Tiles","🎯 Board + 10 Spins","📊 Table","🦅 Falcon Eye"])

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
    cols=st.columns(4)
    for i,L in enumerate(["F","U","N","K"]):
        with cols[i]: display_tile(L, f"P(next) {pct(p_next.get(L,0))}", letter_color(L))
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["T","I","M","E"]):
        with cols[i]: display_tile(L, f"P(next) {pct(p_next.get(L,0))}", letter_color(L))
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(3)
    for i,B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label="VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]:
            display_tile(label, f"P(next) {pct(p_next.get(B,0))}", letter_color(B), height=TILE_H, txt_size=TILE_TXT_BONUS)

with tab_board:
    section_header("لوحة الرهان + توقع الظهور خلال 10 جولات")
    def prob10(seg): return pct(p_at_least_once(p_next.get(seg,0.0),10))
    c1,c2=st.columns(2)
    with c1: display_tile("1", f"≥1 in 10: {prob10('1')}", letter_color("1"), height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    with c2: display_tile("BAR", f"≥1 in 10: {prob10('BAR')}", letter_color("BAR"), height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["P","L","A","Y"]):
        with cols[i]: display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L), height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["F","U","N","K"]):
        with cols[i]: display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L), height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(4)
    for i,L in enumerate(["T","I","M","E"]):
        with cols[i]: display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L), height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols=st.columns(3)
    for i,B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label="VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]: display_tile(label, f"≥1 in 10: {prob10(B)}", letter_color(B), height=TILE_H_SMALL, txt_size=TILE_TXT_BONUS, sub_size=TILE_SUB_SMALL)

with tab_table:
    section_header("📊 جدول التكهّنات (10/15/25 و Exp in 15)")
    rows=[]
    for s in ORDER:
        p=p_next.get(s,0.0)
        rows.append({
            "Segment": "VIP DISCO" if s=="DISCO_VIP" else ("STAYIN'ALIVE" if s=="STAYINALIVE" else s),
            "≥1 in 10": p_at_least_once(p,10),
            "≥1 in 15": p_at_least_once(p,15),
            "≥1 in 25": p_at_least_once(p,25),
            "Exp in 15": exp_count(p,15),
            "_color": letter_color("1" if s=="1" else s),
        })
    tdf=pd.DataFrame(rows)

    def _fmt(v,col):
        return f"{v*100:.1f}%" if col in {"≥1 in 10","≥1 in 15","≥1 in 25"} else (f"{v:.2f}" if col=="Exp in 15" else v)

    base = tdf.drop(columns=["_color"]).style.format({c:(lambda v,c=c:_fmt(v,c)) for c in ["≥1 in 10","≥1 in 15","≥1 in 25","Exp in 15"]})
    # تلوين عمود Segment صفاً بصف:
    def color_segment_col(col):
        if col.name!="Segment": return [""]*len(col)
        return [f"background-color: {tdf.loc[i,'_color']}; color: white; font-weight:700" for i in range(len(col))]
    styled = base.apply(color_segment_col, axis=0)
    st.dataframe(styled, use_container_width=True)

with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")
    any10=1.0; any15=1.0; any25=1.0
    for b in BONUS_SEGMENTS:
        pb=p_next.get(b,0.0)
        any10*=(1-pb)**10; any15*=(1-pb)**15; any25*=(1-pb)**25
    any10=1-any10; any15=1-any15; any25=1-any25
    c0,c1,c2=st.columns(3)
    with c0: st.markdown(f"<div style='background:#1565C0;padding:14px;border-radius:14px;font-weight:700;color:white'>🎲 احتمال أي بونص ≥1 في 10: <span style='float:right'>{pct(any10)}</span></div>", unsafe_allow_html=True)
    with c1: st.markdown(f"<div style='background:#00897B;padding:14px;border-radius:14px;font-weight:700;color:white'>🎲 احتمال أي بونص ≥1 في 15: <span style='float:right'>{pct(any15)}</span></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div style='background:#6A1B9A;padding:14px;border-radius:14px;font-weight:700;color:white'>🎲 احتمال أي بونص ≥1 في 25: <span style='float:right'>{pct(any25)}</span></div>", unsafe_allow_html=True)

    def binom_tail_ge_k(n,p,k):
        p=max(0,min(1,float(p))); tot=0.0
        for r in range(0,k):
            tot+=math.comb(n,r)*(p**r)*((1-p)**(n-r))
        return 1.0-tot
    p1_next=p_next.get("1",0.0)
    p1_in15=p_at_least_once(p1_next,15)
    color15="#D32F2F" if p1_in15>0.85 else "#37474F"
    st.markdown(f"<div style='background:{color15};color:#fff;padding:14px;border-radius:12px'>⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>", unsafe_allow_html=True)
    p1_ge3_in10=binom_tail_ge_k(10,p1_next,3)
    st.markdown(f"<div style='background:#B71C1C;color:#fff;padding:14px;border-radius:12px'>🛑 تنبيه حاد: احتمال أن يتكرر الرقم <b>1</b> ثلاث مرات أو أكثر خلال 10 سبِن = <b>{pct(p1_ge3_in10)}</b> — يُنصح بالتوقف المؤقت.</div>", unsafe_allow_html=True)

with st.expander("عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(50), use_container_width=True)

# -------- تدريب النموذج --------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 تدريب النموذج (اختياري)")
model_path_input=st.sidebar.text_input("مسار حفظ النموذج", value="models/pattern_model.pkl", key="train_model_path")
with st.sidebar.expander("ملخص الداتا المستخدمة في التدريب"):
    st.write(f"عدد الرميات في النافذة الحالية: **{len(df)}**")
    st.write("أعمدة:", list(df.columns))
    st.dataframe(df.tail(10), use_container_width=True)

def train_and_save_model(df, path, horizon, temperature, decay_half_life, bonus_boost):
    p_next_tr,_=recency_softmax_probs(df,horizon,temperature,decay_half_life,bonus_boost)
    model={"type":"recency_softmax","p_next":p_next_tr,"meta":{"horizon":horizon,"temperature":temperature,
           "half_life":decay_half_life,"bonus_boost":bonus_boost,"trained_on_rows":int(len(df)),
           "trained_at":datetime.utcnow().isoformat()+"Z"}}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"wb") as f: pickle.dump(model,f)
    return model

if st.sidebar.button("💾 درِّب النموذج الآن", use_container_width=True):
    if df.empty:
        st.sidebar.error("لا توجد بيانات للتدريب.")
    else:
        try:
            _=train_and_save_model(df, model_path_input, horizon, temperature, decay_half_life, bonus_boost)
            st.sidebar.success(f"تم حفظ النموذج: {model_path_input}")
            with open(model_path_input,"rb") as fh:
                st.sidebar.download_button("⬇️ تحميل النموذج", fh.read(),
                    file_name=os.path.basename(model_path_input) or "pattern_model.pkl",
                    mime="application/octet-stream", use_container_width=True)
        except Exception as e:
            st.sidebar.error(f"فشل التدريب: {e}")
st.sidebar.markdown("---")
st.sidebar.caption("بعد تحميل ملف النموذج ارفعه إلى مجلد models/ في GitHub.")
