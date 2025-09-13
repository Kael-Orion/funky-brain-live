# app.py — Funky Brain LIVE (Stable + Data Cleaner + Model Switch)
# - يقرأ من data/combined_spins.csv أو رفع ملف/Google Sheets
# - تنظيف ذكي للبيانات الخام (ts/segment/multiplier) واستخراج segment من روابط الصور
# - زر تنظيف+إضافة للملف المرفوع + زر دمج ملفات data/spins_cleaned*.*
# - نموذج Recency+Softmax + خيار استخدام نموذج متعلّم pattern_model.pkl
# - تبويبات: Tiles / Board + 10 / Table / Falcon Eye
# - تحذير: احتمال تكرار "1" ≥ 3 مرات في 10 رميات

import os, re, math, pickle
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ===== محاولة استخدام دوالّك الأساسية إن وُجدت (لن نكسر شيئًا) =====
_HAS_CORE = False
try:
    from funkybrain_core import normalize_df, compute_probs, board_model
    _HAS_CORE = True
except Exception:
    _HAS_CORE = False

# ------------------------ إعدادات عامة ------------------------
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain — LIVE")

DATA_DIR = "data"
REPO_COMBINED_PATH = os.path.join(DATA_DIR, "combined_spins.csv")

# ألوان البلاطات
COLORS = {
    "ONE": "#F4D36B", "BAR": "#5AA64F",
    "ORANGE": "#E7903C", "PINK": "#C85C8E", "PURPLE": "#9A5BC2",
    "STAYINALIVE": "#4FC3D9", "DISCO": "#314E96", "DISCO_VIP": "#B03232",
}
BONUS_SEGMENTS = {"DISCO","STAYINALIVE","DISCO_VIP","BAR"}
ALL_SEGMENTS = {
    "1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E",
    "DISCO","STAYINALIVE","DISCO_VIP"
}
ORDER = ["1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]

# أحجام البلاطات
TILE_H=96; TILE_TXT=38; TILE_SUB=13
TILE_H_SMALL=84; TILE_TXT_SMALL=32; TILE_SUB_SMALL=12
TILE_TXT_BONUS=20

# ------------------------ وظائف مساعدة ------------------------
def pct(x: float) -> str:
    try: return f"{float(x)*100:.1f}%"
    except Exception: return "0.0%"

def p_at_least_once(p: float, n: int) -> float:
    return 1.0 - (1.0 - float(p))**int(n)

def exp_count(p: float, n: int) -> float:
    return float(n) * float(p)

def letter_color(letter: str) -> str:
    if letter in {"1","ONE"}: return COLORS["ONE"]
    if letter=="BAR": return COLORS["BAR"]
    if letter in {"P","L","A","Y"}: return COLORS["ORANGE"]
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
        """, unsafe_allow_html=True
    )

def section_header(title):
    st.markdown(f"<div style='font-size:20px;font-weight:700;margin:6px 0 10px'>{title}</div>", unsafe_allow_html=True)

# ---------- تخمين القطاع من اسم ملف الصورة ----------
def _guess_segment_from_url(url: str) -> str | None:
    s = str(url).lower()
    patterns = [
        (r"(disco[_\-]?(vip|v|vip1)|discovip)", "DISCO_VIP"),
        (r"(stay[_\-]?(in)?alive|stayinalive|stayalive)", "STAYINALIVE"),
        (r"/bar(\.png|\.jpg|/|$)", "BAR"),
        (r"/disco(\.png|\.jpg|/|$)", "DISCO"),
        (r"/1(\.png|\.jpg|/|$)", "1"),
    ]
    for L in list("PLAYFUNKYTIME"):
        patterns.append((rf"/{L.lower()}(\.png|\.jpg|/|$)", L))
    for pat, lab in patterns:
        if re.search(pat, s):
            return lab
    return None

def _refine_unknown_with_url(df_raw: pd.DataFrame, seg_series: pd.Series, mult_series: pd.Series):
    url_col = None
    for c in ["image","img","src","url","href"]:
        if c in df_raw.columns:
            url_col = c; break
    if url_col is None:
        return seg_series, mult_series
    hints = df_raw[url_col].astype(str).apply(_guess_segment_from_url)
    mask = (seg_series.str.upper().eq("UNKNOWN")) & (mult_series.str.upper().eq("16X")) & (hints=="1")
    seg_series = seg_series.copy(); mult_series = mult_series.copy()
    seg_series.loc[mask] = "1"; mult_series.loc[mask] = "1X"
    return seg_series, mult_series

# ---------- منظّف مرن ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df_raw = df.copy()

    # ts
    ts_col = None
    for cand in ["ts", "timestamp", "datetime"]:
        if cand in df_raw.columns:
            ts_col = cand; break
    if ts_col is None and ("date" in df_raw.columns and "time" in df_raw.columns):
        df_raw["ts"] = (df_raw["date"].astype(str).str.strip() + " " + df_raw["time"].astype(str).str.strip())
        ts_col = "ts"
    if ts_col is None:
        for c in df_raw.columns:
            sample = str(df_raw[c].astype(str).dropna().head(6).tolist())
            if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", sample) or re.search(r"\d{1,2}:\d{2}", sample):
                ts_col = c; break
    if ts_col is None:
        raise ValueError("Column missing: ts")
    ts = pd.to_datetime(df_raw[ts_col], errors="coerce")

    # multiplier
    mult_col = "multiplier" if "multiplier" in df_raw.columns else None
    if mult_col is None:
        for c in df_raw.columns:
            vals = df_raw[c].astype(str)
            if (vals.str.contains(r"\d+\s*[xX]$", regex=True).mean() > 0.3) or \
               (vals.str.contains(r"^\s*\d+\s*(?:x|X)?\s*$", regex=True).mean() > 0.5):
                mult_col = c; break
    if mult_col is None:
        num_like = [c for c in df_raw.columns if pd.to_numeric(df_raw[c], errors="coerce").notna().mean() > 0.5]
        if num_like: mult_col = num_like[0]
        else: raise ValueError("Column missing: multiplier")
    mult = (
        df_raw[mult_col].astype(str)
        .str.extract(r"(\d+)\s*[xX]?", expand=False)
        .fillna("1").astype(int).astype(str) + "X"
    )

    # segment
    seg_col = "segment" if "segment" in df_raw.columns else None
    if seg_col is None:
        for c in ["symbol","result","letter","tile","name"]:
            if c in df_raw.columns:
                seg_col = c; break
    if seg_col is None:
        url_col = None
        for c in ["image","img","src","url","href"]:
            if c in df_raw.columns:
                url_col = c; break
        if url_col is not None:
            seg = df_raw[url_col].astype(str).apply(_guess_segment_from_url).fillna("UNKNOWN")
        else:
            seg = pd.Series(["UNKNOWN"]*len(df_raw))
    else:
        seg = df_raw[seg_col].astype(str).str.strip().str.upper()

    seg = seg.str.upper().replace({"ONE":"1"})
    mult = mult.where(~seg.eq("1"), "1X")
    seg, mult = _refine_unknown_with_url(df_raw, seg, mult)

    out = pd.DataFrame({"ts": ts, "segment": seg, "multiplier": mult})
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out[["ts","segment","multiplier"]]

# ---------- دمج ملفات داخل data/ ----------
def combine_inside_streamlit() -> tuple[int, str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = []
    for name in os.listdir(DATA_DIR):
        low = name.lower()
        if low.startswith("spins_cleaned") and low.endswith((".csv",".xlsx",".xls")):
            paths.append(os.path.join(DATA_DIR, name))
    if not paths:
        return 0, "لم يتم العثور على ملفات تبدأ بـ spins_cleaned داخل data/."
    frames = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
            frames.append(clean_df(df))
        except Exception as e:
            st.warning(f"تجاوز {os.path.basename(p)}: {e}")
    if not frames: return 0, "لم أستطع قراءة أي ملف صالح."
    big = pd.concat(frames, ignore_index=True)
    big = big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
    big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return len(big), f"تم الدمج في {REPO_COMBINED_PATH} — الصفوف: {len(big):,}"

# ---------- تنظيف ملف مرفوع وإلحاقه بـ combined_spins.csv ----------
def clean_and_append_uploaded(upload_file) -> pd.DataFrame:
    if upload_file is None:
        raise ValueError("لا يوجد ملف مرفوع.")
    raw = pd.read_csv(upload_file) if upload_file.name.lower().endswith(".csv") else pd.read_excel(upload_file)
    cleaned = clean_df(raw)

    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(REPO_COMBINED_PATH):
        # ✅ إصلاح التعارض: طبّق clean_df دومًا على الملف الموجود لتوحيد الأنواع (ts كـ datetime)
        base_raw = pd.read_csv(REPO_COMBINED_PATH)
        base = clean_df(base_raw)
        out = (pd.concat([base, cleaned], ignore_index=True)
                 .drop_duplicates(subset=["ts","segment","multiplier"])
                 .sort_values("ts").reset_index(drop=True))
    else:
        out = cleaned.copy()

    out.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return cleaned

# ---------- قراءة البيانات (repo / upload / sheets) ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    df = None
    if use_repo_file and os.path.exists(repo_path):
        try: df = pd.read_csv(repo_path)
        except Exception as e: st.warning(f"تعذر قراءة {repo_path}: {e}")

    if df is None and file is not None:
        try:
            df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    if df is None and sheet_url:
        url = sheet_url.strip()
        if "docs.google.com/spreadsheets" in url and "export?format=csv" not in url:
            try: gid = url.split("gid=")[-1]
            except Exception: gid = "0"
            doc_id = url.split("/d/")[1].split("/")[0]
            url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"
        try: df = pd.read_csv(url)
        except Exception as e:
            st.error(f"تعذّر تحميل Google Sheets: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    if df is None:
        return pd.DataFrame(columns=["ts","segment","multiplier"])

    try:
        df = clean_df(df)
    except Exception as e:
        st.error(f"تنسيق الجدول غير صالح: {e}")
        return pd.DataFrame(columns=["ts","segment","multiplier"])

    if len(df) > window: df = df.tail(window).copy()
    return df.reset_index(drop=True)

# -------- نموذج الاحتمالات: Recency + Softmax + Bonus boost --------
def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    try:
        dfx = df[~df["segment"].eq("UNKNOWN")].copy()
        if dfx.empty: dfx = df.copy()
        segs = list(ALL_SEGMENTS); n = len(dfx)
        if n == 0:
            vec = np.ones(len(segs), dtype=float)
        else:
            ages = np.arange(n, 0, -1)               # الأحدث عمره 1
            half = max(int(decay_half_life), 1)
            w = np.power(0.5, (ages-1)/half); w = w / w.sum()
            counts = {s:0.0 for s in segs}
            for seg, wt in zip(dfx["segment"], w):
                if seg in counts: counts[seg] += wt
            vec = np.array([counts[s] for s in segs], dtype=float)
        for i,s in enumerate(segs):
            if s in BONUS_SEGMENTS: vec[i] *= float(bonus_boost)
        if vec.sum() <= 0: vec[:] = 1.0
        x = vec / (vec.std()+1e-9); x = x / max(float(temperature),1e-6)
        z = np.exp(x-x.max()); p_next = z/z.sum()
        probs = dict(zip(segs,p_next))
        p_in10 = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_in10
    except Exception:
        counts = df["segment"].value_counts()
        segs = list(ALL_SEGMENTS)
        vec = np.array([counts.get(s,0) for s in segs], dtype=float)
        if vec.sum()==0: vec[:] = 1.0
        z = np.exp((vec-vec.mean())/(vec.std()+1e-6)); p = z/z.sum()
        probs = dict(zip(segs,p))
        p_in10 = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs,p_in10

def get_probs_recency(df, **kw):
    return recency_softmax_probs(df, **kw)

# ------------------------ الواجهة (Sidebar) ------------------------
with st.sidebar:
    st.subheader("⚙️ الإعدادات")
    window = st.slider("Window size (spins)", 50, 300, 120, step=10)
    horizon = st.slider("توقع على كم جولة؟", 5, 20, 10, step=1)
    st.write("---")

    st.subheader("🎛️ معلمات التنبؤ (Recency/Softmax)")
    temperature = st.slider("Temperature", 1.0, 2.5, 1.6, 0.1)
    decay_half_life = st.slider("Half-life", 20, 120, 60, 5)
    bonus_boost = st.slider("Bonus Boost", 1.00, 1.40, 1.15, 0.05)

    st.write("---")
    st.subheader("🧩 إدارة البيانات")

    if st.button("🔁 دمج ملفات data/spins_cleaned*.csv(xlsx) → combined_spins.csv", use_container_width=True):
        rows, msg = combine_inside_streamlit()
        if rows>0:
            st.success(msg)
            load_data.clear(); st.experimental_rerun()
        else:
            st.warning(msg)

    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH, "rb") as f:
            st.download_button("⬇️ تنزيل combined_spins.csv", f.read(), file_name="combined_spins.csv", mime="text/csv", use_container_width=True)

    st.write("---")
    st.subheader("📥 مصدر البيانات")
    use_repo_combined = st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url = st.text_input("رابط Google Sheets (CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

    if upload is not None:
        if st.button("🧹 تنظيف + إضافة إلى combined_spins.csv", use_container_width=True):
            try:
                preview = clean_and_append_uploaded(upload)
                st.success("تمّ تنظيف الملف وإلحاقه بـ combined_spins.csv ✅")
                with st.expander("معاينة بعد التنظيف (أول 20 صف)"):
                    st.dataframe(preview.head(20), use_container_width=True)
                load_data.clear(); st.experimental_rerun()
            except Exception as e:
                st.error(f"فشل تنظيف/إضافة الملف الخام: {e}")

    st.write("---")
    st.subheader("🤖 نموذج متعلّم (اختياري)")
    use_learned = st.toggle("استخدم النموذج المتعلّم إن وجد", value=False)
    model_path = st.text_input("مسار ملف النموذج", value="models/pattern_model.pkl")

# تحميل الداتا
df = load_data(upload if not use_repo_combined else None, sheet_url, window,
               use_repo_file=use_repo_combined, repo_path=REPO_COMBINED_PATH)
if df.empty:
    st.info("أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")
    st.stop()

# -------- اختيار مصدر الاحتمالات --------
source_label = "recency"; p_next={}; p_in10={}
if use_learned and os.path.exists(model_path):
    try:
        with open(model_path, "rb") as fh: model_obj = pickle.load(fh)
        if isinstance(model_obj, dict) and "p_next" in model_obj:
            p_next = model_obj["p_next"]
            p_in10 = {s: p_at_least_once(p_next.get(s,0.0), horizon) for s in ALL_SEGMENTS}
            source_label = "learned-model"
            with st.expander("إعدادات النموذج (meta)"):
                st.json(model_obj.get("meta", {}))
        else:
            st.warning("ملف النموذج لا يحتوي p_next — سيتم استخدام recency.")
    except Exception as e:
        st.warning(f"تعذر تحميل النموذج: {e}")
if not p_next:
    p_next, p_in10 = get_probs_recency(df, horizon=horizon, temperature=temperature, decay_half_life=decay_half_life, bonus_boost=bonus_boost)
st.caption(f"Source of probabilities: {source_label}")

# تبويبات
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(["🎛️ Tiles", "🎯 Board + 10 Spins", "📊 Table", "🦅 Falcon Eye"])

# ========== Tiles ==========
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

# ========== Board + 10 ==========
with tab_board:
    section_header("لوحة الرهان + توقع الظهور خلال 10 جولات")
    st.caption("النسبة أسفل كل خانة هي احتمال الظهور مرة واحدة على الأقل خلال الجولات العشر القادمة.")
    def prob10(seg): return pct(p_at_least_once(p_next.get(seg,0.0), 10))
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

# ========== Table ==========
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
            "_color": letter_color("Y2" if s=="Y" else s),
        })
    tdf=pd.DataFrame(rows)
    def _fmt(v, col):
        return f"{v*100:.1f}%" if col in {"≥1 in 10","≥1 in 15","≥1 in 25"} else (f"{v:.2f}" if col=="Exp in 15" else v)
    styled = (tdf.drop(columns=["_color"])
                .style.format({c:(lambda v, c=c: _fmt(v,c)) for c in ["≥1 in 10","≥1 in 15","≥1 in 25","Exp in 15"]})
                .apply(lambda s: [f"background-color: {tdf.loc[i,'_color']}; color: white; font-weight:700"
                                  if s.name=="Segment" else "" for i in range(len(s))], axis=0))
    st.dataframe(styled, use_container_width=True)

# ========== Falcon Eye ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")
    any10=any15=any25=1.0
    for b in BONUS_SEGMENTS:
        pb=p_next.get(b,0.0)
        any10*=(1.0-pb)**10; any15*=(1.0-pb)**15; any25*=(1.0-pb)**25
    any10=1.0-any10; any15=1.0-any15; any25=1.0-any25

    c0,c1,c2=st.columns(3)
    with c0: st.markdown(f"<div style='background:#1565C0;padding:14px;border-radius:14px;font-weight:700;color:white'>🎲 احتمال أي بونص ≥1 في 10: <span style='float:right'>{pct(any10)}</span></div>", unsafe_allow_html=True)
    with c1: st.markdown(f"<div style='background:#00897B;padding:14px;border-radius:14px;font-weight:700;color:white'>🎲 احتمال أي بونص ≥1 في 15: <span style='float:right'>{pct(any15)}</span></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div style='background:#6A1B9A;padding:14px;border-radius:14px;font-weight:700;color:white'>🎲 احتمال أي بونص ≥1 في 25: <span style='float:right'>{pct(any25)}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    bonus10={b: p_at_least_once(p_next.get(b,0.0),10) for b in BONUS_SEGMENTS}
    p50=sum(bonus10.values())*0.25; p100=sum(bonus10.values())*0.10; pLegend=sum(bonus10.values())*0.04
    d1,d2,d3=st.columns(3)
    with d1: st.markdown(f"<div style='background:#F8E16C;padding:14px;border-radius:14px;font-weight:700'>🎁 بونص ≥ ×50 في 10: <span style='float:right'>{pct(p50)}</span></div>", unsafe_allow_html=True)
    with d2: st.markdown(f"<div style='background:#61C16D;padding:14px;border-radius:14px;font-weight:700;color:white'>💎 بونص ≥ ×100 في 10: <span style='float:right'>{pct(p100)}</span></div>", unsafe_allow_html=True)
    with d3: st.markdown(f"<div style='background:#7C4DFF;padding:14px;border-radius:14px;font-weight:700;color:white'>🚀 بونص أسطوري (+100) في 10: <span style='float:right'>{pct(pLegend)}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    Wmini=min(30,len(df))
    if Wmini>=10:
        tail=df.tail(Wmini); counts=tail["segment"].value_counts(normalize=True)
        meanp=counts.mean(); varp=((counts-meanp)**2).mean()
        if varp>0.005: change_label="High change"; badge="<span style='color:#D32F2F;font-weight:700'>HIGH</span>"
        elif varp>0.002: change_label="Medium change"; badge="<span style='color:#FB8C00;font-weight:700'>MEDIUM</span>"
        else: change_label="Low change"; badge="<span style='color:#2E7D32;font-weight:700'>LOW</span>"
    else:
        change_label="Not enough data"; badge="<span style='color:#999'>N/A</span>"
    st.markdown(f"<div style='background:#1E1E1E;color:#fff;padding:14px;border-radius:12px'>🔎 التقلب العام: {change_label} — {badge}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    def binom_tail_ge_k(n,p,k):
        p=max(0.0,min(1.0,float(p))); total=0.0
        for r in range(0,k): total += math.comb(n,r)*(p**r)*((1-p)**(n-r))
        return 1.0-total
    p1_next=p_next.get("1",0.0); p1_in15=p_at_least_once(p1_next,15)
    color15="#D32F2F" if p1_in15>0.85 else "#37474F"
    st.markdown(f"<div style='background:{color15};color:#fff;padding:14px;border-radius:12px'>⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>", unsafe_allow_html=True)
    p1_ge3_in10 = binom_tail_ge_k(10, p1_next, 3)
    st.markdown(f"<div style='background:#B71C1C;color:#fff;padding:14px;border-radius:12px'>🛑 تنبيه حاد: احتمال تكرار الرقم <b>1</b> ثلاث مرات أو أكثر خلال 10 سبِن = <b>{pct(p1_ge3_in10)}</b> — يُنصح بالتوقف المؤقت.</div>", unsafe_allow_html=True)

# ========== أسفل الصفحة ==========
with st.expander("عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(50), use_container_width=True)

# ---------- تدريب النموذج من داخل التطبيق ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ تدريب النموذج (اختياري)")
model_path_input = st.sidebar.text_input("مسار حفظ النموذج", value="models/pattern_model.pkl")
with st.sidebar.expander("ملخص الداتا المستخدمة في التدريب"):
    st.write(f"عدد الرميات: **{len(df)}**"); st.write("أعمدة:", list(df.columns)); st.dataframe(df.tail(10), use_container_width=True)

def train_and_save_model(df, path, horizon, temperature, decay_half_life, bonus_boost):
    p_next, _ = recency_softmax_probs(df, horizon=horizon, temperature=temperature, decay_half_life=decay_half_life, bonus_boost=bonus_boost)
    model = {"type":"recency_softmax","p_next":p_next,"meta":{
        "horizon":horizon,"temperature":temperature,"half_life":decay_half_life,"bonus_boost":bonus_boost,
        "trained_on_rows":int(len(df)),"trained_at":datetime.utcnow().isoformat()+"Z"}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"wb") as f: pickle.dump(model,f)
    return model

if st.sidebar.button("💾 درِّب النموذج الآن", use_container_width=True):
    if df.empty: st.sidebar.error("لا توجد بيانات للتدريب.")
    else:
        try:
            _ = train_and_save_model(df, model_path_input, horizon, temperature, decay_half_life, bonus_boost)
            st.sidebar.success(f"تم حفظ النموذج: {model_path_input}")
            with open(model_path_input,"rb") as fh:
                st.sidebar.download_button("⬇️ تحميل النموذج", fh.read(), file_name="pattern_model.pkl", mime="application/octet-stream", use_container_width=True)
        except Exception as e:
            st.sidebar.error(f"فشل التدريب: {e}")

st.sidebar.caption("بعد تحميل pattern_model.pkl ارفعه إلى مجلد models/ في GitHub ليبقى دائمًا.")
