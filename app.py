# app.py — Funky Brain LIVE
# - يقرأ من data/combined_spins.csv أو من رفع ملف / Google Sheets
# - زر داخلي لدمج ملفات data/spins_cleaned_*.csv(xlsx) إلى combined_spins.csv
# - زر تنظيف ملف casinoscores الخام (روابط صور) + إضافته للـcombined
# - نموذج Recency+Softmax مع Bonus boost + إمكانية تحميل نموذج مُتعلِّم .pkl
# - تبويبات: Tiles / Board + 10 / Table / Falcon Eye
# - تنبيه عين الصقر: احتمال تكرار "1" ≥ 3 مرات في 10 رميات

import os
import re
import math
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ===== محاولة استخدام دوالّك الأساسية إن وُجدت (لا نكسر شيء) =====
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
    "1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"
}
ORDER = ["1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]

# أحجام البلاطات
TILE_H=96; TILE_TXT=38; TILE_SUB=13
TILE_H_SMALL=84; TILE_TXT_SMALL=32; TILE_SUB_SMALL=12
TILE_TXT_BONUS=20

# ------------------------ وظائف مساعدة ------------------------
def pct(x: float) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "0.0%"

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
        """,
        unsafe_allow_html=True,
    )

def section_header(title):
    st.markdown(
        f"<div style='font-size:20px;font-weight:700;margin:6px 0 10px'>{title}</div>",
        unsafe_allow_html=True,
    )

# ---------- منظف قياسي لجدول ts,segment,multiplier ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["ts","segment","multiplier"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Column missing: {c}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["segment"] = out["segment"].astype(str).str.strip().str.upper()
    out["multiplier"] = (
        out["multiplier"].astype(str)
        .str.extract(r"(\d+)\s*[xX]?", expand=False)
        .fillna("1").astype(int).astype(str) + "X"
    )
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out[needed]

# ---------- تنظيف ملف casinoscores الخام (روابط صور) ----------
NAME2SEG = {
    "1": "1",
    "bar": "BAR", "barstatpin": "BAR", "barstat": "BAR",
    "discovip": "DISCO_VIP", "disco_vip": "DISCO_VIP", "vipdisco": "DISCO_VIP",
    "disco": "DISCO",
    "stayinalive": "STAYINALIVE", "stayin_alive": "STAYINALIVE", "stayin": "STAYINALIVE",
    "p":"P","l":"L","a":"A","y":"Y","f":"F","u":"U","n":"N","k":"K","t":"T","i":"I","m":"M","e":"E",
}
IMG_PATTERNS = [
    re.compile(r"/funky[-_]?time/([a-z0-9_]+)\.png", re.I),
    re.compile(r"/(barstatpin|barstat|bar|discovip|disco|stayinalive|stayin_alive|stayin|[playfuknytime1])\.png", re.I),
    re.compile(r"/([playfuknytime1])\.png", re.I),
]
def _guess_segment_from_text(text: str) -> str:
    t = str(text)
    for pat in IMG_PATTERNS:
        m = pat.search(t)
        if m:
            key = m.group(1).lower().strip().replace(" ", "")
            if key in NAME2SEG:
                return NAME2SEG[key]
            if len(key) == 1 and key in NAME2SEG:
                return NAME2SEG[key]
    return "UNKNOWN"

def _extract_multiplier_from_text(text: str) -> str:
    m = re.search(r"(\d+)\s*[xX]\b", str(text))
    return (m.group(1) + "X") if m else None

def clean_raw_casinoscores(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["ts","segment","multiplier"])
    # تعرّف عمود الزمن
    ts_col = None
    for c in df_raw.columns:
        if str(c).strip().lower() in {"ts","time","timestamp","date"}:
            ts_col = c; break
    if ts_col is None:
        # حاول أي عمود يبدو زمنًا
        for c in df_raw.columns:
            try:
                pd.to_datetime(df_raw[c])
                ts_col = c; break
            except Exception:
                continue
    if ts_col is None:
        raise ValueError("لا يوجد عمود تاريخ/وقت مفهوم (ts/time/timestamp/date).")

    text_cols = [c for c in df_raw.columns if df_raw[c].dtype == "object"]
    if not text_cols:
        text_cols = list(df_raw.columns)

    segs, mults = [], []
    for _, row in df_raw.iterrows():
        blob = " ".join(str(row[c]) for c in text_cols)
        seg = _guess_segment_from_text(blob)
        mul = _extract_multiplier_from_text(blob)
        if seg == "UNKNOWN" and "segment" in df_raw.columns:
            seg = str(row.get("segment","UNKNOWN")).strip().upper() or "UNKNOWN"
        if not mul and "multiplier" in df_raw.columns:
            mv = str(row.get("multiplier","")).strip()
            m2 = re.search(r"(\d+)\s*[xX]?", mv)
            mul = (m2.group(1)+"X") if m2 else None
        segs.append(seg)
        mults.append((mul or "1X").upper())

    out = pd.DataFrame({
        "ts": pd.to_datetime(df_raw[ts_col], errors="coerce"),
        "segment": pd.Series(segs, dtype="string").str.upper().str.replace(r"\s+","", regex=True),
        "multiplier": pd.Series(mults, dtype="string").str.upper(),
    })

    # تحسين: UNKNOWN + 1X -> "1"
    mask_heur = (out["segment"]=="UNKNOWN") & (out["multiplier"]=="1X")
    out.loc[mask_heur, "segment"] = "1"

    allowed = {"1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP","UNKNOWN"}
    out.loc[~out["segment"].isin(allowed), "segment"] = "UNKNOWN"

    out["multiplier"] = out["multiplier"].str.extract(r"(\d+)", expand=False).fillna("1").astype(int).astype(str) + "X"
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out[["ts","segment","multiplier"]]

# ---------- دمج ملفات spins_cleaned_* إلى combined ----------
def combine_inside_streamlit() -> tuple[int, str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = []
    for name in os.listdir(DATA_DIR):
        low = name.lower()
        if low.startswith("spins_cleaned") and (low.endswith(".csv") or low.endswith(".xlsx") or low.endswith(".xls")):
            paths.append(os.path.join(DATA_DIR, name))
    if not paths:
        return 0, "لم يتم العثور على أي ملفات تبدأ بـ spins_cleaned داخل data/."

    frames = []
    for p in sorted(paths):
        try:
            if p.lower().endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
            dfc = clean_df(df)
            frames.append(dfc)
        except Exception as e:
            st.warning(f"تجاوز الملف {os.path.basename(p)} بسبب: {e}")

    if not frames:
        return 0, "لم يتمكن القارئ من تحميل أي ملف صالح."

    big = pd.concat(frames, ignore_index=True)
    big["ts"] = pd.to_datetime(big["ts"], errors="coerce")
    big = big.dropna(subset=["ts"]).drop_duplicates(subset=["ts","segment","multiplier"])\
             .sort_values("ts").reset_index(drop=True)
    big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return len(big), f"تم الدمج في {REPO_COMBINED_PATH} — إجمالي الصفوف: {len(big):,}"

# ---------- زر تنظيف ملف مرفوع وإلحاقه بـ combined ----------
def try_clean_and_append_raw(file) -> str:
    """
    يحاول:
      1) قراءة الملف بصيغة قياسية ts/segment/multiplier
      2) أو تنظيفه بصيغة casinoscores الخام
    ثم يدمج الناتج في combined_spins.csv
    """
    if file is None:
        return "لا يوجد ملف مرفوع."

    # اقرأ الملف لأي صيغة
    if str(file.name).lower().endswith(".csv"):
        raw = pd.read_csv(file, dtype=str, keep_default_na=False)
    else:
        raw = pd.read_excel(file, dtype=str)

    # 1) محاولة الصيغة القياسية
    df_clean = None
    try:
        df_clean = clean_df(raw)
    except Exception:
        # 2) تنظيف الخام
        df_clean = clean_raw_casinoscores(raw)

    if df_clean.empty:
        return "بعد التنظيف لا توجد صفوف صالحة."

    # دمج مع الموجود
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(REPO_COMBINED_PATH):
        base = pd.read_csv(REPO_COMBINED_PATH)
        # توحيد أنواع
        base["ts"] = pd.to_datetime(base["ts"], errors="coerce")
        base["segment"] = base["segment"].astype(str).str.upper()
        base["multiplier"] = (
            base["multiplier"].astype(str).str.extract(r"(\d+)", expand=False).fillna("1").astype(int).astype(str) + "X"
        )
        all_df = pd.concat([base, df_clean], ignore_index=True)
    else:
        all_df = df_clean.copy()

    all_df["ts"] = pd.to_datetime(all_df["ts"], errors="coerce")
    all_df = all_df.dropna(subset=["ts"]).drop_duplicates(subset=["ts","segment","multiplier"])\
                   .sort_values("ts").reset_index(drop=True)
    all_df.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return f"✅ تم تنظيف {len(df_clean):,} صفًا وإضافتها. الحجم الكلي الآن: {len(all_df):,} صف."

# ---------- قراءة البيانات (repo / upload / sheets) ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    df = None

    if use_repo_file and os.path.exists(repo_path):
        try:
            df = pd.read_csv(repo_path)
        except Exception as e:
            st.warning(f"تعذر قراءة {repo_path}: {e}")

    if df is None and file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    if df is None and sheet_url:
        url = sheet_url.strip()
        if "docs.google.com/spreadsheets" in url and "export?format=csv" not in url:
            try:
                gid = url.split("gid=")[-1]
            except Exception:
                gid = "0"
            doc_id = url.split("/d/")[1].split("/")[0]
            url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"
        try:
            df = pd.read_csv(url)
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

    if len(df) > window:
        df = df.tail(window).copy()

    return df.reset_index(drop=True)

# -------- نموذج الاحتمالات: Recency + Softmax + Bonus boost --------
def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    try:
        dfx = df[~df["segment"].eq("UNKNOWN")].copy()
        if dfx.empty:
            dfx = df.copy()
        segs = list(ALL_SEGMENTS)
        n = len(dfx)

        if n == 0:
            vec = np.ones(len(segs), dtype=float)
        else:
            ages = np.arange(n, 0, -1)
            half = max(int(decay_half_life), 1)
            w = np.power(0.5, (ages-1)/half)
            w = w / w.sum()

            counts = {s: 0.0 for s in segs}
            for seg, wt in zip(dfx["segment"], w):
                if seg in counts:
                    counts[seg] += wt
            vec = np.array([counts[s] for s in segs], dtype=float)

        for i, s in enumerate(segs):
            if s in BONUS_SEGMENTS:
                vec[i] *= float(bonus_boost)

        if vec.sum() <= 0:
            vec[:] = 1.0
        x = vec / (vec.std() + 1e-9)
        x = x / max(float(temperature), 1e-6)
        z = np.exp(x - x.max())
        p_next = z / z.sum()

        probs = dict(zip(segs, p_next))
        p_in10 = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_in10
    except Exception:
        counts = df["segment"].value_counts()
        segs = list(ALL_SEGMENTS)
        vec = np.array([counts.get(s, 0) for s in segs], dtype=float)
        if vec.sum() == 0:
            vec[:] = 1.0
        z = np.exp((vec - vec.mean()) / (vec.std() + 1e-6))
        p = z / z.sum()
        probs = dict(zip(segs, p))
        p_in10 = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_in10

def get_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    if _HAS_CORE:
        try:
            dfn = normalize_df(df)
            comp = compute_probs(dfn, horizon=horizon)
            p_next = comp.get("p_next", {})
            p_in10 = comp.get("p_in10", {})
            if len(p_next) == 0 or len(p_in10) == 0:
                raise ValueError("Empty core probs -> use recency/softmax")
            return p_next, p_in10
        except Exception:
            pass

    return recency_softmax_probs(
        df,
        horizon=horizon,
        temperature=temperature,
        decay_half_life=decay_half_life,
        bonus_boost=bonus_boost,
    )

# ------------------------ الواجهة ------------------------
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

    st.subheader("🧩 إدارة البيانات")
    if st.button("🔁 دمج ملفات data/spins_cleaned*.csv(xlsx) إلى combined_spins.csv"):
        rows, msg = combine_inside_streamlit()
        if rows > 0:
            st.success(msg)
            load_data.clear(); st.experimental_rerun()
        else:
            st.warning(msg)

    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH, "rb") as f:
            st.download_button("⬇️ تنزيل combined_spins.csv", f.read(), file_name="combined_spins.csv", mime="text/csv")

    st.write("---")
    st.subheader("📥 مصدر البيانات")
    use_repo_combined = st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

    # زر تنظيف الملف الخام + الإضافة
    if upload is not None:
        if st.button("🧹 تنظيف + إضافة إلى combined_spins.csv"):
            try:
                msg = try_clean_and_append_raw(upload)
                st.success(msg)
                load_data.clear(); st.experimental_rerun()
            except Exception as e:
                st.error(f"فشل تنظيف/إضافة الملف الخام: {e}")

# تحميل الداتا
df = load_data(
    upload, sheet_url, window,
    use_repo_file=use_repo_combined, repo_path=REPO_COMBINED_PATH
)
if df.empty:
    st.info("أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")
    st.stop()

# حساب الاحتمالات
p_next, p_in10 = get_probs(
    df,
    horizon=horizon,
    temperature=temperature,
    decay_half_life=decay_half_life,
    bonus_boost=bonus_boost,
)

# تبويبات
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(
    ["🎛️ Tiles", "🎯 Board + 10 Spins", "📊 Table", "🦅 Falcon Eye"]
)

# ========== تبويب البلاطات ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")
    c1, c2, _, _ = st.columns([1.2, 1.2, 0.1, 0.1])
    with c1:
        display_tile("1", f"P(next) {pct(p_next.get('1', 0))}", letter_color("1"))
    with c2:
        display_tile("BAR", f"P(next) {pct(p_next.get('BAR', 0))}", letter_color("BAR"), txt_size=34)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, L in enumerate(["F","U","N","K","Y2"]):
        key = "Y" if L == "Y2" else L
        with cols[i]:
            display_tile(key, f"P(next) {pct(p_next.get(key, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        with cols[i]:
            display_tile(
                "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO"),
                f"P(next) {pct(p_next.get(B, 0))}",
                letter_color(B),
                height=TILE_H, txt_size=TILE_TXT_BONUS
            )

# ========== تبويب اللوحة + 10 ==========
with tab_board:
    section_header("لوحة الرهان + توقع الظهور خلال 10 جولات")
    st.caption("النسبة أسفل كل خانة هي احتمال الظهور مرة واحدة على الأقل خلال الجولات العشر القادمة.")
    def prob10(seg): return pct(p_at_least_once(p_next.get(seg, 0.0), 10))

    c1, c2 = st.columns(2)
    with c1:
        display_tile("1", f"≥1 in 10: {prob10('1')}", letter_color("1"),
                     height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    with c2:
        display_tile("BAR", f"≥1 in 10: {prob10('BAR')}", letter_color("BAR"),
                     height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, L in enumerate(["F","U","N","K","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L if L!="Y" else "Y2"),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label = "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]:
            display_tile(label, f"≥1 in 10: {prob10(B)}", letter_color(B),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_BONUS, sub_size=TILE_SUB_SMALL)

# ========== تبويب الجدول ==========
with tab_table:
    section_header("📊 جدول التكهّنات (10/15/25 و Exp in 15)")
    rows = []
    for s in ORDER:
        p = p_next.get(s, 0.0)
        rows.append({
            "Segment": "VIP DISCO" if s=="DISCO_VIP" else ("STAYIN'ALIVE" if s=="STAYINALIVE" else s),
            "≥1 in 10": p_at_least_once(p, 10),
            "≥1 in 15": p_at_least_once(p, 15),
            "≥1 in 25": p_at_least_once(p, 25),
            "Exp in 15": exp_count(p, 15),
            "_color": letter_color("Y2" if s=="Y" else s),
        })
    tdf = pd.DataFrame(rows)

    def _fmt(v, col):
        return f"{v*100:.1f}%" if col in {"≥1 in 10","≥1 in 15","≥1 in 25"} else (f"{v:.2f}" if col=="Exp in 15" else v)

    styled = (
        tdf.drop(columns=["_color"])
           .style.format({c: (lambda v, c=c: _fmt(v, c)) for c in ["≥1 in 10","≥1 in 15","≥1 in 25","Exp in 15"]})
           .apply(lambda s: [f"background-color: {tdf.loc[i,'_color']}; color: white; font-weight:700"
                             if s.name=="Segment" else "" for i in range(len(s))], axis=0)
    )
    st.dataframe(styled, use_container_width=True)

# ========== تبويب عين الصقر ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")
    any10 = 1.0; any15 = 1.0; any25 = 1.0
    for b in BONUS_SEGMENTS:
        pb = p_next.get(b, 0.0)
        any10 *= (1.0 - pb)**10
        any15 *= (1.0 - pb)**15
        any25 *= (1.0 - pb)**25
    any10 = 1.0 - any10; any15 = 1.0 - any15; any25 = 1.0 - any25

    c0, c1, c2 = st.columns(3)
    with c0:
        st.markdown(
            f"<div style='background:#6A1B9A;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 25: <span style='float:right'>{pct(any25)}</span></div>",
            unsafe_allow_html=True
        )
    with c1:
        st.markdown(
            f"<div style='background:#F8E16C;padding:14px;border-radius:14px;font-weight:700'>"
            f"🎁 بونص ≥ ×50 في 10: <span style='float:right'>{pct(sum((p_at_least_once(p_next.get(b,0.0),10) for b in BONUS_SEGMENTS))*0.25)}</span></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div style='background:#61C16D;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"💎 بونص ≥ ×100 في 10: <span style='float:right'>{pct(sum((p_at_least_once(p_next.get(b,0.0),10) for b in BONUS_SEGMENTS))*0.10)}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # تغيّر عام
    Wmini = min(30, len(df))
    if Wmini >= 10:
        tail = df.tail(Wmini)
        counts = tail["segment"].value_counts(normalize=True)
        meanp = counts.mean()
        varp = ((counts - meanp)**2).mean()
        if varp > 0.005:
            change_label = "High change"; badge = "<span style='color:#D32F2F;font-weight:700'>HIGH</span>"
        elif varp > 0.002:
            change_label = "Medium change"; badge = "<span style='color:#FB8C00;font-weight:700'>MEDIUM</span>"
        else:
            change_label = "Low change"; badge = "<span style='color:#2E7D32;font-weight:700'>LOW</span>"
    else:
        change_label = "Not enough data"; badge = "<span style='color:#999'>N/A</span>"

    st.markdown(
        f"<div style='background:#1E1E1E;color:#fff;padding:14px;border-radius:12px'>"
        f"🔎 التقلب العام: {change_label} — {badge}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # سيطرة محتملة للرقم 1 في 15
    p1_next = p_next.get("1", 0.0)
    p1_in15 = p_at_least_once(p1_next, 15)
    color15 = "#D32F2F" if p1_in15 > 0.85 else "#37474F"
    st.markdown(
        f"<div style='background:{color15};color:#fff;padding:14px;border-radius:12px'>"
        f"⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>",
        unsafe_allow_html=True
    )

    # احتمال تكرار 1 ثلاث مرات أو أكثر في 10
    def binom_tail_ge_k(n, p, k):
        p = max(0.0, min(1.0, float(p)))
        total = 0.0
        for r in range(0, k):
            total += math.comb(n, r) * (p**r) * ((1-p)**(n-r))
        return 1.0 - total
    p1_ge3_in10 = binom_tail_ge_k(10, p1_next, 3)
    st.markdown(
        f"<div style='background:#B71C1C;color:#fff;padding:14px;border-radius:12px'>"
        f"🛑 تنبيه حاد: احتمال أن يتكرر الرقم <b>1</b> ثلاث مرات أو أكثر خلال 10 سبِن = "
        f"<b>{pct(p1_ge3_in10)}</b> — يُنصح بالتوقف المؤقت.</div>",
        unsafe_allow_html=True
    )

# ========== أسفل الصفحة ==========
with st.expander("عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(50), use_container_width=True)

# ---------- تدريب النموذج من داخل التطبيق ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 تدريب النموذج (اختياري)")
model_path_input = st.sidebar.text_input("مسار حفظ النموذج", value="models/pattern_model.pkl")

with st.sidebar.expander("ملخص الداتا المستخدمة في التدريب"):
    st.write(f"عدد الرميات في النافذة الحالية: **{len(df)}**")
    st.write("أعمدة:", list(df.columns))
    st.dataframe(df.tail(10), use_container_width=True)

def train_and_save_model(df, path, horizon, temperature, decay_half_life, bonus_boost):
    p_next, _ = recency_softmax_probs(
        df, horizon=horizon, temperature=temperature,
        decay_half_life=decay_half_life, bonus_boost=bonus_boost
    )
    model = {
        "type": "recency_softmax",
        "p_next": p_next,
        "meta": {
            "horizon": horizon,
            "temperature": temperature,
            "half_life": decay_half_life,
            "bonus_boost": bonus_boost,
            "trained_on_rows": int(len(df)),
            "trained_at": datetime.utcnow().isoformat() + "Z",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model

if st.sidebar.button("💾 درِّب النموذج الآن", use_container_width=True):
    if df.empty:
        st.sidebar.error("لا توجد بيانات للتدريب.")
    else:
        try:
            model_obj = train_and_save_model(
                df, model_path_input, horizon, temperature, decay_half_life, bonus_boost
            )
            st.sidebar.success(f"تم حفظ النموذج: {model_path_input}")
            with open(model_path_input, "rb") as fh:
                st.sidebar.download_button(
                    label="⬇️ تحميل النموذج",
                    data=fh.read(),
                    file_name="pattern_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        except Exception as e:
            st.sidebar.error(f"فشل التدريب: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("نصيحة: بعد تحميل pattern_model.pkl ارفعه إلى مجلد models/ في GitHub ليبقى دائمًا.")
