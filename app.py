# app.py — Funky Brain LIVE (V3)
# - قراءة من data/combined_spins.csv أو رفع/Google Sheets
# - نموذج Recency+Softmax مع Bonus boost
# - تبويبات: Tiles / Board (+ horizon) / Table / Falcon Eye
# - تنبيه: تكرار "1" ≥3 في 10
# - دمج وتنظيف ملفات خام إلى combined_spins.csv
# - نافذة ديناميكية حتى 5000 رمية + إدخال رقمي متزامن

import os, re, math, pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# ===== محاولة استخدام دوالّك الأساسية إن وُجدت =====
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

# ألوان وتخطيط
COLORS = {
    "ONE": "#F4D36B", "BAR": "#5AA64F",
    "ORANGE": "#E7903C", "PINK": "#C85C8E", "PURPLE": "#9A5BC2",
    "STAYINALIVE": "#4FC3D9", "DISCO": "#314E96", "DISCO_VIP": "#B03232",
}
BONUS_SEGMENTS = {"DISCO","STAYINALIVE","DISCO_VIP","BAR"}
ALL_SEGMENTS = {
    "1","BAR","P","L","A","F","U","N","K","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP","Y"  # Y تُستعمل فقط عند وجودها بالبيانات
}
ORDER = ["1","BAR","P","L","A","F","U","N","K","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]  # بدون Y في اللوحات

TILE_H=92; TILE_TXT=36; TILE_SUB=12
TILE_H_SMALL=82; TILE_TXT_SMALL=30; TILE_SUB_SMALL=11
TILE_TXT_BONUS=18

def pct(x: float) -> str:
    try: return f"{float(x)*100:.1f}%"
    except: return "0.0%"

def p_at_least_once(p: float, n: int) -> float:
    return 1.0 - (1.0 - float(p))**int(n)

def exp_count(p: float, n: int) -> float:
    return float(n) * float(p)

def letter_color(letter: str) -> str:
    if letter in {"1","ONE"}: return COLORS["ONE"]
    if letter=="BAR": return COLORS["BAR"]
    if letter in {"P","L","A"}: return COLORS["ORANGE"]
    if letter in {"F","U","N","K"}: return COLORS["PINK"]
    if letter in {"T","I","M","E"}: return COLORS["PURPLE"]
    if letter=="STAYINALIVE": return COLORS["STAYINALIVE"]
    if letter=="DISCO": return COLORS["DISCO"]
    if letter=="DISCO_VIP": return COLORS["DISCO_VIP"]
    return "#444"

def display_tile(label, subtext, bg, height=TILE_H, radius=14, txt_size=TILE_TXT, sub_size=TILE_SUB):
    st.markdown(
        f"""
        <div style="background:{bg};color:white;border-radius:{radius}px;height:{height}px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:700;">
            <div style="font-size:{txt_size}px;line-height:1">{label}</div>
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

# ---------- تنظيف قياسي + منقّي ملفات خام ----------
SEG_MAP = {
    # رموز الصور/النصوص الشائعة إلى الحروف/البونص
    "bar": "BAR",
    "discovip": "DISCO_VIP",
    "disco_vip": "DISCO_VIP",
    "disco": "DISCO",
    "stayinalive": "STAYINALIVE",
    "stay_in_alive": "STAYINALIVE",
    # حروف
    "p": "P","l":"L","a":"A","f":"F","u":"U","n":"N","k":"K","t":"T","i":"I","m":"M","e":"E","y":"Y",
    "1": "1","one":"1"
}
LETTER_SET = set(list("PLAFUNKTIMEY"))

def _extract_segment_from_any(x: str) -> str:
    """يحاول استخراج القطاع من نص خام (مسار صورة/نص حر)."""
    s = str(x).strip()
    if not s: return "UNKNOWN"
    low = s.lower()
    # 1) لو فيه /funky-time/<token>.png
    m = re.search(r"funky[-_]?time/([a-z0-9_]+)\.png", low)
    if m:
        token = m.group(1)
        token = token.replace("barstatpin","bar").replace("barsstatpin","bar")
        token = token.replace("stayinalive","stayinalive").replace("stayinalive","stayinalive")
        if token in SEG_MAP: return SEG_MAP[token]
        # token حرف مفرد؟
        if len(token)==1 and token in SEG_MAP: return SEG_MAP[token]
    # 2) إن لم نجد، التقط حرفًا مفردًا/كلمة ديسكو/بار
    for k,v in SEG_MAP.items():
        if re.search(rf"\b{k}\b", low): return v
    # 3) كحل أخير: إذا السلسلة مجرد حرف
    if len(s)==1 and s.upper() in LETTER_SET: return s.upper()
    if s.strip()=="1": return "1"
    return "UNKNOWN"

def _extract_multiplier_any(x: str) -> str:
    """يعيد multiplier بصيغة 12X دومًا."""
    s = str(x)
    m = re.search(r"(\d+)\s*[xX]?", s)
    if not m: return "1X"
    return f"{int(m.group(1))}X"

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    يضمن الأعمدة: ts, segment, multiplier
    - يحوّل التواريخ
    - يطبّع segment من نص خام/روابط صور
    - multiplier إلى 12X
    """
    df = df.copy()
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # مصادر محتملة للقطاع/الضارب
    seg_col = None
    for cand in ["segment","symbol","result","outcome","icon","img","image","tile"]:
        if cand in df.columns: seg_col = cand; break
    mult_col = None
    for cand in ["multiplier","multi","x","coef","payout","winx"]:
        if cand in df.columns: mult_col = cand; break

    # تاريخ/وقت
    if "ts" not in df.columns:
        # أحيانًا يكون "time" أو "date"
        tcol = None
        for cand in ["time","date","timestamp","datetime"]:
            if cand in df.columns: tcol = cand; break
        if tcol is None:
            raise ValueError("Column missing: ts")
        df["ts"] = df[tcol]

    # اصطياد segment
    if seg_col is None:
        # حاول من أي عمود نصي فيه صور/نصوص
        text_cols = [c for c in df.columns if df[c].dtype==object]
        if text_cols:
            # خذ أول عمود نصّي كمرجع
            base = text_cols[0]
            df["segment"] = df[base].apply(_extract_segment_from_any)
        else:
            df["segment"] = "UNKNOWN"
    else:
        df["segment"] = df[seg_col].apply(_extract_segment_from_any)

    # multiplier
    if mult_col is None:
        # التقط من أي عمود
        got = None
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\d+\s*[xX]?$").any():
                got = c; break
        if got is None:
            df["multiplier"] = "1X"
        else:
            df["multiplier"] = df[got].apply(_extract_multiplier_any)
    else:
        df["multiplier"] = df[mult_col].apply(_extract_multiplier_any)

    # توحيد الأنواع
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["segment"] = df["segment"].astype(str).str.strip().str.upper()
    df["multiplier"] = df["multiplier"].astype(str).str.extract(r"(\d+)", expand=False).fillna("1").astype(int).astype(str) + "X"

    # إسقاط الفارغ وترتيب
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    df = df[["ts","segment","multiplier"]].sort_values("ts").reset_index(drop=True)
    return df

# ---------- دمج داخلي ----------
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
            if p.lower().endswith(".csv"): raw = pd.read_csv(p)
            else: raw = pd.read_excel(p)
            dfc = clean_df(raw)
            frames.append(dfc)
        except Exception as e:
            st.warning(f"تجاوز الملف {os.path.basename(p)} بسبب: {e}")

    if not frames:
        return 0, "لم يُحمّل أي ملف صالح."
    big = pd.concat(frames, ignore_index=True)
    big = big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
    big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return len(big), f"تم الدمج في {REPO_COMBINED_PATH} — إجمالي الصفوف: {len(big):,}"

# ---------- تحميل البيانات ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    df = None
    # (أ) من المستودع
    if use_repo_file and os.path.exists(repo_path):
        try: df = pd.read_csv(repo_path)
        except Exception as e: st.warning(f"تعذر قراءة {repo_path}: {e}")

    # (ب) ملف مرفوع
    if df is None and file is not None:
        try:
            if file.name.lower().endswith(".csv"): df = pd.read_csv(file)
            else: df = pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    # (ج) Google Sheets -> CSV
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

    return df.reset_index(drop=True)

# -------- نموذج الاحتمالات: Recency + Softmax + Bonus boost --------
def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    try:
        dfx = df[~df["segment"].eq("UNKNOWN")].copy()
        if dfx.empty: dfx = df.copy()
        segs = list(ALL_SEGMENTS)
        n = len(dfx)

        if n == 0:
            vec = np.ones(len(segs), dtype=float)
        else:
            ages = np.arange(n, 0, -1)               # الأحدث عمره 1
            half = max(int(decay_half_life), 1)
            w = np.power(0.5, (ages-1)/half)         # وزن أسي
            w = w / w.sum()

            counts = {s: 0.0 for s in segs}
            for seg, wt in zip(dfx["segment"], w):
                if seg in counts:
                    counts[seg] += wt
            vec = np.array([counts[s] for s in segs], dtype=float)

        # تعزيز للبونص
        for i, s in enumerate(segs):
            if s in BONUS_SEGMENTS:
                vec[i] *= float(bonus_boost)

        if vec.sum() <= 0: vec[:] = 1.0
        x = vec / (vec.std() + 1e-9)
        x = x / max(float(temperature), 1e-6)
        z = np.exp(x - x.max())
        p_next = z / z.sum()

        probs = dict(zip(segs, p_next))
        p_inH = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_inH
    except Exception:
        counts = df["segment"].value_counts()
        segs = list(ALL_SEGMENTS)
        vec = np.array([counts.get(s, 0) for s in segs], dtype=float)
        if vec.sum() == 0: vec[:] = 1.0
        z = np.exp((vec - vec.mean()) / (vec.std() + 1e-6))
        p = z / z.sum()
        probs = dict(zip(segs, p))
        p_inH = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_inH

def get_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    if _HAS_CORE:
        try:
            dfn = normalize_df(df)
            comp = compute_probs(dfn, horizon=horizon)
            p_next = comp.get("p_next", {})
            p_inH = comp.get("p_in10", {})  # قد يرجع مفاتيح أخرى، لكن نضمن وجود dict
            if len(p_next) == 0:
                raise ValueError("Empty core probs -> use recency/softmax")
            # إن لم يوجد p_inH سنعيد حسابه محليًا
            if not p_inH:
                p_inH = {k: p_at_least_once(v, horizon) for k, v in p_next.items()}
            return p_next, p_inH
        except Exception:
            pass
    return recency_softmax_probs(
        df,
        horizon=horizon,
        temperature=temperature,
        decay_half_life=decay_half_life,
        bonus_boost=bonus_boost,
    )

# ------------------------ الواجهة: إدارة البيانات ------------------------
with st.sidebar:
    st.subheader("🧩 إدارة البيانات")
    use_repo_combined = st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

# حمّل كل الداتا أولًا (قبل قص النافذة) لمعرفة الحد الأقصى للنافذة
df_all = load_data(upload, sheet_url, use_repo_file=use_repo_combined, repo_path=REPO_COMBINED_PATH)
if df_all.empty:
    st.info("أضف مصدر بيانات صالح يحتوي: ts, segment, multiplier")
    st.stop()

# إعدادات عامة + نافذة ديناميكية 5000 مع إدخال رقمي
with st.sidebar:
    st.subheader("⚙️ الإعدادات")
    max_window = int(min(len(df_all), 5000))
    c1, c2 = st.columns([1, 1])
    with c1:
        window = st.slider("Window size (spins)", min_value=50, max_value=max_window, value=min(300, max_window), step=10)
    with c2:
        window_num = st.number_input("أدخل عدد الرميات يدويًا", min_value=50, max_value=max_window, value=window, step=10)
    # تزامن القيمتين
    if window_num != window:
        window = int(window_num)

    horizon = st.slider("توقع على كم جولة؟", 5, 20, 10, step=1)

    st.write("---")
    st.subheader("🎛️ معلمات التنبؤ (Recency/Softmax)")
    temperature = st.slider("Temperature (تركيز السوفت-ماكس)", 1.0, 2.5, 1.6, 0.1)
    decay_half_life = st.slider("Half-life (ترجيح الحداثة)", 20, 200, 60, 5)
    bonus_boost = st.slider("تعزيز البونص", 1.00, 1.50, 1.15, 0.05)

    st.write("---")
    # زر الدمج التقليدي
    if st.button("🔁 دمج data/spins_cleaned*.csv(xlsx) إلى combined_spins.csv"):
        rows, msg = combine_inside_streamlit()
        if rows > 0:
            st.success(msg)
            load_data.clear()
            st.experimental_rerun()
        else:
            st.warning(msg)

    # تنظيف/إضافة ملف خام مرفوع مباشرةً
    if upload is not None:
        if st.button("🧹 تنظيف + إضافة إلى combined_spins.csv"):
            try:
                if upload.name.lower().endswith(".csv"): raw = pd.read_csv(upload)
                else: raw = pd.read_excel(upload)
                cleaned = clean_df(raw)
                st.success(f"تم التنظيف: {len(cleaned):,} صفًا")
                # ضمّه إلى الملف الموحّد
                if os.path.exists(REPO_COMBINED_PATH):
                    base = pd.read_csv(REPO_COMBINED_PATH)
                    base = clean_df(base)  # تأكيد النوع
                    merged = pd.concat([base, cleaned], ignore_index=True)
                else:
                    merged = cleaned.copy()
                merged = merged.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
                os.makedirs(DATA_DIR, exist_ok=True)
                merged.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
                st.success(f"أضيف إلى {REPO_COMBINED_PATH} — الحجم الآن: {len(merged):,}")
                # معاينة صغيرة
                with st.expander("معاينة بعد التنظيف/الدمج"):
                    st.dataframe(cleaned.tail(20), use_container_width=True)
                load_data.clear()
            except Exception as e:
                st.error(f"فشل تنظيف/إضافة الملف الخام: {e}")

    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH, "rb") as f:
            st.download_button("⬇️ تنزيل combined_spins.csv", f.read(), file_name="combined_spins.csv", mime="text/csv")

# قص النافذة الحالية
df = df_all.tail(window).copy()

# ------------------------ حساب الاحتمالات ------------------------
p_next, p_inH = get_probs(
    df,
    horizon=horizon,                  # الآن اللوحة تستعمل هذا الـ horizon
    temperature=temperature,
    decay_half_life=decay_half_life,
    bonus_boost=bonus_boost,
)

# تبويبات
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(
    ["🎛️ Tiles", "🎯 Board + Spins", "📊 Table", "🦅 Falcon Eye"]
)

# ========== Tiles ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")
    c1, c2, _, _ = st.columns([1.2, 1.2, 0.1, 0.1])
    with c1: display_tile("1", f"P(next) {pct(p_next.get('1', 0))}", letter_color("1"))
    with c2: display_tile("BAR", f"P(next) {pct(p_next.get('BAR', 0))}", letter_color("BAR"), txt_size=34)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(3)  # (بدون Y)
    for i, L in enumerate(["P","L","A"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # F U N K
    for i, L in enumerate(["F","U","N","K"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # T I M E
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(3)  # BONUS
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        with cols[i]:
            display_tile(
                "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO"),
                f"P(next) {pct(p_next.get(B, 0))}",
                letter_color(B),
                height=TILE_H, txt_size=TILE_TXT_BONUS
            )

# ========== Board + horizon ==========
with tab_board:
    section_header(f"لوحة الرهان + توقع الظهور خلال {horizon} جولات")
    st.caption("النسبة أسفل كل خانة هي احتمال الظهور مرة واحدة على الأقل خلال عدد الجولات المحددة (horizon).")
    def probH(seg): return pct(p_at_least_once(p_next.get(seg, 0.0), horizon))

    c1, c2 = st.columns(2)
    with c1:
        display_tile("1", f"≥1 in {horizon}: {probH('1')}", letter_color("1"),
                     height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    with c2:
        display_tile("BAR", f"≥1 in {horizon}: {probH('BAR')}", letter_color("BAR"),
                     height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(3)  # P L A
    for i, L in enumerate(["P","L","A"]):
        with cols[i]:
            display_tile(L, f"≥1 in {horizon}: {probH(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # F U N K
    for i, L in enumerate(["F","U","N","K"]):
        with cols[i]:
            display_tile(L, f"≥1 in {horizon}: {probH(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # T I M E
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"≥1 in {horizon}: {probH(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label = "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]:
            display_tile(label, f"≥1 in {horizon}: {probH(B)}", letter_color(B),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_BONUS, sub_size=TILE_SUB_SMALL)

# ========== Table ==========
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
            "_color": letter_color(s),
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

# ========== Falcon Eye ==========
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
            f"<div style='background:#1565C0;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 10: <span style='float:right'>{pct(any10)}</span></div>",
            unsafe_allow_html=True
        )
    with c1:
        st.markdown(
            f"<div style='background:#00897B;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 15: <span style='float:right'>{pct(any15)}</span></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div style='background:#6A1B9A;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🎲 احتمال أي بونص ≥1 في 25: <span style='float:right'>{pct(any25)}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    bonus10 = {b: p_at_least_once(p_next.get(b,0.0), 10) for b in BONUS_SEGMENTS}
    p50 = sum(bonus10.values()) * 0.25
    p100 = sum(bonus10.values()) * 0.10
    pLegend = sum(bonus10.values()) * 0.04

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            f"<div style='background:#F8E16C;padding:14px;border-radius:14px;font-weight:700'>"
            f"🎁 بونص ≥ ×50 في 10: <span style='float:right'>{pct(p50)}</span></div>",
            unsafe_allow_html=True
        )
    with d2:
        st.markdown(
            f"<div style='background:#61C16D;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"💎 بونص ≥ ×100 في 10: <span style='float:right'>{pct(p100)}</span></div>",
            unsafe_allow_html=True
        )
    with d3:
        st.markdown(
            f"<div style='background:#7C4DFF;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🚀 بونص أسطوري (+100) في 10: <span style='float:right'>{pct(pLegend)}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

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

    # تحذير: سيطرة 1 خلال 15
    p1_next = p_next.get("1", 0.0)
    p1_in15 = p_at_least_once(p1_next, 15)
    color15 = "#D32F2F" if p1_in15 > 0.85 else "#37474F"
    st.markdown(
        f"<div style='background:{color15};color:#fff;padding:14px;border-radius:12px'>"
        f"⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>",
        unsafe_allow_html=True
    )

    # تنبيه أحمر: P(X≥3 في 10) لـ 1
    def binom_tail_ge_k(n, p, k):
        p = max(0.0, min(1.0, float(p)))
        total = 0.0
        for r in range(0, k):
            total += math.comb(n, r) * (p**r) * ((1-p)**(n-r))
        return 1.0 - total
    p1_ge3_in10 = binom_tail_ge_k(10, p1_next, 3)
    st.markdown(
        f"<div style='background:#B71C1C;color:#fff;padding:14px;border-radius:12px'>"
        f"🛑 تنبيه حاد: احتمال تكرار الرقم <b>1</b> ثلاث مرات أو أكثر خلال 10 سبِن = "
        f"<b>{pct(p1_ge3_in10)}</b> — يُنصح بالتوقف المؤقت.</div>",
        unsafe_allow_html=True
    )

# ========== أسفل الصفحة ==========
with st.expander("عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(50), use_container_width=True)

# ---------- تدريب النموذج (اختياري) ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 تدريب النموذج (اختياري)")
model_path_input = st.sidebar.text_input("مسار حفظ النموذج", value="models/pattern_model.pkl")

with st.sidebar.expander("ملخص الداتا المستخدمة في التدريب"):
    st.write(f"عدد الرميات في النافذة الحالية: **{len(df)}**")
    st.write("أعمدة:", list(df.columns))
    st.dataframe(df.tail(10), use_container_width=True)

def train_and_save_model(df, path, horizon, temperature, decay_half_life, bonus_boost):
    p_next, _ = recency_softmax_probs(
        df,
        horizon=horizon,
        temperature=temperature,
        decay_half_life=decay_half_life,
        bonus_boost=bonus_boost,
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
            _ = train_and_save_model(
                df, model_path_input,
                horizon=horizon,
                temperature=temperature,
                decay_half_life=decay_half_life,
                bonus_boost=bonus_boost,
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
