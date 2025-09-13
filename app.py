# app.py — Funky Brain LIVE (V3)
# - مصدر البيانات: data/combined_spins.csv أو رفع ملف / Google Sheets
# - نموذج Recency+Softmax مع Bonus boost + (اختياري) نموذج متعلّم من الملف
# - تبويبات: Tiles / Board + 10 / Table / Falcon Eye
# - زر داخل التطبيق لدمج ملفات spins_cleaned* أو تنظيف ملف خام casinoscores* وإضافته
# - منظِّف ذكي يستخرج ts, segment, multiplier حتى لو كانت البيانات روابط صور فقط

import os
import re
import math
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone

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

# مسارات
DATA_DIR = "data"
MODELS_DIR = "models"
REPO_COMBINED_PATH = os.path.join(DATA_DIR, "combined_spins.csv")

# ------------------------ إعدادات اللون والقطاعات ------------------------
COLORS = {
    "ONE": "#F4D36B", "BAR": "#5AA64F",
    "ORANGE": "#E7903C", "PINK": "#C85C8E", "PURPLE": "#9A5BC2",
    "STAYINALIVE": "#4FC3D9", "DISCO": "#314E96", "DISCO_VIP": "#B03232",
}
BONUS_SEGMENTS = {"DISCO", "STAYINALIVE", "DISCO_VIP", "BAR"}
ALL_SEGMENTS = {
    "1","BAR","P","L","A","Y","F","U","N","K","T","I","M","E",
    "DISCO","STAYINALIVE","DISCO_VIP"
}
# ترتيب العرض (Y تحت A في الجدول)
ORDER = ["1","BAR","P","L","A","Y","F","U","N","K","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]

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

def letter_color(seg: str) -> str:
    if seg in {"1","ONE"}: return COLORS["ONE"]
    if seg=="BAR": return COLORS["BAR"]
    if seg in {"P","L","A","Y"}: return COLORS["ORANGE"]
    if seg in {"F","U","N","K"}: return COLORS["PINK"]
    if seg in {"T","I","M","E"}: return COLORS["PURPLE"]
    if seg=="STAYINALIVE": return COLORS["STAYINALIVE"]
    if seg=="DISCO": return COLORS["DISCO"]
    if seg=="DISCO_VIP": return COLORS["DISCO_VIP"]
    return "#444"

def display_tile(label, subtext, bg, height=TILE_H, radius=16, txt_size=TILE_TXT, sub_size=TILE_SUB):
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

# ---------- منظف الصفوف القياسي ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["ts", "segment", "multiplier"]
    df = df.copy()

    # تحقق من الأعمدة أو خرائط أسماء شائعة
    col_map = {c.lower().strip(): c for c in df.columns}
    for want in needed:
        if want not in df.columns:
            # جرّب أسماء بديلة
            alt = {"ts":["time","timestamp","date","datetime"],
                   "segment":["seg","symbol","tile","result"],
                   "multiplier":["multi","x","payout","odds","mult"]}
            matched = None
            for cand in alt.get(want, []):
                if cand in col_map:
                    matched = col_map[cand]
                    df = df.rename(columns={matched: want})
                    break
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Column missing: {c}")

    # ts → datetime
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    # segment
    df["segment"] = df["segment"].astype(str).str.strip().str.upper()
    # multiplier → "12X"
    df["multiplier"] = (
        df["multiplier"].astype(str)
        .str.extract(r"(\d+)\s*[xX]?", expand=False)
        .fillna("1").astype(int).astype(str) + "X"
    )
    df = df.dropna(subset=["ts", "segment"]).reset_index(drop=True)
    df = df.sort_values("ts")
    return df[needed]

# ---------- منظف ذكي للملفات الخام ----------
# يقرأ صفاً نصياً ويحاول استخراج (segment, multiplier) من الرابط أو النص
_SEG_PATTERNS = [
    (r"/1(?:\.png|\.jpg)|[^\w]one[^\w]", "1"),
    (r"/bar(?:[_./]|\.png|\.jpg)", "BAR"),
    (r"/disco[_-]?vip|/discovip|vip\s*disco", "DISCO_VIP"),
    (r"/stay(in'?alive)?|stayin'?alive", "STAYINALIVE"),
    (r"/disco(?:[_./]|\.png|\.jpg)", "DISCO"),
    # حروف
    (r"/p(?:[_./]|\.png|\.jpg)|[^\w]p[^\w]", "P"),
    (r"/l(?:[_./]|\.png|\.jpg)|[^\w]l[^\w]", "L"),
    (r"/a(?:[_./]|\.png|\.jpg)|[^\w]a[^\w]", "A"),
    (r"/y(?:[_./]|\.png|\.jpg)|[^\w]y[^\w]", "Y"),
    (r"/f(?:[_./]|\.png|\.jpg)|[^\w]f[^\w]", "F"),
    (r"/u(?:[_./]|\.png|\.jpg)|[^\w]u[^\w]", "U"),
    (r"/n(?:[_./]|\.png|\.jpg)|[^\w]n[^\w]", "N"),
    (r"/k(?:[_./]|\.png|\.jpg)|[^\w]k[^\w]", "K"),
    (r"/t(?:[_./]|\.png|\.jpg)|[^\w]t[^\w]", "T"),
    (r"/i(?:[_./]|\.png|\.jpg)|[^\w]i[^\w]", "I"),
    (r"/m(?:[_./]|\.png|\.jpg)|[^\w]m[^\w]", "M"),
    (r"/e(?:[_./]|\.png|\.jpg)|[^\w]e[^\w]", "E"),
]

def _guess_segment(text: str) -> str | None:
    s = str(text).lower()
    for pat, lab in _SEG_PATTERNS:
        if re.search(pat, s):
            return lab
    return None

def _guess_multiplier(text: str) -> str | None:
    s = str(text)
    # أمثلة: 25X, x25, _25x, ",25X", " 25 x "
    m = re.search(r"(\d{1,3})\s*[xX]\b", s)
    if not m:
        # أحياناً تأتي مفصولة بفواصل
        m = re.search(r"[^\d](\d{1,3})\s*X", s, flags=re.IGNORECASE)
    if not m:
        # في بعض الروابط: '_96X' أو ',96X'
        m = re.search(r"[_,](\d{1,3})\s*[xX]\b", s)
    if m:
        val = max(1, int(m.group(1)))
        return f"{val}X"
    return None

def smart_clean_any(df_raw: pd.DataFrame, source_name: str | None = None) -> pd.DataFrame:
    """
    يحاول إرجاع DataFrame بأعمدة ts, segment, multiplier حتى لو كانت البيانات خام (روابط / نص).
    قواعد:
      - إذا وُجدت ts/segment/multiplier نستخدم clean_df القياسية.
      - وإلا: نبحث في كل الأعمدة النصية عن segment و multiplier.
      - إذا لم يوجد ts: نولّد سلسلة زمنية دقيقة-بدقيقة تنتهي الآن (UTC).
      - إذا تعذّر معرفة القطاع لكن المضاعِف صغير (≤9X) نفترض "1".
    """
    # 1) جرّب المنظّف القياسي مباشرة
    try:
        return clean_df(df_raw)
    except Exception:
        pass

    # 2) منظف مرن
    df = df_raw.copy()

    # ابحث عن نص مجمّع (قد تكون كل البيانات في عمود واحد)
    text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not text_cols:
        # إن لم نجد أعمدة نصية، حوّل الكلّ إلى نص مؤقتاً
        text_cols = list(df.columns)

    seg_list = []
    mult_list = []

    for _, row in df.iterrows():
        blob = " | ".join([str(row[c]) for c in text_cols])
        seg = _guess_segment(blob)
        mult = _guess_multiplier(blob)

        # تخمينات إضافية من أعمدة شائعة
        for c in df.columns:
            s = str(row[c])
            if seg is None:
                seg = _guess_segment(s)
            if mult is None:
                mm = _guess_multiplier(s)
                if mm: mult = mm

        # إذا ما زال القطاع مجهول ولكن المضاعف صغير → اعتبره "1"
        if seg is None and mult is not None:
            try:
                mv = int(re.findall(r"\d+", mult)[0])
                if mv in {1,2,3,5,7,9}:
                    seg = "1"
            except Exception:
                pass

        seg_list.append(seg if seg is not None else "UNKNOWN")
        mult_list.append(mult if mult is not None else "1X")

    out = pd.DataFrame({"segment": seg_list, "multiplier": mult_list})

    # ts: حاول اكتشاف عمود وقت
    ts_col = None
    for cand in ["ts","time","timestamp","date","datetime"]:
        for c in df.columns:
            if c.lower().strip() == cand:
                ts_col = c
                break
        if ts_col: break

    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        # لا يوجد ts → أنشئ مواعيد دقيقة-بدقيقة حتى الآن
        now = pd.Timestamp.utcnow().floor("min")
        ts = pd.date_range(end=now, periods=len(df), freq="min")
    out["ts"] = ts

    # تنظيف نهائي بنفس قواعد القياسي
    out = clean_df(out)
    return out

# ---------- مدمج داخلي ----------
def combine_inside_streamlit() -> tuple[int, str]:
    """
    يقرأ كل الملفات التي تبدأ بـ spins_cleaned في data/ ويصنع combined_spins.csv
    """
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
    big = big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
    big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return len(big), f"تم الدمج في {REPO_COMBINED_PATH} — إجمالي الصفوف: {len(big):,}"

# ---------- قراءة البيانات ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    """
    يعيد آخر window صفوفًا مع ts, segment, multiplier
    """
    df = None

    # ملف المستودع
    if use_repo_file and os.path.exists(repo_path):
        try:
            df = pd.read_csv(repo_path)
        except Exception as e:
            st.warning(f"تعذر قراءة {repo_path}: {e}")

    # ملف مرفوع
    if df is None and file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                raw = pd.read_csv(file, dtype=str, engine="python")
            else:
                raw = pd.read_excel(file, dtype=str)
            # جرّب القياسي ثم الذكي
            try:
                df = clean_df(raw)
            except Exception:
                df = smart_clean_any(raw, file.name)
        except Exception as e:
            st.error(f"فشل قراءة/تنظيف الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    # Google Sheets
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
            raw = pd.read_csv(url, dtype=str)
            try:
                df = clean_df(raw)
            except Exception:
                df = smart_clean_any(raw, "google-sheets")
        except Exception as e:
            st.error(f"تعذّر تحميل Google Sheets: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    if df is None:
        return pd.DataFrame(columns=["ts","segment","multiplier"])

    # قص النافذة
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

        # softmax بدرجة حرارة
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

def get_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15,
              use_trained=False, model_path=os.path.join(MODELS_DIR,"pattern_model.pkl")):
    if use_trained:
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            p_next = model.get("p_next", {})
            # أعد تطبيع الاحتمالات إذا لزم
            s = sum(p_next.values()) if p_next else 0
            if s > 0:
                p_next = {k: v/s for k,v in p_next.items()}
            p_in10 = {k: p_at_least_once(p_next.get(k,0.0), horizon) for k in ALL_SEGMENTS}
            return p_next, p_in10
        except Exception as e:
            st.warning(f"تعذّر تحميل النموذج المتعلّم ({model_path}): {e}")

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
    window = st.slider("Window size (spins)", 50, 5000, 120, step=10)
    horizon = st.slider("توقّع على كم جولة؟", 5, 20, 10, step=1)

    st.write("---")
    st.subheader("🎛️ معلمات التنبؤ (Recency/Softmax)")
    temperature = st.slider("Temperature (تركيز السوفت-ماكس)", 1.0, 2.5, 1.6, 0.1)
    decay_half_life = st.slider("Half-life (ترجيح الحداثة)", 20, 120, 60, 5)
    bonus_boost = st.slider("تعزيز البونص", 1.00, 1.40, 1.15, 0.05)

    st.write("---")
    st.subheader("🧩 إدارة البيانات")

    # دمج spins_cleaned*
    if st.button("🔁 دمج ملفات data/spins_cleaned*.csv(xlsx) → combined_spins.csv", use_container_width=True):
        rows, msg = combine_inside_streamlit()
        if rows > 0:
            st.success(msg)
            load_data.clear()
            st.experimental_rerun()
        else:
            st.warning(msg)

    # زر التنظيف + الإضافة لملف خام مرفوع
    st.caption("إن رفعت ملف casinoscores خام: استعمل الزر التالي للتنظيف والإضافة إلى combined_spins.csv")
    raw_upload = st.file_uploader("…أو ارفع ملف CSV/Excel خام للتنظيف والإضافة", type=["csv","xlsx","xls"], key="raw_up")
    if raw_upload is not None:
        if st.button("🧹 تنظيف + إضافة إلى combined_spins.csv", use_container_width=True):
            try:
                raw = pd.read_csv(raw_upload, dtype=str) if raw_upload.name.lower().endswith(".csv") else pd.read_excel(raw_upload, dtype=str)
                cleaned = smart_clean_any(raw, raw_upload.name)
                os.makedirs(DATA_DIR, exist_ok=True)
                # أضِف إلى الموجود
                if os.path.exists(REPO_COMBINED_PATH):
                    base = pd.read_csv(REPO_COMBINED_PATH)
                    base = clean_df(base)
                    big = pd.concat([base, cleaned], ignore_index=True)
                else:
                    big = cleaned.copy()
                big = big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
                big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
                st.success(f"تم تنظيف ({len(cleaned)}) صفًا وإضافتها. الحجم الكلي الآن: {len(big):,} صفًا.")
                load_data.clear()
            except Exception as e:
                st.error(f"فشل تنظيف/إضافة الملف الخام: {e}")

    # تنزيل المدموج
    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH, "rb") as f:
            st.download_button("⬇️ تنزيل combined_spins.csv", f.read(), file_name="combined_spins.csv", mime="text/csv", use_container_width=True)

    st.write("---")
    st.subheader("📥 مصدر البيانات")
    use_repo_combined = st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel نظيف", type=["csv","xlsx","xls"])

    st.write("---")
    st.subheader("🤖 نموذج متعلّم (اختياري)")
    use_trained = st.toggle("استخدم النموذج المتعلّم إن وجد", value=False)
    model_path_input = st.text_input("مسار ملف النموذج", value=os.path.join(MODELS_DIR, "pattern_model.pkl"))

# تحميل الداتا
df = load_data(upload, sheet_url, window, use_repo_file=use_repo_combined, repo_path=REPO_COMBINED_PATH)
if df.empty:
    st.error("تنسيق الجدول غير صالح: أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")
    st.stop()

# حساب الاحتمالات
p_next, p_in10 = get_probs(
    df,
    horizon=horizon,
    temperature=temperature,
    decay_half_life=decay_half_life,
    bonus_boost=bonus_boost,
    use_trained=use_trained,
    model_path=model_path_input,
)

# تبويبات
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(["🎛️ Tiles", "🎯 Board + 10 Spins", "📊 Table", "🦅 Falcon Eye"])

# ========== تبويب البلاطات ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")
    c1, c2, _, _ = st.columns([1.2, 1.2, 0.1, 0.1])
    with c1:
        display_tile("1", f"P(next) {pct(p_next.get('1', 0))}", letter_color("1"))
    with c2:
        display_tile("BAR", f"P(next) {pct(p_next.get('BAR', 0))}", letter_color("BAR"), txt_size=34)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # PLAY (مع Y)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # FUNK
    for i, L in enumerate(["F","U","N","K"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # TIME
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label = "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]:
            display_tile(label, f"P(next) {pct(p_next.get(B, 0))}", letter_color(B),
                         height=TILE_H, txt_size=TILE_TXT_BONUS)

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

    cols = st.columns(4)  # PLAY
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # FUNK
    for i, L in enumerate(["F","U","N","K"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)  # TIME
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

# ========== تبويب عين الصقر ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")
    any10 = 1.0; any15 = 1.0; any25 = 1.0
    for b in BONUS_SEGMENTS:
        pb = p_next.get(b, 0.0)
        any10 *= (1.0 - pb)**10
        any15 *= (1.0 - pb)**15
        any25 *= (1.0 - pb)**25
    any10 = 1.0 - any10
    any15 = 1.0 - any15
    any25 = 1.0 - any25

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

    # تقلب عام مبسّط
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

    # تحذير 1 وحاد
    p1_next = p_next.get("1", 0.0)
    p1_in15 = p_at_least_once(p1_next, 15)
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
    st.dataframe(df.tail(200), use_container_width=True)

# ---------- تدريب النموذج من داخل التطبيق ----------
import pickle

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 تدريب النموذج (اختياري)")

save_path = st.sidebar.text_input("مسار حفظ النموذج", value=os.path.join(MODELS_DIR, "pattern_model.pkl"))

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
            _ = train_and_save_model(
                df, save_path, horizon=horizon,
                temperature=temperature, decay_half_life=decay_half_life,
                bonus_boost=bonus_boost,
            )
            st.sidebar.success(f"تم حفظ النموذج: {save_path}")
            with open(save_path, "rb") as fh:
                st.sidebar.download_button(
                    label="⬇️ تحميل النموذج",
                    data=fh.read(),
                    file_name=os.path.basename(save_path),
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        except Exception as e:
            st.sidebar.error(f"فشل التدريب: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("نصيحة: بعد تحميل pattern_model.pkl ارفعه إلى مجلد models/ في GitHub ليبقى دائمًا.")
