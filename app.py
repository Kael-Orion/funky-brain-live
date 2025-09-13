# app.py — Funky Brain LIVE (Stable + Cleaner + Model Toggle)
# - يقرأ من data/combined_spins.csv أو من رفع ملف / Google Sheets
# - زر: تنظيف ملف خام casinoscores وإضافته إلى combined_spins.csv
# - نموذج Recency+Softmax مع Bonus boost + (اختياري) نموذج متعلّم من pkl
# - تبويبات: Tiles / Board + 10 / Table / Falcon Eye
# - تحذير عين الصقر: تكرار "1" ≥ 3 مرات في 10 / سيطرة محتملة خلال 15
# - إصلاحات: parsing للتواريخ، فرز آمن، UNKNOWN+1X => "1"، إظهار التدريب دائمًا

import os
import re
import math
import json
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

# مسارات
DATA_DIR = "data"
MODELS_DIR = "models"
REPO_COMBINED_PATH = os.path.join(DATA_DIR, "combined_spins.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ألوان البلاطات
COLORS = {
    "ONE": "#F4D36B", "BAR": "#5AA64F",
    "ORANGE": "#E7903C", "PINK": "#C85C8E", "PURPLE": "#9A5BC2",
    "STAYINALIVE": "#4FC3D9", "DISCO": "#314E96", "DISCO_VIP": "#B03232",
}
BONUS_SEGMENTS = {"DISCO","STAYINALIVE","DISCO_VIP","BAR"}
ALL_SEGMENTS = {
    "1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP","UNKNOWN"
}
ORDER = ["1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"]

# أحجام البلاطات
TILE_H=96; TILE_TXT=38; TILE_SUB=13
TILE_H_SMALL=84; TILE_TXT_SMALL=32; TILE_SUB_SMALL=12
TILE_TXT_BONUS=20

# ------------------------ وظائف UI مساعدة ------------------------
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

# ---------- منظّف قياسي لجدول نظيف ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["ts","segment","multiplier"]
    df = df.copy()

    # محاولات لإيجاد الأعمدة إن كانت بأسماء مختلفة
    colmap = {c.lower(): c for c in df.columns}
    ts_col = colmap.get("ts") or colmap.get("time") or colmap.get("timestamp") or colmap.get("date")
    seg_col = colmap.get("segment") or colmap.get("tile") or colmap.get("symbol")
    mul_col = colmap.get("multiplier") or colmap.get("multi") or colmap.get("x") or colmap.get("payout")

    if ts_col is None or seg_col is None or mul_col is None:
        # لو فشل، جرب القراءة بدون رؤوس (ملفات سيئة)
        try:
            tmp = pd.read_csv(df.to_csv(index=False), header=None)
            tmp.columns = ["ts","segment","multiplier"][:tmp.shape[1]] + [f"c{i}" for i in range(3, tmp.shape[1])]
            df = tmp
            ts_col, seg_col, mul_col = "ts","segment","multiplier"
        except Exception:
            raise ValueError("Column missing: ts/segment/multiplier")

    # تحويل
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], errors="coerce"),
        "segment": df[seg_col].astype(str).str.strip().str.upper(),
        "multiplier": df[mul_col].astype(str),
    })

    # multiplier -> "12X"
    out["multiplier"] = (
        out["multiplier"].str.extract(r"(\d+)\s*[xX]?", expand=False)
        .fillna("1").astype(int).astype(str) + "X"
    )

    # UNKNOWN + 1X => "1"
    mask_one = (out["segment"].isin(["UNKNOWN","N/A","NULL","-",""])) & (out["multiplier"].eq("1X"))
    out.loc[mask_one, "segment"] = "1"

    allowed = set(ALL_SEGMENTS)
    out.loc[~out["segment"].isin(allowed), "segment"] = "UNKNOWN"

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
            return NAME2SEG.get(key, "UNKNOWN")
    return "UNKNOWN"

def clean_raw_casinoscores(df_raw: pd.DataFrame) -> pd.DataFrame:
    """يدعم ملفات بلا رؤوس، أو رؤوس غير موحّدة، أو أعمدة نصية فقط."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["ts","segment","multiplier"])

    # 1) تحديد عمود الزمن (أعلى عمود قابل للتحويل)
    ts_col = None
    best_nonnull = -1
    for c in df_raw.columns:
        try:
            cand = pd.to_datetime(df_raw[c], errors="coerce")
            nn = cand.notna().mean()
            if nn >= 0.30 and nn > best_nonnull:
                ts_col = c
                best_nonnull = nn
        except Exception:
            pass
    if ts_col is None:
        # محاولة أخيرة: بدون رؤوس
        try:
            tmp = pd.read_csv(df_raw.to_csv(index=False), header=None)
            return clean_raw_casinoscores(tmp)
        except Exception:
            raise ValueError("لا يوجد عمود تاريخ/وقت مفهوم (ts/time/timestamp/date).")

    # 2) استخراج segment & multiplier من النصوص المجاورة (الروابط…)
    text_cols = [c for c in df_raw.columns if c != ts_col]
    if not text_cols:
        text_cols = [ts_col]

    segs, mults = [], []
    for _, row in df_raw.iterrows():
        blob = " ".join(str(row[c]) for c in text_cols)
        seg = _guess_segment_from_text(blob)

        # multiplier
        mul = None
        m = re.search(r"(\d+)\s*[xX]\b", blob)
        if m:
            mul = m.group(1) + "X"

        # fallback من أعمدة مسماة
        if seg == "UNKNOWN" and "segment" in df_raw.columns:
            seg = str(row.get("segment","UNKNOWN")).strip().upper() or "UNKNOWN"
        if not mul and "multiplier" in df_raw.columns:
            mv = str(row.get("multiplier","")).strip()
            m2 = re.search(r"(\d+)\s*[xX]?", mv)
            mul = (m2.group(1)+"X") if m2 else None

        # قاعدة خاصة: UNKNOWN + 1X => "1"
        if (seg == "UNKNOWN") and (mul or "").upper() == "1X":
            seg = "1"

        segs.append(seg)
        mults.append((mul or "1X").upper())

    out = pd.DataFrame({
        "ts": pd.to_datetime(df_raw[ts_col], errors="coerce"),
        "segment": pd.Series(segs, dtype="string").str.upper().str.replace(r"\s+","", regex=True),
        "multiplier": pd.Series(mults, dtype="string").str.upper(),
    })
    allowed = set(ALL_SEGMENTS)
    out.loc[~out["segment"].isin(allowed), "segment"] = "UNKNOWN"
    out["multiplier"] = out["multiplier"].str.extract(r"(\d+)", expand=False)\
                                        .fillna("1").astype(int).astype(str) + "X"
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out[["ts","segment","multiplier"]]

# ---------- دمج إلى combined_spins.csv ----------
def append_to_combined(df_new: pd.DataFrame, path=REPO_COMBINED_PATH) -> int:
    try:
        if os.path.exists(path):
            old = pd.read_csv(path)
            # توحيد قبل الدمج
            old = clean_df(old)
            big = pd.concat([old, df_new], ignore_index=True)
        else:
            big = df_new.copy()
        big = clean_df(big)
        big = big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts")
        big.to_csv(path, index=False, encoding="utf-8")
        return len(big)
    except Exception as e:
        raise RuntimeError(f"فشل الدمج/الحفظ: {e}")

# -------- نموذج الاحتمالات: Recency + Softmax + Bonus boost --------
def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    try:
        dfx = df.copy()
        segs = list(ALL_SEGMENTS - {"UNKNOWN"})  # نتجاهل UNKNOWN في التقدير
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
                if seg in counts: counts[seg] += wt
            vec = np.array([counts[s] for s in segs], dtype=float)

        for i, s in enumerate(segs):
            if s in BONUS_SEGMENTS:
                vec[i] *= float(bonus_boost)

        if vec.sum() <= 0: vec[:] = 1.0
        x = vec / (vec.std() + 1e-9)
        x = x / max(float(temperature), 1e-6)
        z = np.exp(x - x.max())
        p_next = z / z.sum()

        probs = dict(zip(segs, p_next))
        p_in10 = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_in10
    except Exception:
        counts = df["segment"].value_counts()
        segs = list(ALL_SEGMENTS - {"UNKNOWN"})
        vec = np.array([counts.get(s, 0) for s in segs], dtype=float)
        if vec.sum() == 0: vec[:] = 1.0
        z = np.exp((vec - vec.mean()) / (vec.std() + 1e-6))
        p = z / z.sum()
        probs = dict(zip(segs, p))
        p_in10 = {s: p_at_least_once(probs[s], horizon) for s in segs}
        return probs, p_in10

def get_probs_recency(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    if _HAS_CORE:
        try:
            dfn = clean_df(df)
            comp = compute_probs(dfn, horizon=horizon)
            p_next = comp.get("p_next", {})
            p_in10 = comp.get("p_in10", {})
            if len(p_next) == 0 or len(p_in10) == 0:
                raise ValueError("Empty core probs")
            return p_next, p_in10
        except Exception:
            pass
    return recency_softmax_probs(df, horizon, temperature, decay_half_life, bonus_boost)

# ------------------------ الواجهة: الشريط الجانبي أولًا (التدريب/النموذج) ------------------------
with st.sidebar:
    st.subheader("🤖 نموذج متعلّم (اختياري)")
    use_learned = st.toggle("استخدم النموذج المتعلّم إن وجد", value=False)
    model_path = st.text_input("مسار ملف النموذج", value=os.path.join(MODELS_DIR, "pattern_model.pkl"))

    # تحميل النموذج (لو مفعّل)
    loaded_model = None
    if use_learned and os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            st.success("تم تحميل النموذج.")
            if "meta" in loaded_model:
                with st.expander("إعدادات النموذج (meta)"):
                    st.code(json.dumps(loaded_model["meta"], ensure_ascii=False, indent=2))
        except Exception as e:
            st.error(f"تعذر تحميل النموذج: {e}")

    st.markdown("---")
    st.subheader("🧪 تدريب النموذج (اختياري)")
    save_model_path = st.text_input("مسار حفظ النموذج", value=os.path.join(MODELS_DIR,"pattern_model.pkl"))
    with st.expander("ملخص الداتا المستخدمة في التدريب (يعتمد على المصدر أسفل)"):
        st.caption("بعد اختيار مصدر البيانات بالأسفل سيتم عرض ملخص سريع هنا.")
    train_now_btn = st.button("💾 درِّب النموذج الآن", use_container_width=True, key="train_btn_top")

# ------------------------ مصدر البيانات + أدوات التنظيف ------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📥 مصدر البيانات")

    # زر تنظيف/إضافة لملف خام casinoscores
    st.caption("لو لديك ملف خام من Instant Data Scraper (يحتوي روابط صور)، نظِّفه وأضفه مباشرة إلى المخزون:")
    raw_file = st.file_uploader("ملف خام (CSV/XLSX/XLS)", type=["csv","xlsx","xls"], key="raw_uploader")

    if raw_file is not None:
        if st.button("🧹 تنظيف + إضافة إلى combined_spins.csv", use_container_width=True):
            try:
                if raw_file.name.lower().endswith(".csv"):
                    df_raw = pd.read_csv(raw_file, header=0, engine="python", encoding_errors="ignore")
                else:
                    df_raw = pd.read_excel(raw_file)

                df_clean = clean_raw_casinoscores(df_raw)
                total_rows = append_to_combined(df_clean, REPO_COMBINED_PATH)
                st.success(f"✅ تم التنظيف والإضافة ({len(df_clean)} صفًا). الحجم الكلي الآن: {total_rows} صفًا.")
                st.dataframe(df_clean.tail(20), use_container_width=True)
            except Exception as e:
                st.error(f"❌ فشل تنظيف/إضافة الملف الخام: {e}")

    use_repo_combined = st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel نظيف", type=["csv","xlsx","xls"], key="clean_uploader")

# ---------- تحميل البيانات (repo / upload / sheets) ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    df = None
    # من المستودع
    if use_repo_file and os.path.exists(repo_path):
        try:
            df = pd.read_csv(repo_path)
        except Exception as e:
            st.warning(f"تعذر قراءة {repo_path}: {e}")
    # ملف مرفوع
    if df is None and file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file, engine="python", encoding_errors="ignore")
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])
    # Google Sheets -> CSV
    if df is None and sheet_url:
        url = sheet_url.strip()
        if "docs.google.com/spreadsheets" in url and "export?format=csv" not in url:
            try: gid = url.split("gid=")[-1]
            except Exception: gid = "0"
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

# ------------------------ الإعدادات ------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("🎛️ معلمات التنبؤ (Recency/Softmax)")
    window = st.slider("Window size (spins)", 50, 300, 120, step=10)
    horizon = st.slider("توقع على كم جولة؟", 5, 20, 10, step=1)
    temperature = st.slider("Temperature (تركيز السوفت-ماكس)", 1.0, 2.5, 1.6, 0.1)
    decay_half_life = st.slider("Half-life (ترجيح الحداثة)", 20, 120, 60, 5)
    bonus_boost = st.slider("تعزيز البونص", 1.00, 1.40, 1.15, 0.05)

# حمّل الداتا
df = load_data(upload, sheet_url, window, use_repo_file=use_repo_combined, repo_path=REPO_COMBINED_PATH)

# ملخص التدريب (الذي وُضع سابقًا) يعتمد على df
with st.sidebar:
    with st.expander("ملخص الداتا المستخدمة في التدريب", expanded=False):
        st.write(f"عدد الصفوف المتاحة: **{len(df)}**")
        if not df.empty:
            st.dataframe(df.tail(10), use_container_width=True)

# تنفيذ التدريب لو ضغط الزر
if train_now_btn:
    with st.sidebar:
        if df.empty:
            st.error("لا توجد بيانات للتدريب.")
        else:
            try:
                import pickle
                # نستخدم نفس منطق الـ recency لضمان التطابق
                p_next_learned, _ = recency_softmax_probs(
                    df,
                    horizon=horizon,
                    temperature=temperature,
                    decay_half_life=decay_half_life,
                    bonus_boost=bonus_boost,
                )
                model = {
                    "type": "recency_softmax",
                    "p_next": p_next_learned,
                    "meta": {
                        "horizon": horizon,
                        "temperature": temperature,
                        "half_life": decay_half_life,
                        "bonus_boost": bonus_boost,
                        "trained_on_rows": int(len(df)),
                        "trained_at": datetime.utcnow().isoformat() + "Z",
                    },
                }
                with open(save_model_path, "wb") as f:
                    pickle.dump(model, f)
                st.success(f"تم حفظ النموذج: {save_model_path}")
                with open(save_model_path, "rb") as fh:
                    st.download_button("⬇️ تحميل النموذج", fh.read(), file_name="pattern_model.pkl",
                                       mime="application/octet-stream", use_container_width=True)
            except Exception as e:
                st.error(f"فشل التدريب: {e}")

# إن لم توجد بيانات، نعرض رسالة ونكمل (لا نوقف الصفحة كي تبقى الأقسام ظاهرة)
if df.empty:
    st.info("أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")

# ------------------------ حساب الاحتمالات ------------------------
if not df.empty:
    if loaded_model and use_learned and "p_next" in loaded_model:
        # استخدم النموذج المتعلم
        p_next = {k: float(v) for k, v in loaded_model["p_next"].items() if k in ALL_SEGMENTS and k != "UNKNOWN"}
        # إعادة التطبيع تحسّبًا
        s = sum(p_next.values()) or 1.0
        for k in p_next: p_next[k] /= s
        p_in10 = {s: p_at_least_once(p_next.get(s,0.0), 10) for s in p_next}
        source_label = "learned model"
    else:
        p_next, p_in10 = get_probs_recency(
            df, horizon=horizon, temperature=temperature,
            decay_half_life=decay_half_life, bonus_boost=bonus_boost
        )
        source_label = "recency"
else:
    p_next, p_in10 = {}, {}
    source_label = "none"

st.caption(f"Source of probabilities: {source_label}")

# ------------------------ التبويبات ------------------------
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
    for i, L in enumerate(["F","U","N","K","Y"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L if L!="Y" else "Y2"))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)
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
    if not df.empty:
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
    else:
        st.info("لا توجد بيانات لعرض الجدول.")

# ========== تبويب عين الصقر ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")

    if not df.empty:
        # احتمال أي بونص ≥1 خلال 10/15/25
        any10 = 1.0
        any15 = 1.0
        any25 = 1.0
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
                f"<div style='background:#5E35B1;padding:14px;border-radius:14px;font-weight:700;color:white'>"
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

        # تقديرات تقريبية
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

        # تغيُّر ديناميكي مبسط
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

        # سيطرة محتملة للرقم 1 خلال 15
        p1_next = p_next.get("1", 0.0)
        p1_in15 = p_at_least_once(p1_next, 15)
        color15 = "#D32F2F" if p1_in15 > 0.85 else "#37474F"
        st.markdown(
            f"<div style='background:{color15};color:#fff;padding:14px;border-radius:12px'>"
            f"⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>",
            unsafe_allow_html=True
        )

        # تكرار 1 ثلاث مرات+ خلال 10
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
    else:
        st.info("أضف بيانات لعرض التنبيهات.")

# ========== أسفل الصفحة ==========
with st.expander("عرض البيانات (آخر نافذة)"):
    if not df.empty:
        st.dataframe(df.tail(50), use_container_width=True)
    else:
        st.write("لا بيانات.")

with st.expander("تنزيل ملف البيانات المدموج"):
    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH, "rb") as f:
            st.download_button(
                label="Download combined_spins.csv",
                data=f.read(),
                file_name="combined_spins.csv",
                mime="text/csv"
            )
    else:
        st.info("لا يوجد data/combined_spins.csv في المستودع بعد.")
