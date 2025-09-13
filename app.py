# app.py — Funky Brain LIVE (V3)
# - يقـرأ من data/combined_spins.csv أو من رفع ملف / Google Sheets
# - نموذج Recency+Softmax مع Bonus boost
# - تبويبات: Tiles / Board + 10 / Table / Falcon Eye
# - تنبيه عين الصقر: احتمال تكرار "1" ≥ 3 مرات في 10 رميات
# - زر داخل التطبيق لدمج ملفات data/spins_cleaned_*.csv(xlsx) إلى combined_spins.csv
# - معاينة فورية بعد التنظيف
# - خلفية "جو الديسكو"
# - تنبيه اتجاه RTP + تنبيهات صوتية للبونص الكبير و"عين الصقر"

import os
import math
import base64
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ===== محاولة استخدام دوالّك الأساسية إن وُجدت (لا نكسر شيء) =====
_HAS_CORE = False
try:
    from funkybrain_core import normalize_df, compute_probs, board_model
    _HAS_CORE = True
except Exception:
    _HAS_CORE = False

# ------------------------ إعدادات عامة ------------------------
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")

# خلفية "جو الديسكو"
st.markdown("""
<style>
/* خلفية كرة ديسكو خفيفة */
[data-testid="stAppViewContainer"]{
  background: radial-gradient( circle at 20% 20%, rgba(255,255,255,.08), rgba(0,0,0,0) 40% ),
              radial-gradient( circle at 80% 30%, rgba(255,255,255,.06), rgba(0,0,0,0) 40% ),
              url('https://images.unsplash.com/photo-1545239351-1141bd82e8a6?q=80&w=1200&auto=format&fit=crop') center/cover no-repeat fixed;
}
[data-testid="stHeader"] {background: transparent;}
/* بطاقات عائمة */
.block-container{backdrop-filter: blur(2px);}
</style>
""", unsafe_allow_html=True)

st.title("🪩 Funky Brain — LIVE (V3)")

# مسار ملف البيانات المدموج داخل المستودع
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

# أحجام البلاطات (تصغير متوازن)
TILE_H=86; TILE_TXT=32; TILE_SUB=12
TILE_H_SMALL=78; TILE_TXT_SMALL=28; TILE_SUB_SMALL=11
TILE_TXT_BONUS=18

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
    if letter in {"F","U","N","K","Y"}: return COLORS["PINK"]
    if letter in {"T","I","M","E"}: return COLORS["PURPLE"]
    if letter=="STAYINALIVE": return COLORS["STAYINALIVE"]
    if letter=="DISCO": return COLORS["DISCO"]
    if letter=="DISCO_VIP": return COLORS["DISCO_VIP"]
    return "#444"

def display_tile(label, subtext, bg, height=TILE_H, radius=16, txt_size=TILE_TXT, sub_size=TILE_SUB):
    st.markdown(
        f"""
        <div style="background:{bg};color:white;border-radius:{radius}px;height:{height}px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:700; box-shadow:0 4px 16px rgba(0,0,0,.20);">
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

# ---------- منظف الصفوف المعياري ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    يتوقع df فيه ts/segment/multiplier أو خام يحوي أعمدة بديلة.
    - يحاول إيجاد ts في أعمدة شائعة (ts, time, timestamp, date, created_at ...)
    - يحاول إيجاد segment مباشرة أو من مسار صورة ينتهي بـ .../<token>.png
    - multiplier إلى "12X"
    """
    data = df.copy()

    # ----- ts -----
    ts_cols = [c for c in data.columns if str(c).strip().lower() in
               {"ts","time","timestamp","datetime","date","created_at"}]
    if ts_cols:
        tsc = ts_cols[0]
        data["ts"] = pd.to_datetime(data[tsc], errors="coerce")
    else:
        # إن لم يوجد، نحاول من نص طويل يحوي وقتًا
        # وإلا نولّد تساويًا بفواصل دقيقة
        data["ts"] = pd.to_datetime(data.iloc[:,0], errors="coerce")
        if data["ts"].isna().all():
            base = datetime.utcnow() - timedelta(minutes=len(data))
            data["ts"] = [base + timedelta(minutes=i) for i in range(len(data))]

    # ----- segment -----
    if "segment" not in data.columns:
        data["segment"] = None
    # لو يوجد عمود URL/IMG نستخرِج من نهاية الاسم بلا الامتداد
    url_like_cols = [c for c in data.columns if any(x in str(c).lower() for x in ["img","image","url","src","icon","path"])]
    def _from_url(val: str) -> str|None:
        if not isinstance(val,str) or "." not in val: return None
        token = val.split("/")[-1].split(".")[0].strip().upper()
        # خرائط شائعة
        maps = {
            "1":"1","ONE":"1","BAR":"BAR","DISCO":"DISCO","DISCOVIP":"DISCO_VIP","DISCO_VIP":"DISCO_VIP",
            "STAYINALIVE":"STAYINALIVE",
            "P":"P","L":"L","A":"A","Y":"Y","F":"F","U":"U","N":"N","K":"K","T":"T","I":"I","M":"M","E":"E",
        }
        # حالات مثل funky-time/Y.png, barstatspin.png...
        token = token.replace("BARSTATSPIN","BAR").replace("DISCOVIP","DISCO_VIP").replace("VIPDISCO","DISCO_VIP")
        return maps.get(token, None)

    if data["segment"].isna().any() and url_like_cols:
        s = data["segment"].copy()
        for c in url_like_cols:
            s = s.fillna(data[c].map(_from_url))
        data["segment"] = s

    # أي قيم مجهولة نتركها UNKNOWN (ولا نحذف الصفوف)
    data["segment"] = data["segment"].astype(str).str.strip().str.upper().replace({"NONE":"UNKNOWN","NAN":"UNKNOWN","":"UNKNOWN"})

    # ----- multiplier -----
    mult_cols = [c for c in data.columns if str(c).strip().lower() in {"multiplier","multi","x","factor","payout"}]
    if mult_cols:
        mc = mult_cols[0]
        mult = data[mc].astype(str).str.extract(r"(\d+)", expand=False)
    else:
        # نحاول من النص الطويل (قد يكون "… 25X …")
        mult = data.apply(lambda r: pd.Series(str(r.astype(str).to_string())).str.extract(r"(\d+)[xX]"), axis=1)[0]
    mult = mult.fillna("1").astype(int).clip(lower=1, upper=500).astype(str) + "X"
    data["multiplier"] = mult

    # إسقاط ts الفارغ فقط
    data = data.dropna(subset=["ts"]).copy()
    data["ts"] = pd.to_datetime(data["ts"], errors="coerce")

    # ترتيب
    data = data.sort_values("ts")

    # أعمدة الناتج النهائي
    out = data[["ts","segment","multiplier"]].reset_index(drop=True)
    return out

# ---------- مدمج داخلي داخل التطبيق ----------
def combine_inside_streamlit(raw_file=None) -> tuple[int, str, pd.DataFrame|None]:
    """
    - يقرأ كل الملفات التي تبدأ بـ spins_cleaned في مجلد data/ (CSV/XLSX/XLS)
    - لو raw_file مُمرّر: ينظفه ويضمه أيضًا
    - يدمج إلى data/combined_spins.csv
    - يرجع (عدد_الصفوف, رسالة, معاينة_cleaned إن وُجدت للـ raw)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = []
    for name in os.listdir(DATA_DIR):
        low = name.lower()
        if low.startswith("spins_cleaned") and (low.endswith(".csv") or low.endswith(".xlsx") or low.endswith(".xls")):
            paths.append(os.path.join(DATA_DIR, name))

    frames = []
    # ملفات المجلد
    for p in sorted(paths):
        try:
            if p.lower().endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
            frames.append(clean_df(df))
        except Exception as e:
            st.warning(f"تجاوز الملف {os.path.basename(p)} بسبب: {e}")

    # ملف خام مرفوع الآن (اختياري)
    cleaned_preview = None
    if raw_file is not None:
        try:
            if raw_file.name.lower().endswith(".csv"):
                rdf = pd.read_csv(raw_file)
            else:
                rdf = pd.read_excel(raw_file)
            cleaned_preview = clean_df(rdf)
            frames.append(cleaned_preview)
        except Exception as e:
            return 0, f"فشل تنظيف/إضافة الملف الخام: {e}", None

    if not frames:
        return 0, "لم يتم العثور على أي بيانات صالحة للدمج.", cleaned_preview

    big = pd.concat(frames, ignore_index=True)
    # إزالة تكرارات
    big = big.drop_duplicates(subset=["ts","segment","multiplier"]).sort_values("ts").reset_index(drop=True)
    big.to_csv(REPO_COMBINED_PATH, index=False, encoding="utf-8")
    return len(big), f"تم الدمج في {REPO_COMBINED_PATH} — إجمالي الصفوف: {len(big):,}", cleaned_preview

# ---------- قراءة البيانات (repo / upload / sheets) ----------
@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window, use_repo_file=False, repo_path=REPO_COMBINED_PATH):
    """
    يحمّل البيانات من:
    - ملف المستودع data/combined_spins.csv (إن طُلب وموجود)
    - ملف مرفوع CSV/Excel
    - Google Sheets (نحوّل رابط العرض إلى export?format=csv تلقائيًا)
    ثم يرجع آخر window صفوف مع الأعمدة: ts, segment, multiplier
    """
    df = None

    # (أ) ملف المستودع
    if use_repo_file and os.path.exists(repo_path):
        try:
            df = pd.read_csv(repo_path)
        except Exception as e:
            st.warning(f"تعذر قراءة {repo_path}: {e}")

    # (ب) ملف مرفوع
    if df is None and file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    # (ج) Google Sheets -> CSV
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

    # قص النافذة
    if len(df) > window:
        df = df.tail(window).copy()

    return df.reset_index(drop=True)

# -------- نموذج الاحتمالات: Recency + Softmax + Bonus boost --------
def recency_softmax_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    """احتمالات مبنية على ترجيح حداثة أُسّي + Softmax بحرارة + تعزيز بسيط للبونص."""
    try:
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
        # Fallback بسيط (تكرارات)
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

def get_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15, use_model=False, model_path="models/pattern_model.pkl"):
    """يحاول استخدام دوالّك الأصلية أو ملف نموذج مدرّب، وإن فشل يستخدم نموذج الترجيح/السوفتماكس."""
    if use_model:
        try:
            import pickle
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "p_next" in obj:
                p_next = obj["p_next"]
                p_in10 = {s: p_at_least_once(p_next.get(s,0.0), horizon) for s in ALL_SEGMENTS}
                return p_next, p_in10
        except Exception:
            pass

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

# ------------------------ أصوات (base64) ------------------------
# بيب قصير
_BEEP = b'UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAACAAABaW1hAA=='
def play_beep():
    src = "data:audio/wav;base64," + base64.b64encode(base64.b64decode(_BEEP)).decode()
    st.markdown(f"""
    <audio autoplay="true">
      <source src="{src}" type="audio/wav">
    </audio>
    """, unsafe_allow_html=True)

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
    st.subheader("📥 مصدر البيانات")
    use_repo_combined = st.toggle("استخدم ملف المستودع data/combined_spins.csv", value=True)
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

    st.write("---")
    st.subheader("🧹 تنظيف خام + دمج")
    if upload is not None:
        if st.button("🪛 تنظيف + إضافة إلى combined_spins.csv", use_container_width=True):
            rows, msg, cleaned_preview = combine_inside_streamlit(raw_file=upload)
            if rows > 0:
                st.success(msg)
                if cleaned_preview is not None and not cleaned_preview.empty:
                    st.caption("📄 معاينة بعد التنظيف (أول/آخر 5):")
                    st.dataframe(pd.concat([cleaned_preview.head(5), cleaned_preview.tail(5)]), use_container_width=True)
                # إعادة تحميل الكاش حتى تظهر البيانات الجديدة
                load_data.clear()
                st.experimental_rerun()
            else:
                st.error(msg)
    else:
        st.caption("أرفق ملف خام إن أردت تنظيفه الآن.")

    # تحميل الملف المدموج
    if os.path.exists(REPO_COMBINED_PATH):
        with open(REPO_COMBINED_PATH, "rb") as f:
            st.download_button("⬇️ تنزيل combined_spins.csv", f.read(), file_name="combined_spins.csv", mime="text/csv")

    st.write("---")
    st.subheader("🤖 نموذج متعلِّم (اختياري)")
    use_trained_model = st.toggle("استخدم النموذج المتعلّم إن وجد", value=False)
    model_path_ui = st.text_input("مسار ملف النموذج", value="models/pattern_model.pkl")

# تحميل الداتا
df = load_data(
    upload if not use_repo_combined else None, sheet_url, window,
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
    use_model=use_trained_model,
    model_path=model_path_ui
)

# ---------------- RTP Trend (صعود/هبوط) ----------------
def _to_num(mult_str: str) -> float:
    try:
        return float(str(mult_str).lower().replace("x",""))
    except: return 1.0

rtp_recent_window = min(60, len(df))
baseline_window = min(300, len(df))
recent_avg = df.tail(rtp_recent_window)["multiplier"].map(_to_num).mean()
baseline_avg = df.tail(baseline_window)["multiplier"].map(_to_num).mean()
trend = (recent_avg - baseline_avg) / (baseline_avg + 1e-9)
if trend > 0.07:
    rtp_msg = f"⬆️ RTP يصعد (+{trend*100:.1f}%)"; rtp_color = "#2E7D32"
elif trend < -0.07:
    rtp_msg = f"⬇️ RTP ينزل ({trend*100:.1f}%)"; rtp_color = "#C62828"
else:
    rtp_msg = f"↔️ RTP مستقر (+{trend*100:.1f}%)"; rtp_color = "#6D4C41"

st.markdown(
    f"<div style='background:{rtp_color};color:#fff;padding:10px 14px;border-radius:10px;font-weight:700;margin-bottom:6px'>"
    f"{rtp_msg}</div>", unsafe_allow_html=True
)

# تبويبات: البلاطات + اللوحة + الجدول + عين الصقر
tab_tiles, tab_board, tab_table, tab_falcon = st.tabs(
    ["🎛️ Tiles", "🎯 Board + 10 Spins", "📊 Table", "🦅 Falcon Eye"]
)

# ========== تبويب البلاطات ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")

    c1, c2 = st.columns(2)
    with c1:
        display_tile("1", f"P(next) {pct(p_next.get('1', 0))}", letter_color("1"))
    with c2:
        display_tile("BAR", f"P(next) {pct(p_next.get('BAR', 0))}", letter_color("BAR"), txt_size=30)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    cols = st.columns(5)
    for i, L in enumerate(["F","U","N","K","Y"]):   # ← حذف Y المكرر (لم نعد نستخدم Y2)
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

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
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
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
            f"<div style='background:#4527A0;padding:14px;border-radius:14px;font-weight:700;color:white'>"
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

    # تقديرات ≥×50 / ≥×100 / أسطوري (تقريب)
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
            f"<div style='background:#1E88E5;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"💎 بونص ≥ ×100 في 10: <span style='float:right'>{pct(p100)}</span></div>",
            unsafe_allow_html=True
        )
    with d3:
        st.markdown(
            f"<div style='background:#7C4DFF;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🚀 بونص أسطوري (+100) في 10: <span style='float:right'>{pct(pLegend)}</span></div>",
            unsafe_allow_html=True
        )

    # أصوات عند تخطي العتبات
    if p100 >= 0.10:
        play_beep()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # تغيُّر ديناميكي (Variance مؤشر تقلب)
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

    # تحذير: سيطرة محتملة للرقم 1 خلال 15
    p1_next = p_next.get("1", 0.0)
    p1_in15 = p_at_least_once(p1_next, 15)
    high_risk_15 = p1_in15 > 0.85
    color15 = "#D32F2F" if high_risk_15 else "#37474F"
    st.markdown(
        f"<div style='background:{color15};color:#fff;padding:14px;border-radius:12px'>"
        f"⚠️ تحذير: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>",
        unsafe_allow_html=True
    )

    # NEW: تحذير أحمر إذا احتمال تكرار '1' ≥ 3 مرات في 10 رميات
    def binom_tail_ge_k(n, p, k):
        p = max(0.0, min(1.0, float(p)))
        total = 0.0
        for r in range(0, k):  # sum P[X = 0..k-1]
            total += math.comb(n, r) * (p**r) * ((1-p)**(n-r))
        return 1.0 - total

    p1_ge3_in10 = binom_tail_ge_k(10, p1_next, 3)
    color_ge3 = "#B71C1C"  # أحمر داكن
    st.markdown(
        f"<div style='background:{color_ge3};color:#fff;padding:14px;border-radius:12px'>"
        f"🛑 تنبيه حاد: احتمال أن يتكرر الرقم <b>1</b> ثلاث مرات أو أكثر خلال 10 سبِن = "
        f"<b>{pct(p1_ge3_in10)}</b> — يُنصح بالتوقف المؤقت.</div>",
        unsafe_allow_html=True
    )
    if p1_ge3_in10 >= 0.70:
        play_beep()

# ========== أسفل الصفحة ==========
with st.expander("📄 عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(80), use_container_width=True)

# ---------- تدريب النموذج من داخل التطبيق (يستخدم الداتا المحمّلة df) ----------
import pickle

with st.sidebar:
    st.markdown("---")
    st.subheader("🧪 تدريب النموذج (اختياري)")

    # مكان الحفظ الافتراضي
    model_path_input = st.text_input("مسار حفظ النموذج", value="models/pattern_model.pkl", key="model_save_path")

    # عرض لمحة سريعة عن الداتا الحالية
    with st.expander("ملخص الداتا المستخدمة في التدريب"):
        st.write(f"عدد الرميات في النافذة الحالية: **{len(df)}**")
        st.write("أعمدة:", list(df.columns))
        st.dataframe(df.tail(10), use_container_width=True)

    def train_and_save_model(df, path, horizon, temperature, decay_half_life, bonus_boost):
        # إعادة استعمال نفس الدالة لضمان التطابق
        pnext, _ = recency_softmax_probs(
            df,
            horizon=horizon,
            temperature=temperature,
            decay_half_life=decay_half_life,
            bonus_boost=bonus_boost,
        )
        model = {
            "type": "recency_softmax",
            "p_next": pnext,
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

    if st.button("💾 درِّب النموذج الآن", use_container_width=True):
        if df.empty:
            st.error("لا توجد بيانات للتدريب.")
        else:
            try:
                model_obj = train_and_save_model(
                    df,
                    model_path_input,
                    horizon=horizon,
                    temperature=temperature,
                    decay_half_life=decay_half_life,
                    bonus_boost=bonus_boost,
                )
                st.success(f"تم حفظ النموذج: {model_path_input}")
                with open(model_path_input, "rb") as fh:
                    st.download_button(
                        label="⬇️ تحميل النموذج",
                        data=fh.read(),
                        file_name=os.path.basename(model_path_input),
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"فشل التدريب: {e}")

    st.caption("نصيحة: بعد تحميل ملف النموذج ارفعه إلى مجلد models/ في GitHub ليبقى دائمًا.")
