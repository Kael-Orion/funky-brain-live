# app.py — Funky Brain LIVE (Design + Recency/Temperature)

import math
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ===== محاولات استيراد دوالك الأصلية إن وُجدت (لن نكسر الأساس) =====
_HAS_CORE = False
try:
    from funkybrain_core import normalize_df, compute_probs, board_model  # إن كانت لديك حزمة خاصة
    _HAS_CORE = True
except Exception:
    _HAS_CORE = False

# ====================== إعدادات عامة ======================
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain — LIVE")

# ألوان البلاطات حسب طلبك
COLORS = {
    "ONE": "#F4D36B",        # رقم 1 أصفر
    "BAR": "#5AA64F",        # BAR أخضر
    "ORANGE": "#E7903C",     # PLAY برتقالي
    "PINK": "#C85C8E",       # FUNKY وردي
    "PURPLE": "#9A5BC2",     # TIME بنفسجي
    "STAYINALIVE": "#4FC3D9",# أزرق فاتح
    "DISCO": "#314E96",      # أزرق غامق
    "DISCO_VIP": "#B03232",  # أحمر غامق
}

# خرائط القطاعات إلى مجموعات (للتلوين واللوحات)
LETTER_GROUP = {
    "P": "ORANGE", "L": "ORANGE", "A": "ORANGE", "Y": "ORANGE",
    "F": "PINK",   "U": "PINK",   "N": "PINK",   "K": "PINK", "Y2":"PINK",
    "T": "PURPLE", "I": "PURPLE", "M": "PURPLE", "E": "PURPLE",
}
GRID_LETTERS = [
    ["1", "BAR"],
    ["P", "L", "A", "Y"],
    ["F", "U", "N", "K", "Y2"],  # Y2 ستُعرض كـ “Y” لكنها تُلوَّن مجموعة FUNKY
    ["T", "I", "M", "E"],
    ["DISCO", "STAYINALIVE", "DISCO_VIP"]
]

BONUS_SEGMENTS = {"DISCO", "STAYINALIVE", "DISCO_VIP", "BAR"}
ALL_SEGMENTS = {
    "1", "BAR",
    "P","L","A","Y","F","U","N","K","Y","T","I","M","E",
    "DISCO","STAYINALIVE","DISCO_VIP"
}

# ========== أحجام البلاطات (صغّرناها) ==========
TILE_H = 96          # كان 110
TILE_TXT = 38        # كان 42
TILE_SUB = 13
TILE_H_SMALL = 84    # كان 90
TILE_TXT_SMALL = 32
TILE_SUB_SMALL = 12
TILE_H_BONUS = 96
TILE_TXT_BONUS = 20

# ====================== وظائف مساعدة ======================

@st.cache_data(show_spinner=False)
def load_data(file, sheet_url, window):
    """
    يحمّل البيانات من:
    - رفع ملف CSV/Excel، أو
    - Google Sheets (CSV export link)،
    ثم يُرجع آخر window صفوف مع الأعمدة ts, segment, multiplier
    """
    df = None

    # 1) ملف مرفوع؟
    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"فشل قراءة الملف: {e}")
            return pd.DataFrame(columns=["ts","segment","multiplier"])

    # 2) Google Sheets بصيغة CSV (نطلب رابط ‘/export?format=csv’ أو نعوّضه تلقائيًا)
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

    # نضمن الأعمدة المطلوبة
    wanted = ["ts","segment","multiplier"]
    for col in wanted:
        if col not in df.columns:
            st.error(f"❗ عمود مفقود في الجدول: {col}")
            return pd.DataFrame(columns=wanted)

    # تحويل ts إلى datetime إذا أمكن
    try:
        df["ts"] = pd.to_datetime(df["ts"])
    except Exception:
        pass

    # تنظيف multiplier ليكون “12X” مثالًا
    df["multiplier"] = (df["multiplier"]
                        .astype(str)
                        .str.extract(r"(\d+)\s*[xX]?", expand=False)
                        .fillna("1")
                        .astype(int)
                        .astype(str) + "X")

    # نأخذ آخر window صفوف
    if len(df) > window:
        df = df.tail(window).copy()

    # توحيد الحقول غير المعروفة
    df["segment"] = df["segment"].astype(str).str.upper()

    return df[["ts","segment","multiplier"]].reset_index(drop=True)


def recency_softmax_probs(
    df,
    horizon=10,
    temperature=1.6,
    decay_half_life=60,
    bonus_boost=1.15,
):
    """
    احتمالات مبنية على:
    - ترجيح حداثة أُسّي Half-life
    - Softmax بحرارة (Temperature)
    - تعزيز بسيط لقطاعات البونص
    """
    # استبعاد UNKNOWN
    dfx = df[~df["segment"].eq("UNKNOWN")].copy()
    if dfx.empty:
        dfx = df.copy()

    segs = list(ALL_SEGMENTS)
    n = len(dfx)

    # إذا ما فيه بيانات، وزّع متساوي
    if n == 0:
        vec = np.ones(len(segs), dtype=float)
    else:
        ages = np.arange(n, 0, -1)             # الأحدث عمره 1
        half = max(int(decay_half_life), 1)
        w = np.power(0.5, (ages-1)/half)       # وزن أُسّي
        w = w / w.sum()

        counts = {s: 0.0 for s in segs}
        for seg, weight in zip(dfx["segment"], w):
            if seg in counts:
                counts[seg] += weight
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
    p_in10 = {s: 1.0 - (1.0 - probs[s])**horizon for s in segs}
    return probs, p_in10


def fallback_naive(df, horizon=10):
    """بديل بسيط إذا لزم الأمر"""
    counts = df["segment"].value_counts()
    segs = list(ALL_SEGMENTS)
    vec = np.array([counts.get(s, 0) for s in segs], dtype=float)
    if vec.sum() == 0:
        vec[:] = 1.0
    z = np.exp((vec - vec.mean()) / (vec.std() + 1e-6))
    p = z / z.sum()
    probs = dict(zip(segs, p))
    prob_in10 = {s: 1.0 - (1.0 - probs[s])**horizon for s in segs}
    return probs, prob_in10


def get_probs(df, horizon=10, temperature=1.6, decay_half_life=60, bonus_boost=1.15):
    """
    يحاول استخدام دوالك الأصلية، وإلا يستخدم نموذج الترجيح/السوفتماكس.
    """
    if _HAS_CORE:
        try:
            dfn = normalize_df(df)
            comp = compute_probs(dfn, horizon=horizon)  # افتراض: يُعيد dict فيه p_next و p_in10
            p_next = comp.get("p_next", {})
            p_in10 = comp.get("p_in10", {})
            # لو ناقص أو فاضي، نكمّل بالطريقة الجديدة
            if len(p_next) == 0 or len(p_in10) == 0:
                raise ValueError("core probs empty -> use recency/softmax")
            return p_next, p_in10
        except Exception:
            pass

    # طريقتنا المحسّنة
    try:
        return recency_softmax_probs(
            df,
            horizon=horizon,
            temperature=temperature,
            decay_half_life=decay_half_life,
            bonus_boost=bonus_boost,
        )
    except Exception:
        return fallback_naive(df, horizon=horizon)


def pct(x):
    return f"{x*100:.1f}%"


def letter_color(letter):
    if letter in {"1","ONE"}:
        return COLORS["ONE"]
    if letter == "BAR":
        return COLORS["BAR"]
    if letter in {"P","L","A","Y"}:
        return COLORS[LETTER_GROUP[letter]]
    if letter in {"F","U","N","K","Y","Y2"}:
        return COLORS["PINK"]
    if letter in {"T","I","M","E"}:
        return COLORS["PURPLE"]
    if letter == "STAYINALIVE":
        return COLORS["STAYINALIVE"]
    if letter == "DISCO":
        return COLORS["DISCO"]
    if letter == "DISCO_VIP":
        return COLORS["DISCO_VIP"]
    return "#444"


def display_tile(label, subtext, bg, height=TILE_H, radius=16, txt_size=TILE_TXT, sub_size=TILE_SUB):
    st.markdown(
        f"""
        <div style="
            background:{bg};
            color:white;
            border-radius:{radius}px;
            height:{height}px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            font-weight:700;">
            <div style="font-size:{txt_size}px; line-height:1">{label if label!='Y2' else 'Y'}</div>
            <div style="font-size:{sub_size}px; opacity:.95; margin-top:2px">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

p_next, p_in10 = get_probs(
    df,
    horizon=horizon,
    temperature=temperature,
    decay_half_life=decay_half_life,
    bonus_boost=bonus_boost,
)

tab_tiles, tab_board, tab_falcon = st.tabs(["🎛️ Tiles", "🎯 Board + 10 Spins", "🦅 Falcon Eye"])

# ========== تبويب البلاطات ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")
    # الصف العلوي: 1 | BAR
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.1, 0.1])
    with c1:
        display_tile("1", f"P(next) {pct(p_next.get('1', 0))}", letter_color("1"))
    with c2:
        display_tile("BAR", f"P(next) {pct(p_next.get('BAR', 0))}", letter_color("BAR"), txt_size=34)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # الصف الثاني: PLAY
    cols = st.columns(4)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # الصف الثالث: FUNKY (مع Y2 تشكيلًا)
    cols = st.columns(5)
    for i, L in enumerate(["F","U","N","K","Y2"]):
        with cols[i]:
            key = "Y" if L == "Y2" else L
            display_tile(key, f"P(next) {pct(p_next.get(key, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # الصف الرابع: TIME
    cols = st.columns(4)
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"P(next) {pct(p_next.get(L, 0))}", letter_color(L))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # الصف السفلي: البونص
    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        with cols[i]:
            display_tile(
                "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO"),
                f"P(next) {pct(p_next.get(B, 0))}",
                letter_color(B),
                height=TILE_H, txt_size=TILE_TXT_BONUS
            )

# ========== تبويب اللوحة + توقع 10 ==========
with tab_board:
    section_header("لوحة الرهان + توقع الظهور خلال 10 جولات")
    st.caption("النسبة أسفل كل خانة هي احتمال الظهور مرة واحدة على الأقل خلال الجولات العشر القادمة.")

    def prob10(seg):
        return pct(p_in10.get(seg, 0))

    # الصف: 1 | BAR
    c1, c2 = st.columns(2)
    with c1: display_tile("1", f"≥1 in 10: {prob10('1')}", letter_color("1"),
                           height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)
    with c2: display_tile("BAR", f"≥1 in 10: {prob10('BAR')}", letter_color("BAR"),
                           height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # PLAY
    cols = st.columns(4)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # FUNKY
    cols = st.columns(5)
    for i, L in enumerate(["F","U","N","K","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L if L!="Y" else "Y2"),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # TIME
    cols = st.columns(4)
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_SMALL, sub_size=TILE_SUB_SMALL)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Bonuses
    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label = "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]:
            display_tile(label, f"≥1 in 10: {prob10(B)}", letter_color(B),
                         height=TILE_H_SMALL, txt_size=TILE_TXT_BONUS, sub_size=TILE_SUB_SMALL)

# ========== تبويب عين الصقر ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")

    # 1) تقدير مبسّط لاحتمالات البونص
    bonus10 = {b: p_in10.get(b, 0.0) for b in BONUS_SEGMENTS}
    p50 = sum(bonus10.values()) * 0.25
    p100 = sum(bonus10.values()) * 0.10
    pLegend = sum(bonus10.values()) * 0.04

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background:#F8E16C;padding:14px;border-radius:14px;font-weight:700'>"
            f"🎁 احتمال بونص ≥ ×50 خلال 10: <span style='float:right'>{pct(p50)}</span></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div style='background:#61C16D;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"💎 احتمال بونص ≥ ×100 خلال 10: <span style='float:right'>{pct(p100)}</span></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div style='background:#7C4DFF;padding:14px;border-radius:14px;font-weight:700;color:white'>"
            f"🚀 بونص أسطوري (+100) خلال 10: <span style='float:right'>{pct(pLegend)}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # 2) تغيُّر ديناميكي مبسّط
    Wmini = min(30, len(df))
    if Wmini >= 10:
        tail = df.tail(Wmini)
        counts = tail["segment"].value_counts(normalize=True)
        meanp = counts.mean()
        varp = ((counts - meanp)**2).mean()
        if varp > 0.005:
            change_label = "High change"
            badge = "<span style='color:#D32F2F;font-weight:700'>HIGH</span>"
        elif varp > 0.002:
            change_label = "Medium change"
            badge = "<span style='color:#FB8C00;font-weight:700'>MEDIUM</span>"
        else:
            change_label = "Low change"
            badge = "<span style='color:#2E7D32;font-weight:700'>LOW</span>"
    else:
        change_label = "Not enough data"
        badge = "<span style='color:#999'>N/A</span>"

    st.markdown(
        f"<div style='background:#1E1E1E;color:#fff;padding:14px;border-radius:12px'>"
        f"🔎 التقلب العام: {change_label} — {badge}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # 3) تحذير High Risk: سيطرة “1” في 15 جولة قادمة (بديل مبسط)
    p1_next, p1_in15 = p_next.get("1", 0.0), (1 - (1 - p_next.get("1", 0.0))**15)
    high_risk = p1_in15 > 0.85
    color = "#D32F2F" if high_risk else "#37474F"
    st.markdown(
        f"<div style='background:{color};color:#fff;padding:14px;border-radius:12px'>"
        f"⚠️ تحذير المخاطرة: سيطرة محتملة للرقم 1 خلال 15 سبِن — P(≥1 خلال 15) = {pct(p1_in15)}</div>",
        unsafe_allow_html=True
    )

    st.caption("🔧 ملاحظة: عندما تفعّل نماذجك الخاصة سيُستبدل كل ما سبق بتقديراتك الدقيقة تلقائيًا.")

# ========== أسفل الصفحة: معاينة الداتا ==========
with st.expander("عرض البيانات (آخر نافذة)"):
    st.dataframe(df.tail(50), use_container_width=True)
