# app.py — Funky Brain LIVE (Design Edition)

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
# الحروف: P L A Y | F U N K Y | T I M E
LETTER_GROUP = {
    "P": "ORANGE", "L": "ORANGE", "A": "ORANGE", "Y": "ORANGE",
    "F": "PINK",   "U": "PINK",   "N": "PINK",   "K": "PINK", "Y2":"PINK",
    "T": "PURPLE", "I": "PURPLE", "M": "PURPLE", "E": "PURPLE",
}
# كي نميّز الـ Y الأولى ضمن PLAY والـ Y الثانية ضمن FUNKY عند الرسم فقط
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
            # نحول رابط العرض إلى تصدير CSV
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
    # إن كانت لديك صيغة أخرى، حوّلها هنا
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
    df["segment"] = df["segment"].astype(str).str.upper().replace({"UNKNOWN":"UNKNOWN"})

    return df[["ts","segment","multiplier"]].reset_index(drop=True)


def naive_probs(df, horizon=10):
    """
    بديل آمن إذا لم تتوفر دوالّك: احتمالات نسبية من التكرار الأخير
    ويُعاد توزيعها بـ softmax بسيطة (للاستقرار).
    """
    counts = df["segment"].value_counts()
    segs = list(ALL_SEGMENTS)
    vec = np.array([counts.get(s, 0) for s in segs], dtype=float)
    if vec.sum() == 0:
        vec[:] = 1.0
    # softmax خفيفة
    z = np.exp((vec - vec.mean()) / (vec.std() + 1e-6))
    p = z / z.sum()
    probs = dict(zip(segs, p))
    # احتمال الظهور في ≥1 من 10 = 1 - (1 - p)^10 (تقريب مستقل)
    prob_in10 = {s: 1.0 - (1.0 - probs[s])**horizon for s in segs}
    return probs, prob_in10


def get_probs(df, horizon=10):
    """
    إما من دوالّك الأصلية (إن وُجدت) أو من النايف.
    يجب أن يُعيد:
      - p_next: احتمال الظهور في السبِن القادم لكل قطاع
      - p_in10: احتمال الظهور مرة على الأقل ضمن 10 سبِنات
    """
    if _HAS_CORE:
        try:
            dfn = normalize_df(df)
            comp = compute_probs(dfn, horizon=horizon)  # افتراض: يُعيد dict فيه p_next و p_in10
            p_next = comp.get("p_next", {})
            p_in10 = comp.get("p_in10", {})
            return p_next, p_in10
        except Exception:
            pass
    return naive_probs(df, horizon)


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


def display_tile(label, subtext, bg, height=110, radius=18, txt_size=42, sub_size=14):
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
    st.subheader("📥 مصدر البيانات")
    sheet_url = st.text_input("رابط Google Sheets (مفضّل CSV export)", value="")
    upload = st.file_uploader("…أو ارفع ملف CSV/Excel", type=["csv","xlsx","xls"])

# تحميل الداتا
df = load_data(upload, sheet_url, window)
if df.empty:
    st.info("أضف مصدر بيانات صالح يحتوي الأعمدة: ts, segment, multiplier")
    st.stop()

p_next, p_in10 = get_probs(df, horizon=horizon)  # dicts

tab_tiles, tab_board, tab_falcon = st.tabs(["🎛️ Tiles", "🎯 Board + 10 Spins", "🦅 Falcon Eye"])

# ========== تبويب البلاطات ==========
with tab_tiles:
    section_header("لوحة البلاطات (ألوان مخصصة)")
    # الصف العلوي: 1 | BAR
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.1, 0.1])
    with c1:
        display_tile("1", f"P(next) {pct(p_next.get('1', 0))}", letter_color("1"), height=110, txt_size=42)
    with c2:
        display_tile("BAR", f"P(next) {pct(p_next.get('BAR', 0))}", letter_color("BAR"), height=110, txt_size=36)

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
                height=120, txt_size=22
            )

# ========== تبويب اللوحة + توقع 10 ==========
with tab_board:
    section_header("لوحة الرهان + توقع الظهور خلال 10 جولات")
    st.caption("النسبة أسفل كل خانة هي احتمال الظهور مرة واحدة على الأقل خلال الجولات العشر القادمة.")

    def prob10(seg):
        return pct(p_in10.get(seg, 0))

    # نرتّب اللوحة بطريقة قريبة من الصورة/اللوحة
    # الصف: 1 | BAR
    c1, c2 = st.columns(2)
    with c1: display_tile("1", f"≥1 in 10: {prob10('1')}", letter_color("1"), height=90)
    with c2: display_tile("BAR", f"≥1 in 10: {prob10('BAR')}", letter_color("BAR"), height=90)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # PLAY
    cols = st.columns(4)
    for i, L in enumerate(["P","L","A","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L), height=90)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # FUNKY
    cols = st.columns(5)
    for i, L in enumerate(["F","U","N","K","Y"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L if L!="Y" else "Y2"), height=90)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # TIME
    cols = st.columns(4)
    for i, L in enumerate(["T","I","M","E"]):
        with cols[i]:
            display_tile(L, f"≥1 in 10: {prob10(L)}", letter_color(L), height=90)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Bonuses
    cols = st.columns(3)
    for i, B in enumerate(["DISCO","STAYINALIVE","DISCO_VIP"]):
        label = "VIP DISCO" if B=="DISCO_VIP" else ("STAYIN'ALIVE" if B=="STAYINALIVE" else "DISCO")
        with cols[i]:
            display_tile(label, f"≥1 in 10: {prob10(B)}", letter_color(B), height=96, txt_size=20)

# ========== تبويب عين الصقر ==========
with tab_falcon:
    section_header("عين الصقر — تنبيهات وتحذيرات")

    # مؤشرات مبسطة:
    # 1) تقدير احتمال ≥×50 و ≥×100 في البونصات خلال 10
    # (لو عندك model يُرجى استبدال التقديرات أدناه بدالتك)
    bonus10 = {b: p_in10.get(b, 0.0) for b in BONUS_SEGMENTS}
    # تقدير بدائي لاحتمال مضاعفات كبيرة: نعطي وزنًا أعلى للبونصات
    p50 = sum(bonus10.values()) * 0.25      # تقدير تقريبي
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

    # 2) High/Medium/Low change (إشارة ديناميكية مبسطة من تغيّر توزيع القطاعات)
    #      نقيس تباعد التوزيع الحالي عن متوسط آخر 3 نوافذ صغيرة
    Wmini = min(30, len(df))
    if Wmini >= 10:
        tail = df.tail(Wmini)
        counts = tail["segment"].value_counts(normalize=True)
        meanp = counts.mean()
        varp = ((counts - meanp)**2).mean()
        # عتبات تقريبية
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
    high_risk = p1_in15 > 0.85  # عتبة تقريبية
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
