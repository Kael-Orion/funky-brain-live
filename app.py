# app.py — Funky Brain LIVE (Cloud)

import time
import pandas as pd
import streamlit as st

# دوال المشروع الأساسية
from funkybrain_core import normalize_df, compute_probs, board_matrix

# محاولة استيراد الجالب الاختياري (قد لا يكون متوفراً)
_FETCH_AVAILABLE = False
try:
    from fetchers.casinoscores import fetch_latest_df  # يجب أن تُرجع DataFrame
    _FETCH_AVAILABLE = True
except Exception:
    _FETCH_AVAILABLE = False


# ===================== الإعدادات العامة للصفحة =====================
st.set_page_config(page_title="Funky Brain LIVE", layout="wide")
st.title("🧠 Funky Brain – LIVE (Cloud)")

# ===================== الشريط الجانبي =====================
st.sidebar.header("الإعدادات")

window = st.sidebar.slider("Window size (spins)", 50, 200, 120, step=10)

st.sidebar.subheader("رفع ملف CSV من casinoscores")
uploads = st.sidebar.file_uploader("يمكن رفع أكثر من ملف (CSV)", type=["csv"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.subheader("التحديث التلقائي")

auto = st.sidebar.checkbox("Auto-refresh")
every = st.sidebar.slider("كل كم ثانية؟", 10, 90, 45, help="الفاصل الزمني لإعادة الحساب/العرض")

st.sidebar.markdown("---")
if _FETCH_AVAILABLE:
    fetch_now = st.sidebar.button("جلب آخر الرميات تلقائيًا (تجريبي)")
else:
    fetch_now = False
    st.sidebar.button("جلب آخر الرميات تلقائيًا (غير متاح)", disabled=True)
    st.sidebar.caption("◻ لإتاحة الجلب: فعّل fetchers/casinoscores.py في المستودع.")

# ===================== تجهيز البيانات =====================
df_source_msg = None
raw_df = None

# أولوية 1: الجلب المباشر (لو متاح وتم الضغط)
if fetch_now and _FETCH_AVAILABLE:
    try:
        raw_df = fetch_latest_df()
        df_source_msg = "تم الجلب تلقائيًا من CasinoScores."
    except Exception as e:
        st.warning(f"تعذّر الجلب المباشر: {e}. سيتم الاعتماد على ملفات CSV المرفوعة إن وُجدت.")

# أولوية 2: ملفات CSV المرفوعة
if raw_df is None and uploads:
    try:
        dfs = [pd.read_csv(f) for f in uploads]
        raw_df = pd.concat(dfs, ignore_index=True)
        df_source_msg = "تم تحميل البيانات من ملفات CSV."
    except Exception as e:
        st.error(f"فشل قراءة CSV: {e}")
        st.stop()

# لا توجد بيانات؟
if raw_df is None:
    st.info("ابدأ التحليل برفع ملف/ملفات CSV من casinoscores أو استخدم زر الجلب (إن كان متاحًا).")
    # تشغيل التحديث التلقائي حتى لو ما في بيانات بعد (مفيد لو تنتظر الجلب)
    if auto:
        st.sidebar.write("⏳ في انتظار بيانات…")
        st.rerun()
    st.stop()

# ===================== معالجة واحتساب =====================
try:
    df = normalize_df(raw_df)  # توحيد الأعمدة ورموز البلاطات
except Exception as e:
    st.error(f"خطأ أثناء تهيئة البيانات normalize: {e}")
    st.stop()

try:
    # المتوقع: ترجع DataFrame للبلاطات + قاموس/DF لعين الصقر + قيمة/قاموس للنافذة
    tiles_df, eyes_info, win_info = compute_probs(df, window)
except Exception as e:
    st.error(f"خطأ أثناء الحساب compute_probs: {e}")
    st.stop()

if df_source_msg:
    st.success(df_source_msg)

# ===================== عرض جداول الـ Tiles =====================
st.subheader("Tiles – احتمالات وتوقعات")

# نجرب نعيد تسمية الأعمدة إلى أسماء ودّية لو كانت موجودة بأسماء داخلية
rename_map = {
    "title": "Tile",
    "p_next": "P(next)",
    "exp10": "Exp in 10",
    "p_any10": "P(≥1 in 10)",
    "exp15": "Exp in 15",
    "p_any15": "P(≥1 in 15)"
}
safe_tiles = tiles_df.copy()

for src, dst in rename_map.items():
    if src in safe_tiles.columns and dst not in safe_tiles.columns:
        safe_tiles = safe_tiles.rename(columns={src: dst})

# تأكد من الأعمدة المطلوبة
for col in ["Tile", "P(next)", "Exp in 10", "P(≥1 in 10)", "Exp in 15", "P(≥1 in 15)"]:
    if col not in safe_tiles.columns:
        # لو العمود ناقص، أنشئه بقيمة افتراضية
        if col.startswith("P("):
            safe_tiles[col] = 0.0
        else:
            safe_tiles[col] = 0.0

# تنسيق نسبي/نسبة
def fmt_percent(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"

def fmt_float(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return "-"

show = safe_tiles[["Tile", "P(next)", "Exp in 10", "P(≥1 in 10)", "Exp in 15", "P(≥1 in 15)"]].copy()
show["P(next)"] = show["P(next)"].apply(fmt_percent)
show["P(≥1 in 10)"] = show["P(≥1 in 10)"].apply(fmt_percent)
show["P(≥1 in 15)"] = show["P(≥1 in 15)"].apply(fmt_percent)
show["Exp in 10"] = show["Exp in 10"].apply(fmt_float)
show["Exp in 15"] = show["Exp in 15"].apply(fmt_float)

st.dataframe(show, use_container_width=True)

# ===================== لوحة Board =====================
st.subheader(f"Board – P(≥1 in 10) • Window={window}")

try:
    board = board_matrix(safe_tiles)  # يتوقع DataFrame بشكل اللوح
    # نعرضه على شكل شبكة ألوان بسيطة
    # ملاحظة: Streamlit لا يدعم Grid حقيقية بسهولة؛ سنعرضه كـ dataframe ملوّن
    def colorize(val):
        # نحاول تحويل النسبة لعدد
        try:
            if isinstance(val, str) and val.endswith("%"):
                p = float(val.replace("%", ""))
            else:
                p = float(val) * 100.0 if float(val) <= 1 else float(val)
        except Exception:
            return ""
        # تدرج بسيط: أحمر منخفض / برتقالي متوسط / أخضر عالي
        if p >= 60:
            color = "#1e8a4b"  # أخضر
        elif p >= 35:
            color = "#f29d38"  # برتقالي
        else:
            color = "#c94a4a"  # أحمر
        return f"background-color: {color}; color: white; font-weight:600; text-align:center;"

    # نتوقع أن اللوح يحمل قيم نسبية (0..1) أو نصوص (%)
    board_to_show = board.copy()
    # لو اللوح فيه نسب كأعداد (0..1) نحوّلها لنص %
    for c in board_to_show.columns:
        board_to_show[c] = board_to_show[c].apply(
            lambda v: f"{float(v)*100:.0f}%" if isinstance(v, (int, float)) and 0 <= float(v) <= 1 else v
        )

    st.dataframe(board_to_show.style.applymap(colorize), use_container_width=True)
except Exception as e:
    st.warning(f"تعذّر رسم اللوح (Board): {e}")

# ===================== عين الصقر (إيجاز) =====================
st.subheader("Eyes Eagle – إشارات سريعة (ملخّص)")
try:
    # نتوقع قاموسًا بسيطًا من compute_probs
    if isinstance(eyes_info, dict):
        for k, v in eyes_info.items():
            st.write(f"• {k}: {v}")
    else:
        st.write(eyes_info)
except Exception as e:
    st.warning(f"تعذّر عرض ملخص Eyes: {e}")

# ===================== التحديث التلقائي =====================
if auto:
    st.sidebar.write("⏳ سيتم التحديث الذاتي…")
    # الاسم الجديد للدالة في الإصدارات الحديثة
    st.rerun()
