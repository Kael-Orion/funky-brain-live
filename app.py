# app.py — Funky Brain LIVE (Manual CSV) v1.3
# -------------------------------------------
# رفع CSV يدويًا + ألوان/زينة مثل اللعبة + Hawk-Eye + احتمالات 10/15 رمية
# الأعمدة المطلوبة في CSV: ts, segment, multiplier
# مثال صف:
# 2025-09-12T23:45:00Z,K,25X

import math
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============ إعداد صفحة ستريملِت ============
st.set_page_config(page_title="Funky Brain LIVE", page_icon="🧠", layout="wide")
VERSION = "Funky Brain LIVE – v1.3 (manual-CSV)"
st.title("🧠 Funky Brain – LIVE")
st.caption(VERSION)

# ============ لوحة ألوان/ستايل قريب من اللعبة ============
PALETTE = {
    "bg": "#0b0f17",
    "card": "#121826",
    "muted": "#6b7280",
    "text": "#E5E7EB",
    "accent": "#22d3ee",   # Turquoise
    "orange": "#fb923c",   # PLAY (برتقالي)
    "pink": "#f472b6",     # FUNK (روز غامق)
    "bar": "#22c55e",      # BAR (أخضر كتابة)
    "vip": "#ef4444",      # VIP DISCO (أحمر غامق للـ VIP Disco)
    "disco": "#38bdf8",    # DISCO (أزرق)
    "stay": "#06b6d4",     # StayinAlive (turquoise)
    "warn": "#f59e0b",
    "good": "#34d399",
    "bad": "#ef4444",
}
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
      [data-testid="stHeader"] {{ background: transparent; }}
      .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
      .pill {{
        padding:.2rem .6rem;border-radius:999px;border:1px solid #1f2937;
        background:#0f172a;color:{PALETTE['text']};font-size:.78rem
      }}
      .table-small td, .table-small th {{ padding:.35rem .5rem; font-size:.85rem; }}
      .group-play {{ color:{PALETTE['orange']}; font-weight:700; }}
      .group-funk {{ color:{PALETTE['pink']};   font-weight:700; }}
      .group-bar  {{ color:{PALETTE['bar']};    font-weight:700; }}
      .group-vip  {{ color:{PALETTE['vip']};    font-weight:700; }}
      .group-disco{{ color:{PALETTE['disco']};  font-weight:700; }}
      .group-stay {{ color:{PALETTE['stay']};   font-weight:700; }}
      .hot {{ color:{PALETTE['good']} }}
      .cold {{ color:{PALETTE['bad']} }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ وظائف مساعدة ============
def parse_multiplier(x):
    """يحّول 25X/1 000X/1kX → رقم float (25/1000/1000)"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    s = str(x).upper().strip().replace("×", "X").replace("*", "X").replace(" ", "")
    # دعم K / M
    if s.endswith("X"):
        s = s[:-1]
    if s.endswith("K"):
        try:
            return float(s[:-1]) * 1_000
        except:
            return np.nan
    if s.endswith("M"):
        try:
            return float(s[:-1]) * 1_000_000
        except:
            return np.nan
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def group_of(seg):
    """تجميع القطع لونيًا كما اللعبة"""
    if not isinstance(seg, str):
        return "Other"
    s = seg.strip().upper()
    if s in list("PLAY"):
        return "PLAY"     # برتقالي
    if s in list("FUNK"):
        return "FUNK"     # روز غامق
    if s == "BAR":
        return "BAR"      # أخضر كتابة
    if s == "VIP":
        return "VIP"      # أحمر غامق
    if s in ("DISCO",):
        return "DISCO"    # أزرق
    if s in ("STAYINALIVE", "STAYIN ALIVE", "STAYINALIVE!"):
        return "STAY"     # Turquoise
    if s.isdigit() or s == "1":
        return "One"
    return "Other"

# ترتيب افتراضي للوحة (عدّله إذا تبغى المطابقة 1:1 مع تخطيطك)
BOARD_ORDER = [
    "1","BAR","P","L","A","Y","3","VIP","N","K","Y","T","F","U","N","K","DISCO","STAYINALIVE"
]

def probs_table(df_win, tiles):
    rows = []
    total = len(df_win)
    if total == 0:
        return pd.DataFrame(columns=["Title","Group","P(next)","Exp in 10","P(≥1 in 10)","P(≥1 in 15)"])
    for t in tiles:
        c = (df_win["segment"] == t).sum()
        p = c/total
        exp10 = 10*p
        p1in10 = 1 - (1 - p)**10
        p1in15 = 1 - (1 - p)**15
        rows.append([t, group_of(t), p, exp10, p1in10, p1in15])
    out = pd.DataFrame(rows, columns=["Title","Group","P(next)","Exp in 10","P(≥1 in 10)","P(≥1 in 15)"])
    out["P(next)"] = (out["P(next)"]*100).map(lambda v: f"{v:.2f}%")
    out["Exp in 10"] = out["Exp in 10"].map(lambda v: f"{v:.1f}")
    out["P(≥1 in 10)"] = (out["P(≥1 in 10)"]*100).map(lambda v: f"{v:.2f}%")
    out["P(≥1 in 15)"] = (out["P(≥1 in 15)"]*100).map(lambda v: f"{v:.2f}%")
    return out

def hawkeye(df, window):
    """عين الصقر: حار/بارد + ستريكات"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    recent = df.tail(window)
    base   = df

    freq_recent = recent["segment"].value_counts(normalize=True)
    freq_base   = base["segment"].value_counts(normalize=True)

    common = pd.concat([freq_recent, freq_base], axis=1).fillna(0.0)
    common.columns = ["recent","base"]
    common["delta"] = common["recent"] - common["base"]
    hot  = common.sort_values("delta", ascending=False).head(6).reset_index().rename(columns={"index":"segment"})
    cold = common.sort_values("delta", ascending=True ).head(6).reset_index().rename(columns={"index":"segment"})

    # أطول ستريك في آخر نافذة
    streaks = []
    cur_seg = None
    cur_len = 0
    for s in recent["segment"]:
        if s == cur_seg:
            cur_len += 1
        else:
            if cur_seg is not None:
                streaks.append((cur_seg, cur_len))
            cur_seg = s
            cur_len = 1
    if cur_seg is not None:
        streaks.append((cur_seg, cur_len))
    streaks.sort(key=lambda x: x[1], reverse=True)
    return hot, cold, streaks[:5]

# ============ الشريط الجانبي ============
with st.sidebar:
    st.subheader("⚙️ الإعدادات")
    window = st.slider("Window size (spins)", 50, 500, 200, step=10)
    auto = st.checkbox("تحديث تلقائي", value=False)
    every = st.slider("كل كم ثانية؟", 10, 180, 45, step=5)
    st.markdown("---")
    st.subheader("📤 ارفع ملفات CSV")
    upl = st.file_uploader("اختر ملفًا أو أكثر (CSV)", type=["csv"], accept_multiple_files=True)
    st.markdown('<div class="pill">صيغة الأعمدة: ts | segment | multiplier</div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 Force reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption("الألوان: "
               f"<span class='group-play'>PLAY</span>, "
               f"<span class='group-funk'>FUNK</span>, "
               f"<span class='group-bar'>BAR</span>, "
               f"<span class='group-vip'>VIP DISCO</span>, "
               f"<span class='group-disco'>DISCO</span>, "
               f"<span class='group-stay'>STAYINALIVE</span>", unsafe_allow_html=True)

# ============ تحميل CSV وتنظيفه ============
df = pd.DataFrame(columns=["ts","segment","multiplier"])
errors = []
if upl:
    parts = []
    for f in upl:
        try:
            raw = pd.read_csv(f)
            parts.append(raw)
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    if parts:
        df = pd.concat(parts, ignore_index=True)

def normalize_df(df_in):
    if df_in.empty:
        return df_in
    data = df_in.copy()

    # قبول رؤوس بأحرف مختلفة
    lower = {c.lower(): c for c in data.columns}
    rename_map = {}
    for want in ["ts","segment","multiplier"]:
        if want in data.columns:
            continue
        if want in lower:
            rename_map[lower[want]] = want
    if rename_map:
        data = data.rename(columns=rename_map)

    # تأكد من الأعمدة
    for c in ["ts","segment","multiplier"]:
        if c not in data.columns:
            data[c] = np.nan

    # parse ts
    def parse_ts(x):
        try:
            return pd.to_datetime(x, errors="coerce")
        except:
            return pd.NaT
    data["ts"] = data["ts"].apply(parse_ts)

    # segment/group
    data["segment"] = data["segment"].astype(str).str.strip().str.upper()
    data["group"] = data["segment"].map(group_of)

    # multiplier numeric
    data["mult_num"] = data["multiplier"].apply(parse_multiplier)

    # ترتيب بالتاريخ (لو موجود) وإزالة الفارغ
    if data["ts"].notna().any():
        data = data.sort_values("ts")
    data = data.dropna(subset=["segment"]).reset_index(drop=True)
    return data

df = normalize_df(df)

if errors:
    st.error("أخطاء أثناء قراءة بعض الملفات:")
    for e in errors:
        st.code(e, language="bash")

if df.empty:
    st.info("⬆️ ارفع CSV بصيغة: ts, segment, multiplier لبدء التحليل.")
    st.stop()

# ============ كروت سريعة ============
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total spins (uploaded)", f"{len(df):,}")
with c2:
    last_ts = df["ts"].dropna().max()
    st.metric("Last spin time", str(last_ts) if pd.notna(last_ts) else "—")
with c3:
    st.metric("Unique tiles", df["segment"].nunique())

df_win = df.tail(window)

# ============ التابات ============
tab_tiles, tab_board, tab_hawk, tab_raw = st.tabs(["📊 Tiles & Probabilities", "🎡 Board Overview", "🦅 Hawk-Eye", "📄 Raw"])

with tab_tiles:
    st.subheader("احتمالات/توقعات العشر/الخمسة عشر رمية القادمة")
    all_tiles = list(pd.unique(df["segment"]))
    tiles = [t for t in BOARD_ORDER if t in all_tiles] + [t for t in all_tiles if t not in BOARD_ORDER]
    table = probs_table(df_win, tiles)

    def color_group_html(g):
        cls = {
            "PLAY":"group-play",
            "FUNK":"group-funk",
            "BAR":"group-bar",
            "VIP":"group-vip",
            "DISCO":"group-disco",
            "STAY":"group-stay",
            "One":"", "Other":""
        }.get(g, "")
        return f"<span class='{cls}'>{g}</span>"

    if not table.empty:
        show = table.copy()
        show["Group"] = show["Group"].map(color_group_html)
        st.write(show.to_html(escape=False, index=False, classes=["table-small"]), unsafe_allow_html=True)
    else:
        st.warning("بيانات غير كافية لحساب الاحتمالات ضمن النافذة.")

with tab_board:
    st.subheader("لوحة القطع – التوزيع داخل النافذة")
    freq = df_win["segment"].value_counts().rename("count").to_frame()
    freq["rate%"] = (freq["count"] / len(df_win) * 100).map(lambda v: f"{v:.2f}%")
    last_seen = df_win.groupby("segment")["ts"].max().rename("last_seen")
    merged = freq.join(last_seen, how="left").reset_index().rename(columns={"index":"segment"})
    merged["order"] = merged["segment"].apply(lambda s: BOARD_ORDER.index(s) if s in BOARD_ORDER else 999)
    merged = merged.sort_values(["order","segment"]).drop(columns=["order"])
    st.dataframe(merged, use_container_width=True)

with tab_hawk:
    st.subheader("🦅 عين الصقر – حار/بارد + ستريكات")
    hot, cold, streaks = hawkeye(df, window)

    a, b = st.columns(2)
    with a:
        st.markdown("**Hot (الأكثر نشاطًا مقابل التاريخ):**")
        if not hot.empty:
            hot["recent%"] = (hot["recent"]*100).map(lambda v: f"{v:.2f}%")
            hot["base%"]   = (hot["base"]  *100).map(lambda v: f"{v:.2f}%")
            hot["Δ"] = (hot["delta"]*100).map(lambda v: f"+{v:.2f} pp")
            st.dataframe(hot[["segment","recent%","base%","Δ"]], use_container_width=True)
        else:
            st.caption("—")

    with b:
        st.markdown("**Cold (الأقل نشاطًا مقابل التاريخ):**")
        if not cold.empty:
            cold["recent%"] = (cold["recent"]*100).map(lambda v: f"{v:.2f}%")
            cold["base%"]   = (cold["base"]  *100).map(lambda v: f"{v:.2f}%")
            cold["Δ"] = (cold["delta"]*100).map(lambda v: f"{v:.2f} pp")
            st.dataframe(cold[["segment","recent%","base%","Δ"]], use_container_width=True)
        else:
            st.caption("—")

    st.markdown("**أطول ستريكات حديثة:**")
    if streaks:
        st.write(" | ".join([f"**{s}** × {l}" for s,l in streaks]))
    else:
        st.caption("—")

with tab_raw:
    st.subheader("Raw (cleaned)")
    st.dataframe(df.tail(1000)[["ts","segment","multiplier","mult_num","group"]], use_container_width=True)

# تحديث تلقائي بسيط
if auto:
    st.caption(f"⟳ سيتم إعادة التحميل كل {every} ثانية…")
    time.sleep(every)
    st.rerun()
