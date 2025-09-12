# app.py — Funky Brain v2.7.2 (Manual CSV, Game-like UI, Hawk-Eye)
# ---------------------------------------------------------------
# يقرأ CSV يدويًا: ts, segment, multiplier
# يقدّم: واجهة ملوّنة مثل اللعبة، احتمالات 10/15 رمية، Exp in 15 لكل خانة،
# Hawk-Eye (Stop / Go / Medium) مع ضوابط من الشريط الجانبي.

import math
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============ إعداد الصفحة ============
st.set_page_config(page_title="Funky Brain", page_icon="🧠", layout="wide")
BUILD = "Funky Brain – v2.7.2 (manual CSV)"
st.title("🧠 Funky Brain – LIVE")
st.caption(BUILD)

# ============ لوحة الألوان (مطابقة للعبة) ============
PALETTE = {
    "bg": "#0b0f17",
    "card": "#0f172a",
    "text": "#E5E7EB",
    "muted": "#9CA3AF",
    "border": "#1f2937",

    # مجموعات الحروف (كما طلبت):
    "orange": "#fb923c",   # PLAY
    "pink":   "#f472b6",   # FUNK
    "bar_txt": "#eab308",  # كتابة BAR: أصفر
    "bar_bg": "#166534",   # خلفية BAR: أخضر غامق
    "vip_txt": "#ffffff",  # VIP Disco كتابة: أبيض
    "vip_bg": "#991b1b",   # VIP Disco خلفية: أحمر غامق
    "disco_txt": "#0ea5e9",# Disco (الأيقونة الأزرق)
    "stay_txt": "#14b8a6", # StayinAlive تركواز

    "good": "#22c55e",
    "warn": "#f59e0b",
    "bad":  "#ef4444",
    "tile": "#111827",
}

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
      [data-testid="stHeader"] {{ background: transparent; }}
      .block-container {{ padding-top: 0.8rem; padding-bottom: 2rem; }}
      .pill {{
        display:inline-block; padding:.25rem .6rem; border-radius:999px;
        border:1px solid {PALETTE['border']}; background:{PALETTE['card']}; color:{PALETTE['text']};
        font-size:.78rem; margin-right:.35rem;
      }}
      .tile {{
        background:{PALETTE['tile']}; border:1px solid {PALETTE['border']};
        border-radius:14px; padding:.6rem .8rem; text-align:center;
        display:flex; flex-direction:column; gap:.25rem; justify-content:center; align-items:center;
      }}
      .tile .name {{ font-weight:800; letter-spacing:.5px; }}
      .tile .sub {{ font-size:.78rem; color:{PALETTE['muted']}; }}
      .grid {{
        display:grid; gap:.6rem; grid-template-columns: repeat(8, 1fr);
      }}
      .badge {{
        border-radius:10px; padding:.15rem .45rem; border:1px solid {PALETTE['border']};
        font-size:.75rem; display:inline-block; margin-left:.25rem;
      }}
      .exp-mini {{ font-size:.75rem; color:{PALETTE['muted']}; }}
      .hot {{ color:{PALETTE['good']}; }}
      .cold {{ color:{PALETTE['bad']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ دوال مساعدة ============
def parse_multiplier(x):
    """ 25X/1 000X/1kX → رقم """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    s = str(x).upper().strip().replace("×", "X").replace("*", "X").replace(" ", "")
    if s.endswith("X"):
        s = s[:-1]
    # دعم K/M
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
    """ خريطة المجموعات اللونية """
    if not isinstance(seg, str):
        return "Other"
    s = seg.strip().upper()
    if s in list("PLAY"):
        return "PLAY"     # برتقالي
    if s in list("FUNK"):
        return "FUNK"     # روز غامق
    if s == "BAR":
        return "BAR"      # كتابة صفراء وخلفية خضراء
    if s == "VIP":
        return "VIP"      # VIP Disco (أحمر غامق)
    if s in ("DISCO",):
        return "DISCO"    # أزرق
    if s in ("STAYINALIVE", "STAYIN ALIVE", "STAYINALIVE!"):
        return "STAY"     # تركواز
    if s.isdigit() or s == "1":
        return "ONE"
    return "Other"

# ترتيب اللوح (قرّبناه من تخطيطك، عدّله لو عندك ترتيب أدق)
BOARD_ORDER = [
    "1","BAR","P","L","A","Y","VIP","N","K","T","F","U","N","K","DISCO","STAYINALIVE"
]

def probs_block(df_win, tiles):
    """ يُرجع DataFrame باحتمالات/توقعات """
    rows = []
    total = len(df_win)
    if total == 0:
        return pd.DataFrame(columns=["Title","Group","p","Exp10","P1in10","P1in15"])
    for t in tiles:
        c = (df_win["segment"] == t).sum()
        p = c/total
        rows.append([t, group_of(t), p, 10*p, 1 - (1 - p)**10, 1 - (1 - p)**15])
    out = pd.DataFrame(rows, columns=["Title","Group","p","Exp10","P1in10","P1in15"])
    return out

def hawkeye_signal(df, window,
                   thr_stop_ones=0.55,
                   thr_stop_orange=0.25,
                   thr_go_bonus=0.22,
                   thr_medium_x50=0.10):
    """
    يحدّد إشارة عين الصقر:
      - STOP (أحمر): لو %1 داخل النافذة >= thr_stop_ones
                      + و PLAY (برتقالي) >= thr_stop_orange
      - GO (أخضر): لو مجموع حروف/بونص (غير الرقم 1) >= thr_go_bonus
      - MEDIUM (برتقالي): إذا نسبة الرميات ذات multiplier >= 50 داخل النافذة >= thr_medium_x50
      - Otherwise: NEUTRAL
    """
    if df.empty:
        return "NEUTRAL", {}

    recent = df.tail(window)
    if recent.empty:
        return "NEUTRAL", {}

    share_one = (recent["segment"] == "1").mean()
    share_orange = recent["segment"].isin(list("PLAY")).mean()

    # “بونص/حروف” تقريبية: كل ما عدا 1
    share_letters_bonus = (recent["segment"] != "1").mean()

    share_x50 = (recent["mult_num"] >= 50).mean()

    if share_one >= thr_stop_ones and share_orange >= thr_stop_orange:
        return "STOP", {
            "share_one": share_one, "share_orange": share_orange,
            "share_letters_bonus": share_letters_bonus, "share_x50": share_x50
        }

    if share_letters_bonus >= thr_go_bonus:
        return "GO", {
            "share_one": share_one, "share_orange": share_orange,
            "share_letters_bonus": share_letters_bonus, "share_x50": share_x50
        }

    if share_x50 >= thr_medium_x50:
        return "MEDIUM", {
            "share_one": share_one, "share_orange": share_orange,
            "share_letters_bonus": share_letters_bonus, "share_x50": share_x50
        }

    return "NEUTRAL", {
        "share_one": share_one, "share_orange": share_orange,
        "share_letters_bonus": share_letters_bonus, "share_x50": share_x50
    }

def style_tile(name, p_next, exp10, exp15, grp):
    """ يُرجع HTML لبطاقة خانة ملونة + Exp mini """
    # اسم ملوّن
    if grp == "PLAY":
        nm = f"<span class='name' style='color:{PALETTE['orange']}'>{name}</span>"
    elif grp == "FUNK":
        nm = f"<span class='name' style='color:{PALETTE['pink']}'>{name}</span>"
    elif grp == "BAR":
        nm = f"<span class='name' style='color:{PALETTE['bar_txt']}'>{name}</span>"
    elif grp == "VIP":
        nm = f"<span class='name' style='color:{PALETTE['vip_txt']}'>{name}</span>"
    elif grp == "DISCO":
        nm = f"<span class='name' style='color:{PALETTE['disco_txt']}'>{name}</span>"
    elif grp == "STAY":
        nm = f"<span class='name' style='color:{PALETTE['stay_txt']}'>{name}</span>"
    else:
        nm = f"<span class='name'>{name}</span>"

    # Mini exp in 15
    exp15_txt = f"<span class='exp-mini'>exp15: {exp15:.2f}</span>"

    return f"""
    <div class="tile">
      {nm}
      <div class="sub">P(next): {p_next*100:.2f}%</div>
      <div class="sub">Exp10: {exp10:.2f}</div>
      <div>{exp15_txt}</div>
    </div>
    """

# ============ الشريط الجانبي ============
with st.sidebar:
    st.subheader("⚙️ الإعدادات")
    window = st.slider("Window size (spins)", 50, 500, 200, step=10)
    st.caption("تُستخدم النافذة لحساب كل الإحصاءات الفورية.")

    st.markdown("---")
    st.subheader("🦅 Hawk-Eye thresholds")
    thr_stop_ones = st.slider("STOP: حد نسبة (1) في النافذة", 0.30, 0.90, 0.55, step=0.01)
    thr_stop_orange = st.slider("STOP: حد نسبة (PLAY) برتقالي", 0.10, 0.60, 0.25, step=0.01)
    thr_go_bonus = st.slider("GO: حد نسبة الحروف/البونص (≠ 1)", 0.10, 0.60, 0.22, step=0.01)
    thr_medium_x50 = st.slider("MEDIUM: حد نسبة (mult ≥ 50)", 0.02, 0.40, 0.10, step=0.01)

    st.markdown("---")
    st.subheader("📤 رفع CSV")
    upl = st.file_uploader("اختر ملفًا أو أكثر (CSV)", type=["csv"], accept_multiple_files=True)
    st.markdown('<span class="pill">ts | segment | multiplier</span>', unsafe_allow_html=True)

    st.markdown("---")
    auto = st.checkbox("تحديث تلقائي", value=False)
    every = st.slider("كل كم ثانية؟", 10, 180, 45, step=5)
    if st.button("🔄 Force reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ============ تحميل CSV وتنظيفه ============
df = pd.DataFrame(columns=["ts","segment","multiplier"])
if upl:
    parts = []
    for f in upl:
        try:
            raw = pd.read_csv(f)
            parts.append(raw)
        except Exception as e:
            st.error(f"خطأ قراءة {f.name}: {e}")
    if parts:
        df = pd.concat(parts, ignore_index=True)

def normalize_df(df_in):
    if df_in.empty:
        return df_in
    data = df_in.copy()

    # قبول رؤوس متنوعة
    lower = {c.lower(): c for c in data.columns}
    rename_map = {}
    for want in ["ts","segment","multiplier"]:
        if want not in data.columns and want in lower:
            rename_map[lower[want]] = want
    if rename_map:
        data = data.rename(columns=rename_map)

    for c in ["ts","segment","multiplier"]:
        if c not in data.columns:
            data[c] = np.nan

    def parse_ts(x):
        try:
            return pd.to_datetime(x, errors="coerce")
        except:
            return pd.NaT
    data["ts"] = data["ts"].apply(parse_ts)
    data["segment"] = data["segment"].astype(str).str.strip().str.upper()
    data["group"] = data["segment"].map(group_of)
    data["mult_num"] = data["multiplier"].apply(parse_multiplier)

    if data["ts"].notna().any():
        data = data.sort_values("ts")
    data = data.dropna(subset=["segment"]).reset_index(drop=True)
    return data

df = normalize_df(df)

if df.empty:
    st.info("⬆️ ارفع CSV بصيغة: ts, segment, multiplier لبدء التحليل.")
    st.stop()

# ============ بطاقات سريعة ============
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total spins", f"{len(df):,}")
with c2:
    last_ts = df["ts"].dropna().max()
    st.metric("Last spin", str(last_ts) if pd.notna(last_ts) else "—")
with c3:
    st.metric("Unique tiles", df["segment"].nunique())
with c4:
    st.metric("Avg Mult (win)", f"{df['mult_num'].dropna().mean():.2f}")

df_win = df.tail(window)

# ============ لوحة الاحتمالات (شبكة تشبه اللعبة) ============
st.subheader("🎛️ احتمال الظهور + Exp in 10/15 (شبكة)")

# جهّز ترتيب العرض
all_tiles = list(pd.unique(df["segment"]))
tiles = [t for t in BOARD_ORDER if t in all_tiles] + [t for t in all_tiles if t not in BOARD_ORDER]

pb = probs_block(df_win, tiles)

# طبّع بطاقات
cards_html = []
for _, row in pb.iterrows():
    t  = row["Title"]
    g  = row["Group"]
    p  = float(row["p"])
    e10 = float(row["Exp10"])
    e15 = 15 * p
    cards_html.append(style_tile(t, p, e10, e15, g))

grid = "<div class='grid'>" + "".join(cards_html) + "</div>"
st.markdown(grid, unsafe_allow_html=True)

# ============ Board Overview ============
st.subheader("🎡 Board Overview (داخل النافذة)")
freq = df_win["segment"].value_counts().rename("count").to_frame()
freq["rate%"] = (freq["count"]/len(df_win)*100).map(lambda v: f"{v:.2f}%")
last_seen = df_win.groupby("segment")["ts"].max().rename("last_seen")
merged = freq.join(last_seen, how="left").reset_index().rename(columns={"index":"segment"})
merged["order"] = merged["segment"].apply(lambda s: BOARD_ORDER.index(s) if s in BOARD_ORDER else 999)
merged = merged.sort_values(["order","segment"]).drop(columns=["order"])
st.dataframe(merged, use_container_width=True)

# ============ Hawk-Eye (عين الصقر) ============
st.subheader("🦅 Hawk-Eye – قنص البونص والضوارب")

signal, diag = hawkeye_signal(
    df, window,
    thr_stop_ones=thr_stop_ones,
    thr_stop_orange=thr_stop_orange,
    thr_go_bonus=thr_go_bonus,
    thr_medium_x50=thr_medium_x50
)

if signal == "STOP":
    st.error("🛑 **STOP** – انزلاق مرتفع محتمل خلال الـ 15 جولة القادمة (سلسلة 1 وبرتقالي كثيف).")
elif signal == "GO":
    st.success("✅ **GO** – فرصة قوية للحروف/البونص خلال الـ 10–15 جولة القادمة.")
elif signal == "MEDIUM":
    st.warning("🟠 **MEDIUM** – نشاط متوسط (x50+) ظهر بنسبة ملحوظة مؤخّرًا.")
else:
    st.caption("⚪ **NEUTRAL** – لا إشارة قوية الآن.")

if diag:
    cA, cB, cC, cD = st.columns(4)
    with cA: st.metric("Share of 1",  f"{diag['share_one']*100:.1f}%")
    with cB: st.metric("Share of PLAY",f"{diag['share_orange']*100:.1f}%")
    with cC: st.metric("Letters/Bonus (≠1)", f"{diag['share_letters_bonus']*100:.1f}%")
    with cD: st.metric("≥ x50 share", f"{diag['share_x50']*100:.1f}%")

# ستريكات مختصرة
recent = df.tail(window)
streaks = []
cur_seg, cur_len = None, 0
for s in recent["segment"]:
    if s == cur_seg: cur_len += 1
    else:
        if cur_seg is not None: streaks.append((cur_seg, cur_len))
        cur_seg, cur_len = s, 1
if cur_seg is not None: streaks.append((cur_seg, cur_len))
streaks.sort(key=lambda x: x[1], reverse=True)
if streaks:
    st.write("**Top streaks (recent):** " + " | ".join([f"**{s}** × {l}" for s,l in streaks[:6]]))

# ============ Raw ============
with st.expander("📄 Raw (cleaned)"):
    st.dataframe(df.tail(1000)[["ts","segment","multiplier","mult_num","group"]], use_container_width=True)

# ============ Auto refresh ============
if auto:
    st.caption(f"⟳ إعادة تحميل كل {every} ثانية…")
    time.sleep(every)
    st.rerun()
