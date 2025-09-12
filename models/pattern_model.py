# models/pattern_model.py
"""
PatternModel — موديل بسيط يتعلّم من الأرشيف:
- يحسب احتمالات أولية لكل قطاع (priors) بترجيح حداثة أُسّي (half-life)
- يحسب مصفوفة انتقالات (Markov 1st-order) بين القطاعات مع Laplace smoothing
- دوال: fit_model(df) / save_model(path) / load_model(path) / predict_next(df_recent)
"""

from __future__ import annotations
import json, os, math, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ALL_SEGMENTS = [
    "1","BAR","P","L","A","Y","F","U","N","K","Y","T","I","M","E","DISCO","STAYINALIVE","DISCO_VIP"
]
BONUS_SEGMENTS = {"DISCO","STAYINALIVE","DISCO_VIP","BAR"}

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"ts","segment","multiplier"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df = df.copy()
    # زمن اختياري
    try:
        df["ts"] = pd.to_datetime(df["ts"])
    except Exception:
        pass
    df["segment"] = df["segment"].astype(str).str.upper()
    # multiplier -> int
    df["multiplier"] = (
        df["multiplier"].astype(str).str.extract(r"(\d+)", expand=False).fillna("1").astype(int)
    )
    # فلترة على القطاعات المعروفة فقط
    df = df[df["segment"].isin(ALL_SEGMENTS)].reset_index(drop=True)
    return df

def _exp_weights(n: int, half_life: int) -> np.ndarray:
    if n <= 0: 
        return np.array([])
    ages = np.arange(n, 0, -1)  # الأحدث عمره 1
    w = np.power(0.5, (ages-1)/max(half_life,1))
    return w / w.sum()

@dataclass
class PatternModel:
    segments: List[str]
    priors: Dict[str, float]          # P(s)
    trans: Dict[str, Dict[str, float]]# P(next|s)
    meta: Dict

    # ---------- حفظ/تحميل ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "segments": self.segments,
                "priors": self.priors,
                "trans": self.trans,
                "meta": self.meta,
            }, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "PatternModel":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return PatternModel(
            segments=obj["segments"],
            priors={k: float(v) for k,v in obj["priors"].items()},
            trans={a: {b: float(p) for b,p in row.items()} for a,row in obj["trans"].items()},
            meta=obj.get("meta",{})
        )

# ---------- تدريب ----------
def fit_model(
    df_archive: pd.DataFrame,
    half_life:int = 80,              # ترجيح الحداثة
    bonus_boost:float = 1.10,        # تعزيز بسيط للبونص في priors
    laplace:float = 0.5              # سموثينغ لمصفوفة الانتقالات
) -> PatternModel:
    df = _clean_df(df_archive)
    segs = list(ALL_SEGMENTS)
    n = len(df)
    # priors بالترجيح الأُسّي
    if n == 0:
        pri = np.ones(len(segs))/len(segs)
    else:
        w = _exp_weights(n, half_life)
        counts = {s:0.0 for s in segs}
        for s, wt in zip(df["segment"], w):
            counts[s] += wt
        pri = np.array([counts[s] for s in segs], dtype=float)
        # تعزيز للبونص
        for i,s in enumerate(segs):
            if s in BONUS_SEGMENTS:
                pri[i] *= bonus_boost
        pri = pri / pri.sum()

    pri_dict = {s: float(p) for s,p in zip(segs, pri)}

    # مصفوفة انتقالات 1st-order مع Laplace
    T = np.full((len(segs), len(segs)), laplace, dtype=float)
    idx = {s:i for i,s in enumerate(segs)}
    for i in range(len(df)-1):
        a = df.at[i, "segment"]
        b = df.at[i+1, "segment"]
        if a in idx and b in idx:
            T[idx[a], idx[b]] += 1.0
    # تطبيع صفوف الانتقال
    T = T / T.sum(axis=1, keepdims=True)
    trans = {segs[i]: {segs[j]: float(T[i,j]) for j in range(len(segs))} for i in range(len(segs))}

    meta = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_rows": int(n),
            "half_life": half_life,
            "bonus_boost": bonus_boost,
            "laplace": laplace}
    return PatternModel(segs, pri_dict, trans, meta)

# ---------- تنبؤ ----------
def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = x / max(temperature, 1e-6)
    x = x - x.max()
    z = np.exp(x)
    return z / z.sum()

def predict_next(
    model: PatternModel,
    df_recent: pd.DataFrame | None,
    blend_trans: float = 0.6,      # مزج priors مع transition
    temperature: float = 1.3       # تركيز بسيط
) -> Dict[str, float]:
    segs = model.segments
    pri = np.array([model.priors[s] for s in segs], dtype=float)

    # انتقال من آخر خانة (إن وجدت)
    p_trans = pri.copy()
    if df_recent is not None and len(df_recent) > 0:
        last_seg = str(df_recent.iloc[-1]["segment"]).upper()
        if last_seg in model.trans:
            row = model.trans[last_seg]
            p_trans = np.array([row[s] for s in segs], dtype=float)

    mix = (1.0 - blend_trans) * pri + blend_trans * p_trans
    mix = softmax(mix, temperature=temperature)
    return {s: float(p) for s,p in zip(segs, mix)}

# أدوات مساعدة للاحتمال على n سبِنات وعدد متوقع
def p_at_least_once(p: float, n: int) -> float:
    p = float(p)
    return 1.0 - (1.0 - p)**int(n)

def expected_count(p: float, n: int) -> float:
    return float(p) * int(n)
