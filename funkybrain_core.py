import re, numpy as np, pandas as pd
LETTERS = list("PLAYFUNKTIME")
ORDER   = ["1","BAR"] + list("PLAY") + list("FUNK") + list("TIME") + ["DISCO","VIP","STAYINALIVE"]

def beta_prob(s, n, a0=2, b0=2):
    f = max(n-s,0)
    return (s+a0)/(s+f+a0+b0) if n>0 else 0.0

def extract_segment_from_url(url: str):
    m = re.search(r"funky-time/([A-Za-z0-9]+)\.png", str(url))
    if not m:
        t = str(url).strip().upper()
        return t if (t in ORDER or t in LETTERS) else None
    token = m.group(1).upper()
    if token.startswith("DISCO"): 
        return "DISCO"
    mp = {"1":"1","BAR":"BAR","VIP":"VIP","STAYINALIVE":"STAYINALIVE",
          "P":"P","L":"L","A":"A","Y":"Y",
          "F":"F","U":"U","N":"N","K":"K",
          "T":"T","I":"I","M":"M","E":"E"}
    return mp.get(token, token)

def extract_multiplier(v):
    m = re.match(r"(\d+)\s*[Xx]", str(v).strip())
    if m: 
        return int(m.group(1))
    s = str(v).strip()
    if s.isdigit(): 
        return int(s)
    return 1

def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = raw.columns.tolist()
    c_ts, c_img, c_mul = cols[0], (cols[1] if len(cols)>1 else cols[0]), (cols[2] if len(cols)>2 else cols[-1])
    df = pd.DataFrame()
    df["ts"] = raw[c_ts]
    df["segment"] = raw[c_img].apply(extract_segment_from_url)
    df["multiplier"] = raw[c_mul].apply(extract_multiplier)
    return df.dropna(subset=["segment"]).reset_index(drop=True)

def compute_probs(df: pd.DataFrame, window_size: int):
    win = df.tail(min(window_size, len(df))).copy()
    seq, N = win["segment"].tolist(), len(win)
    p_single = {s: beta_prob(seq.count(s), N) for s in ORDER}
    ge1 = lambda p,T: 1 - (1 - p)**T
    T10,T15 = 10,15
    tiles = pd.DataFrame([{
        "Tile": s,
        "P(next)": p_single[s],
        "Exp in 10": T10*p_single[s],
        "P(≥1 in 10)": ge1(p_single[s],T10),
        "Exp in 15": T15*p_single[s],
        "P(≥1 in 15)": ge1(p_single[s],T15),
    } for s in ORDER])
    prob_x50  = beta_prob(int((win["multiplier"]>=50).sum()), N)
    prob_x100 = beta_prob(int((win["multiplier"]>=100).sum()), N)
    eyes = pd.DataFrame([
        {"Metric":"P(≥1 x50 in 15)",  "Value": 1-(1-prob_x50)**15,  "Exp in 15":15*prob_x50,  "Signal":"MEDIUM", "Note":"Chance of 50x+"},
        {"Metric":"P(≥1 x100 in 15)", "Value": 1-(1-prob_x100)**15, "Exp in 15":15*prob_x100, "Signal":"MEDIUM", "Note":"Chance of 100x+"},
        {"Metric":"Stop Alert",        "Value": np.nan,             "Exp in 15":15*p_single["1"], "Signal":"STOP" if p_single["1"]>0.5 else "", "Note":"If '1' dominates"},
    ])
    return tiles, eyes, win

def board_matrix(probs_10: dict):
    layout = [["1","BAR"], list("PLAY"), list("FUNK"), list("TIME"), ["DISCO","STAYINALIVE","VIP"]]
    rows = []
    for row in layout:
        rows.append([(s, f"{probs_10.get(s,0)*100:.0f}%") for s in row])
    return rows
