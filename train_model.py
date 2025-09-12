# train_model.py
"""
يشغل من الطرفية أو GitHub Codespaces:
- يقرأ data/combined_spins.csv  (أو غيّر المسار إذا لزم)
- يدرّب الموديل ويحفظه إلى artifacts/pattern_model.json
"""

import os
import pandas as pd
from models.pattern_model import fit_model

DATA_PATH = "data/combined_spins.csv"   # الناتج من combine_data.py
OUT_PATH  = "artifacts/pattern_model.json"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"لم أجد {DATA_PATH} — تأكد من تشغيل combine_data.py أولاً.")
    df = pd.read_csv(DATA_PATH)
    model = fit_model(
        df,
        half_life=80,        # يمكنك تعديلها لاحقًا
        bonus_boost=1.10,
        laplace=0.5
    )
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    model.save(OUT_PATH)
    print(f"✅ Model saved -> {OUT_PATH}")
    print(f"rows={model.meta['n_rows']}  trained_at={model.meta['trained_at']}")

if __name__ == "__main__":
    main()
