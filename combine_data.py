# combine_data.py
import pandas as pd
import os

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "combined_spins.csv")

def clean_df(df):
    # نتأكد أن الأعمدة موجودة
    needed = ["ts", "segment", "multiplier"]
    df = df[needed]

    # تحويل التاريخ
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # توحيد segment
    df["segment"] = df["segment"].astype(str).str.strip().str.upper()

    # إصلاح multiplier
    df["multiplier"] = (
        df["multiplier"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)  # نأخذ الرقم فقط
        .fillna("1")
        .astype(int)
    )
    return df

def main():
    all_dfs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".csv"):
            df = pd.read_csv(path)
        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            continue

        if not {"ts", "segment", "multiplier"}.issubset(df.columns):
            print(f"⚠️ تخطيت {file}: أعمدة ناقصة")
            continue

        all_dfs.append(clean_df(df))

    if not all_dfs:
        print("❌ لا يوجد ملفات صالحة للدمج")
        return

    final_df = pd.concat(all_dfs, ignore_index=True).dropna()
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ تم حفظ الملف المدموج: {OUTPUT_FILE} ({len(final_df)} صف)")

if __name__ == "__main__":
    main()
