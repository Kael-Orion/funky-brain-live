import pandas as pd
import glob
from pathlib import Path

# المجلد اللي نحفظ فيه كل الملفات
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# اقرأ كل الملفات داخل data/ اللي تنتهي بـ csv أو xlsx
files = list(DATA_DIR.glob("spins_cleaned_*.csv")) + list(DATA_DIR.glob("spins_cleaned_*.xlsx"))

all_dfs = []
for f in files:
    try:
        if f.suffix == ".csv":
            df = pd.read_csv(f)
        else:  # xlsx
            df = pd.read_excel(f)
        all_dfs.append(df)
        print(f"✅ Loaded {f} with {len(df)} rows")
    except Exception as e:
        print(f"⚠️ Failed to read {f}: {e}")

if not all_dfs:
    print("⚠️ لم أجد أي ملفات في مجلد data/. تأكد أنك حفظت هناك ملفات باسم spins_cleaned_*.csv أو .xlsx")
else:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(inplace=True)
    combined.sort_values("ts", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # احفظ النتيجة
    out_csv = DATA_DIR / "all_spins.csv"
    out_xlsx = DATA_DIR / "all_spins.xlsx"

    combined.to_csv(out_csv, index=False)
    combined.to_excel(out_xlsx, index=False)

    print(f"🎉 تم إنشاء {out_csv} و {out_xlsx} بعدد {len(combined)} رمية.")
