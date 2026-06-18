from pathlib import Path
import pandas as pd
from backend.features.elo import ELOModel

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data/processed/matches.csv"
OUTPUT_PATH = ROOT / "data/processed/elo_ratings.csv"


df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
print(f"loaded {len(df)} rows")

model = ELOModel()
model.fit(df)

ratings = model.compute_ratings()
print(f"filtered to {len(ratings)}")

ratings.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(ratings)} rows to {OUTPUT_PATH}")

HISTORY_OUTPUT_PATH = ROOT / "data/processed/elo_history.csv"
history = model.get_history()
history.to_csv(HISTORY_OUTPUT_PATH, index=False)
print(f"Saved {len(history)} rows to {HISTORY_OUTPUT_PATH}")
