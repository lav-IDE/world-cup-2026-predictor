from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.features.odds import load_matches, filter_odds, compute_odds_features, save

INPUT_PATH = ROOT / "data/processed/matches.csv"
OUTPUT_PATH = ROOT / "data/processed/odds_features.csv"

df = load_matches(INPUT_PATH)
print(f"Loaded {len(df)} rows")

df = filter_odds(df)
print(f"Rows with odds data: {len(df)}")

df = compute_odds_features(df)
save(df, OUTPUT_PATH)