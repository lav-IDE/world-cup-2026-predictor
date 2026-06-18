from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.data.preprocessing import *

results = load_results()
results = filter_years(results)
results = add_winner(results)

shootouts = load_shootouts()
shootouts = filter_years(shootouts)

matches = merge_shootouts(
    results,
    shootouts
)

odds = load_odds()

matches = merge_odds(
    matches,
    odds
)

matches.to_csv(
    ROOT / "data/processed/matches.csv",
    index=False
)