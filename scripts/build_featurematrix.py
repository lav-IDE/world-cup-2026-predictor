from pathlib import Path
from backend.models.feature_matrix import build_feature_matrix, save

ROOT = Path(__file__).resolve().parents[1]

ODDS_PATH = ROOT / "data/processed/odds_features.csv"
ELO_HISTORY_PATH = ROOT / "data/processed/elo_history.csv"
RECENT_STATS_HISTORY_PATH = ROOT / "data/processed/recent_stats_history.csv"
OUTPUT_PATH = ROOT / "data/processed/feature_matrix.csv"

df = build_feature_matrix(
    odds_path=str(ODDS_PATH),
    elo_path=str(ELO_HISTORY_PATH),
    recent_stats_path=str(RECENT_STATS_HISTORY_PATH),
)

save(df, str(OUTPUT_PATH))