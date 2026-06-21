from pathlib import Path
from backend.features.recent_stats import load_results, get_team_history, compute_rolling, merge_rolling_stats

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data/processed/matches.csv"
HISTORY_OUTPUT_PATH = ROOT / "data/processed/recent_stats_history.csv"

WINDOW = 5

df = load_results(str(INPUT_PATH))
print(f"Loaded {len(df)} rows")

history = get_team_history(df)
print(f"Flattened to {len(history)} team-match rows")

rolling = compute_rolling(history, window=WINDOW)

df = merge_rolling_stats(df, rolling)

per_match_cols = ["date", "home_team", "away_team",
                   "home_avg_goals_scored", "home_avg_goals_conceded",
                   "home_win_rate", "home_draw_rate", "home_loss_rate",
                   "away_avg_goals_scored", "away_avg_goals_conceded",
                   "away_win_rate", "away_draw_rate", "away_loss_rate"]

df[per_match_cols].to_csv(HISTORY_OUTPUT_PATH, index=False)
print(f"Saved {len(df)} rows to {HISTORY_OUTPUT_PATH}")