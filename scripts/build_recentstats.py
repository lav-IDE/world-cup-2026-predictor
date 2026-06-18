from pathlib import Path
from backend.features.recent_stats import load_results, get_team_history, compute_rolling, merge_rolling_stats, save

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data/processed/matches.csv"
OUTPUT_PATH = ROOT / "data/processed/recent_stats.csv"

WINDOW = 5

TARGET_TEAMS = {
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde",
    "Colombia", "Croatia", "Curaçao", "Czech Republic", "DR Congo",
    "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
    "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
    "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
}

df = load_results(str(INPUT_PATH))
print(f"Loaded {len(df)} rows")

df = df[df["home_team"].isin(TARGET_TEAMS) & df["away_team"].isin(TARGET_TEAMS)]
print(f"Filtered to {len(df)} rows with WC 2026 teams only")

history = get_team_history(df)
print(f"Flattened to {len(history)} team-match rows")

rolling = compute_rolling(history, window=WINDOW)
HISTORY_OUTPUT_PATH = ROOT / "data/processed/recent_stats_history.csv"
rolling.to_csv(HISTORY_OUTPUT_PATH, index=False)
print(f"Saved {len(rolling)} rows to {HISTORY_OUTPUT_PATH}")

df = merge_rolling_stats(df, rolling)
df = df[["date", "home_team", "away_team",
         "home_avg_goals_scored", "home_avg_goals_conceded",
         "home_win_rate", "home_draw_rate", "home_loss_rate",
         "away_avg_goals_scored", "away_avg_goals_conceded",
         "away_win_rate", "away_draw_rate", "away_loss_rate"]]
df = df[df["date"] >= "2024-01-01"]

# the most recent rolling stat row per team
recent = rolling[rolling["team"].isin(TARGET_TEAMS)].copy()
recent = recent.sort_values("date").groupby("team").last().reset_index()
recent = recent[["team", "avg_goals_scored", "avg_goals_conceded", "win_rate", "draw_rate", "loss_rate"]]

save(recent, str(OUTPUT_PATH))