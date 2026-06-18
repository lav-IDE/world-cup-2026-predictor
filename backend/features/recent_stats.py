import pandas as pd
import numpy as np

WINDOW = 5


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_team_history(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["date", "home_team", "away_team", "home_score", "away_score", "winner"]].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_score": "goals_scored",
        "away_score": "goals_conceded",
    })
    home["result"] = home.apply(
        lambda r: "draw" if r["winner"] == "Draw" else ("win" if r["winner"] == r["team"] else "loss"),
        axis=1
    )

    away = df[["date", "away_team", "home_team", "away_score", "home_score", "winner"]].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_score": "goals_scored",
        "home_score": "goals_conceded",
    })
    away["result"] = away.apply(
        lambda r: "draw" if r["winner"] == "Draw" else ("win" if r["winner"] == r["team"] else "loss"),
        axis=1
    )

    history = pd.concat([home, away], ignore_index=True)
    history = history.sort_values(["team", "date"]).reset_index(drop=True)
    return history



def compute_rolling(history: pd.DataFrame, window: int) -> pd.DataFrame:
    records = []

    for team, group in history.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)

        for i, row in group.iterrows():
            past = group[group["date"] < row["date"]].tail(window)

            if len(past) == 0:
                stats = {
                    "avg_goals_scored": np.nan,
                    "avg_goals_conceded": np.nan,
                    "win_rate": np.nan,
                    "draw_rate": np.nan,
                    "loss_rate": np.nan,
                }
            else:
                stats = {
                    "avg_goals_scored": past["goals_scored"].mean(),
                    "avg_goals_conceded": past["goals_conceded"].mean(),
                    "win_rate": (past["result"] == "win").mean(),
                    "draw_rate": (past["result"] == "draw").mean(),
                    "loss_rate": (past["result"] == "loss").mean(),
                }

            records.append({
                "team": team,
                "date": row["date"],
                "opponent": row["opponent"],
                **stats,
            })
    return pd.DataFrame(records)

def merge_rolling_stats(df: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    
    home_stats = rolling.rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "avg_goals_scored": "home_avg_goals_scored",
        "avg_goals_conceded": "home_avg_goals_conceded",
        "win_rate": "home_win_rate",
        "draw_rate": "home_draw_rate",
        "loss_rate": "home_loss_rate",
    })

    away_stats = rolling.rename(columns={
        "team": "away_team",
        "opponent": "home_team",
        "avg_goals_scored": "away_avg_goals_scored",
        "avg_goals_conceded": "away_avg_goals_conceded",
        "win_rate": "away_win_rate",
        "draw_rate": "away_draw_rate",
        "loss_rate": "away_loss_rate",
    })

    df = df.merge(home_stats, on=["date", "home_team", "away_team"], how="left")
    df = df.merge(away_stats, on=["date", "home_team", "away_team"], how="left")
    return df




def save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")