import pandas as pd


def load_odds_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_elo_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_recent_stats_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def merge_elo(odds_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    merged = odds_df.merge(
        elo_df[["date", "home_team", "away_team", "home_elo", "away_elo"]],
        on=["date", "home_team", "away_team"],
        how="inner",
    )
    return merged


def merge_recent_stats(df: pd.DataFrame, recent_stats_df: pd.DataFrame) -> pd.DataFrame:
    stats_cols = [
        "date", "home_team", "away_team",
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "home_win_rate", "home_draw_rate", "home_loss_rate",
        "away_avg_goals_scored", "away_avg_goals_conceded",
        "away_win_rate", "away_draw_rate", "away_loss_rate",
    ]
    merged = df.merge(
        recent_stats_df[stats_cols],
        on=["date", "home_team", "away_team"],
        how="inner",
    )
    return merged


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [
        "date", "home_team", "away_team",
        "tournament", "neutral",
        "home_score", "away_score", "winner", "shootout_winner",
        "home_prob", "draw_prob", "away_prob", "overround",
        "home_elo", "away_elo",
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "home_win_rate", "home_draw_rate", "home_loss_rate",
        "away_avg_goals_scored", "away_avg_goals_conceded",
        "away_win_rate", "away_draw_rate", "away_loss_rate",
    ]
    return df[final_cols]


def build_feature_matrix(odds_path: str, elo_path: str, recent_stats_path: str) -> pd.DataFrame:
    odds_df = load_odds_features(odds_path)
    print(f"Loaded {len(odds_df)} rows from odds_features")

    elo_df = load_elo_history(elo_path)
    print(f"Loaded {len(elo_df)} rows from elo_history")

    recent_stats_df = load_recent_stats_history(recent_stats_path)
    print(f"Loaded {len(recent_stats_df)} rows from recent_stats_history")

    df = merge_elo(odds_df, elo_df)
    print(f"After merging elo_history: {len(df)} rows")

    df = merge_recent_stats(df, recent_stats_df)
    print(f"After merging recent_stats_history: {len(df)} rows")

    df = select_final_columns(df)
    return df


def save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")