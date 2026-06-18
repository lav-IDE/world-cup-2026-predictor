import pandas as pd

def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_odds(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["home_odd", "draw_odd", "away_odd"]).copy()


def compute_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df["overround"] = (1 / df["home_odd"]) + (1 / df["draw_odd"]) + (1 / df["away_odd"])
    df["home_prob"] = (1 / df["home_odd"]) / df["overround"]
    df["draw_prob"] = (1 / df["draw_odd"]) / df["overround"]
    df["away_prob"] = (1 / df["away_odd"]) / df["overround"]
    df = df.drop(columns=["home_odd", "draw_odd", "away_odd"])
    return df


def save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")

