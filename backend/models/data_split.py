import pandas as pd

CUTOFF_DATE = "2023-01-01"

BASE_FEATURE_COLS = [
    "home_elo", "away_elo",
    "home_prob", "draw_prob", "away_prob", "overround",
    "neutral",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
]


def load_feature_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def encode_tournament(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df["tournament"], prefix="tournament")
    df = pd.concat([df, dummies], axis=1)
    return df, list(dummies.columns)


def add_classifier_label(df: pd.DataFrame) -> pd.DataFrame:
    def label(row):
        if row["winner"] == "Draw":
            return 0
        elif row["winner"] == row["home_team"]:
            return 1
        else:
            return 2

    df["home_result"] = df.apply(label, axis=1)
    return df


def add_regressor_label(df: pd.DataFrame) -> pd.DataFrame:
    df["goal_diff"] = df["home_score"] - df["away_score"]
    return df


def chronological_split(df: pd.DataFrame, cutoff: str = CUTOFF_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def prepare_data(path: str):
    df = load_feature_matrix(path)
    print(f"Loaded {len(df)} rows from feature_matrix")

    df, tournament_cols = encode_tournament(df)
    df = add_classifier_label(df)
    df = add_regressor_label(df)

    feature_cols = BASE_FEATURE_COLS + tournament_cols

    train, test = chronological_split(df)
    print(f"Train: {len(train)} rows (before {CUTOFF_DATE})")
    print(f"Test: {len(test)} rows (on/after {CUTOFF_DATE})")

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    y_clf_train = train["home_result"]
    y_clf_test = test["home_result"]

    y_reg_train = train["goal_diff"]
    y_reg_test = test["goal_diff"]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_clf_train": y_clf_train,
        "y_clf_test": y_clf_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "feature_cols": feature_cols,
    }