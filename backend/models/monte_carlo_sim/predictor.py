from __future__ import annotations
import pandas as pd
import xgboost as xgb


BASE_FEATURE_COLS = [
    "home_elo", "away_elo",
    "home_prob", "draw_prob", "away_prob", "overround",
    "neutral",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
]

TOURNAMENT_COLS = [
    "tournament_AFC Asian Cup",
    "tournament_AFC Asian Cup qualification",
    "tournament_African Cup of Nations",
    "tournament_African Cup of Nations qualification",
    "tournament_Al Ain International Cup",
    "tournament_CONCACAF Nations League",
    "tournament_CONCACAF Nations League qualification",
    "tournament_Canadian Shield",
    "tournament_Confederations Cup",
    "tournament_Copa América",
    "tournament_Copa América qualification",
    "tournament_Copa Confraternidad",
    "tournament_Copa Paz del Chaco",
    "tournament_Cyprus International Tournament",
    "tournament_FIFA Series",
    "tournament_FIFA World Cup",
    "tournament_FIFA World Cup qualification",
    "tournament_Friendly",
    "tournament_Gold Cup",
    "tournament_Gold Cup qualification",
    "tournament_Intercontinental Cup",
    "tournament_Jordan International Tournament",
    "tournament_Kirin Challenge Cup",
    "tournament_Kirin Cup",
    "tournament_Nations Cup",
    "tournament_Navruz Cup",
    "tournament_Nile Basin Tournament",
    "tournament_OSN Cup",
    "tournament_Oceania Nations Cup",
    "tournament_Soccer Ashes",
    "tournament_Superclásico de las Américas",
    "tournament_UEFA Euro",
    "tournament_UEFA Euro qualification",
    "tournament_UEFA Nations League",
    "tournament_Unity Cup",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + TOURNAMENT_COLS
assert len(ALL_FEATURE_COLS) == 52, len(ALL_FEATURE_COLS)

SIMULATED_MATCH_TOURNAMENT_COL = "tournament_FIFA World Cup"
assert SIMULATED_MATCH_TOURNAMENT_COL in TOURNAMENT_COLS

# --- Placeholder odds constants for simulated matches (see module docstring) ---
SIMULATED_DRAW_PROB = 0.24
SIMULATED_OVERROUND = 1.05


def _elo_to_match_probs(elo_system, home_team: str, away_team: str, neutral: bool):

    home_exp, away_exp = elo_system.get_expected(home_team, away_team, neutral)

    remaining = 1.0 - SIMULATED_DRAW_PROB
    home_prob = home_exp * remaining
    away_prob = away_exp * remaining

    return home_prob, SIMULATED_DRAW_PROB, away_prob


def build_feature_row(
    home_team: str,
    away_team: str,
    sim_feature_state,
    neutral: bool,
) -> pd.DataFrame:
    home_features = sim_feature_state.get_features(home_team)
    away_features = sim_feature_state.get_features(away_team)

    home_prob, draw_prob, away_prob = _elo_to_match_probs(
        sim_feature_state.elo_system, home_team, away_team, neutral
    )

    row = {
        "home_elo": home_features["elo"],
        "away_elo": away_features["elo"],
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "overround": SIMULATED_OVERROUND,
        "neutral": 1 if neutral else 0,
        "home_avg_goals_scored": home_features["avg_goals_scored"],
        "home_avg_goals_conceded": home_features["avg_goals_conceded"],
        "home_win_rate": home_features["win_rate"],
        "home_draw_rate": home_features["draw_rate"],
        "home_loss_rate": home_features["loss_rate"],
        "away_avg_goals_scored": away_features["avg_goals_scored"],
        "away_avg_goals_conceded": away_features["avg_goals_conceded"],
        "away_win_rate": away_features["win_rate"],
        "away_draw_rate": away_features["draw_rate"],
        "away_loss_rate": away_features["loss_rate"],
    }

    for col in TOURNAMENT_COLS:
        row[col] = 1 if col == SIMULATED_MATCH_TOURNAMENT_COL else 0

    df = pd.DataFrame([row])
    df = df[ALL_FEATURE_COLS]  # enforce exact column order
    return df


class Predictor:

    def __init__(
        self,
        classifier_path: str,
        goal_diff_regressor_path: str,
        home_score_regressor_path: str,
        away_score_regressor_path: str,
    ):
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(classifier_path)

        self.goal_diff_regressor = xgb.XGBRegressor()
        self.goal_diff_regressor.load_model(goal_diff_regressor_path)

        self.home_score_regressor = xgb.XGBRegressor()
        self.home_score_regressor.load_model(home_score_regressor_path)

        self.away_score_regressor = xgb.XGBRegressor()
        self.away_score_regressor.load_model(away_score_regressor_path)

        models_to_check = {
            "classifier": self.classifier,
            "goal_diff_regressor": self.goal_diff_regressor,
            "home_score_regressor": self.home_score_regressor,
            "away_score_regressor": self.away_score_regressor,
        }
        for name, model in models_to_check.items():
            features = list(model.get_booster().feature_names)
            if features != ALL_FEATURE_COLS:
                raise ValueError(
                    f"Loaded {name}'s feature_names does not match "
                    "ALL_FEATURE_COLS hard-coded in predict_fn.py. The "
                    "model was likely retrained with different "
                    "data/columns -- update ALL_FEATURE_COLS/TOURNAMENT_COLS "
                    "to match before trusting any predictions from this "
                    "Predictor."
                )

    def predict(
        self,
        home_team: str,
        away_team: str,
        sim_feature_state,
        neutral: bool,
    ) -> dict:

        X = build_feature_row(home_team, away_team, sim_feature_state, neutral)

        # Classifier label convention confirmed: 0=Draw, 1=Home win, 2=Away win
        clf_probs = self.classifier.predict_proba(X)[0]

        goal_diff_direct = float(self.goal_diff_regressor.predict(X)[0])
        home_score = float(self.home_score_regressor.predict(X)[0])
        away_score = float(self.away_score_regressor.predict(X)[0])

        return {
            "draw": float(clf_probs[0]),
            "home_win": float(clf_probs[1]),
            "away_win": float(clf_probs[2]),
            "home_score": home_score,
            "away_score": away_score,
            "goal_diff_direct": goal_diff_direct,
            "goal_diff_derived": home_score - away_score,
        }