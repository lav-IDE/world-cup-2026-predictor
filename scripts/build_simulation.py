from pathlib import Path
import pandas as pd

from backend.features.elo import ELOModel
from backend.features.recent_stats import get_team_history
from backend.models import monte_carlo as mc
from backend.models.monte_carlo_sim import tournament_structure as ts
from backend.models.monte_carlo_sim.sim_features import build_sim_feature_state, MatchRecord
from backend.models.monte_carlo_sim.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]

MATCHES_PATH = ROOT / "data/processed/matches.csv"
ANNEX_C_PATH = ROOT / "data/raw/ro32.csv"

CLASSIFIER_PATH = ROOT / "backend/models/artifacts/classifier.json"
GOAL_DIFF_REGRESSOR_PATH = ROOT / "backend/models/artifacts/regressor.json"
HOME_SCORE_REGRESSOR_PATH = ROOT / "backend/models/artifacts/home_score_regressor.json"
AWAY_SCORE_REGRESSOR_PATH = ROOT / "backend/models/artifacts/away_score_regressor.json"

ROLLING_STATS_WINDOW = 5
N_ITERATIONS = 10000


def build_canonical_state():
    df = pd.read_csv(MATCHES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    elo_model = ELOModel()
    elo_model.fit(df)

    state = build_sim_feature_state(elo_model, window=ROLLING_STATS_WINDOW)

    team_history = get_team_history(df)
    for team, group in team_history.groupby("team"):
        group = group.sort_values("date")
        records = [
            MatchRecord(
                date=row["date"],
                opponent=row["opponent"],
                goals_scored=row["goals_scored"],
                goals_conceded=row["goals_conceded"],
                result=row["result"],
            )
            for _, row in group.iterrows()
        ]
        if team not in state.rolling_trackers:
            from backend.models.monte_carlo_sim.sim_features import RollingStatsTracker
            state.rolling_trackers[team] = RollingStatsTracker(window=ROLLING_STATS_WINDOW)
        state.rolling_trackers[team].seed(records)

    return state


def build_predictor():
    return Predictor(
        classifier_path=str(CLASSIFIER_PATH),
        goal_diff_regressor_path=str(GOAL_DIFF_REGRESSOR_PATH),
        home_score_regressor_path=str(HOME_SCORE_REGRESSOR_PATH),
        away_score_regressor_path=str(AWAY_SCORE_REGRESSOR_PATH),
    )


def _format_representative_log(representative_log: dict) -> str:
    lines = [f"Representative tournament -- Champion: {representative_log['champion']}", ""]

    lines.append("=== Group Stage ===")
    for group_letter in sorted(representative_log["group_matches"].keys()):
        lines.append(f"\nGroup {group_letter}:")
        for m in representative_log["group_matches"][group_letter]:
            lines.append(f"  {m.home_team} {m.home_score}-{m.away_score} {m.away_team}")

    lines.append("\n=== Knockout Stage ===")
    knockout_matches = representative_log["knockout_matches"]
    round_labels = [
        ("Round of 32", ts.ALL_R32_MATCH_IDS),
        ("Round of 16", list(ts.R16_FROM_R32.keys())),
        ("Quarter-finals", list(ts.QF_FROM_R16.keys())),
        ("Semi-finals", list(ts.SF_FROM_QF.keys())),
        ("Final / Third Place", list(ts.FINAL_FROM_SF.keys()) + list(ts.THIRD_PLACE_MATCH_FROM_SF.keys())),
    ]
    for label, match_ids in round_labels:
        lines.append(f"\n{label}:")
        for match_id in match_ids:
            m = knockout_matches[match_id]
            shootout_note = " (on penalties)" if m.went_to_shootout else ""
            lines.append(
                f"  {match_id}: {m.home_team} {m.home_score}-{m.away_score} "
                f"{m.away_team}{shootout_note} -> {m.winner}"
            )

    return "\n".join(lines)


def main():
    canonical_state = build_canonical_state()
    predictor = build_predictor()
    annex_c_lookup = ts.load_annex_c(str(ANNEX_C_PATH))

    results, representative_log = mc.run_simulation(
        canonical_state, predictor, annex_c_lookup, N_ITERATIONS
    )

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df = results_df.sort_values("champion", ascending=False)
    print(results_df)

    output_path = ROOT / "data/processed/simulation_results.csv"
    results_df.to_csv(output_path)
    print(f"Saved results to {output_path}")

    bracket_text = _format_representative_log(representative_log)
    bracket_output_path = ROOT / "data/processed/representative_bracket.txt"
    bracket_output_path.write_text(bracket_text, encoding="utf-8")
    print(f"Saved representative bracket to {bracket_output_path}")


if __name__ == "__main__":
    main()