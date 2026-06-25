from pathlib import Path
from backend.models.train_regressor import run

ROOT = Path(__file__).resolve().parents[1]

FEATURE_MATRIX_PATH = ROOT / "data/processed/feature_matrix.csv"

GOAL_DIFF_OUTPUT_PATH = ROOT / "backend/models/artifacts/regressor.json"
HOME_SCORE_OUTPUT_PATH = ROOT / "backend/models/artifacts/home_score_regressor.json"
AWAY_SCORE_OUTPUT_PATH = ROOT / "backend/models/artifacts/away_score_regressor.json"

GOAL_DIFF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

run(
    feature_matrix_path=str(FEATURE_MATRIX_PATH),
    goal_diff_output_path=str(GOAL_DIFF_OUTPUT_PATH),
    home_score_output_path=str(HOME_SCORE_OUTPUT_PATH),
    away_score_output_path=str(AWAY_SCORE_OUTPUT_PATH),
)