from pathlib import Path
from backend.models.train_classifier import run

ROOT = Path(__file__).resolve().parents[1]

FEATURE_MATRIX_PATH = ROOT / "data/processed/feature_matrix.csv"
OUTPUT_PATH = ROOT / "backend/models/artifacts/classifier.json"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

run(feature_matrix_path=str(FEATURE_MATRIX_PATH), output_path=str(OUTPUT_PATH))