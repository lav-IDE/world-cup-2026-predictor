import numpy as np
import xgboost as xgb
from backend.models.data_split import prepare_data


def check_agreement(feature_matrix_path: str, classifier_path: str, regressor_path: str) -> None:
    data = prepare_data(feature_matrix_path)
    X_test = data["X_test"]
    y_clf_test = data["y_clf_test"]
    y_reg_test = data["y_reg_test"]

    clf = xgb.XGBClassifier()
    clf.load_model(classifier_path)

    reg = xgb.XGBRegressor()
    reg.load_model(regressor_path)

    clf_preds = clf.predict(X_test)
    reg_preds = reg.predict(X_test)

    # classifier label convention: 0=Draw, 1=Home win, 2=Away win
    draw_margin = 0.5

    def reg_sign_to_label(pred, margin=draw_margin):
        if abs(pred) < margin:
            return 0
        elif pred > 0:
            return 1
        else:
            return 2

    reg_implied_labels = np.array([reg_sign_to_label(p) for p in reg_preds])

    agree = (clf_preds == reg_implied_labels)
    print(f"Total test rows: {len(X_test)}")
    print(f"Classifier vs regressor-implied label agreement: {agree.mean():.4f} ({agree.sum()}/{len(agree)})")

    # breakdown: where do they disagree, and what does each say?
    disagreement_cases = []
    label_names = {0: "Draw", 1: "Home", 2: "Away"}
    for i in range(len(X_test)):
        if clf_preds[i] != reg_implied_labels[i]:
            disagreement_cases.append({
                "clf_pred": label_names[clf_preds[i]],
                "reg_pred_goal_diff": round(float(reg_preds[i]), 3),
                "reg_implied": label_names[reg_implied_labels[i]],
                "actual_clf_label": label_names[y_clf_test.iloc[i]],
                "actual_goal_diff": int(y_reg_test.iloc[i]),
            })

    print(f"\nDisagreement cases: {len(disagreement_cases)}")
    print("\nSample of disagreements (first 15):")
    for case in disagreement_cases[:15]:
        print(case)

    # specifically flag the more severe case: classifier says Home/Away win
    # but regressor's predicted goal_diff sign points the OTHER direction
    # with enough magnitude to be a genuine opposite-outcome call (not just near zero)
    sign_flip_cases = [
        c for c in disagreement_cases
        if (c["clf_pred"] == "Home" and c["reg_pred_goal_diff"] < -draw_margin)
        or (c["clf_pred"] == "Away" and c["reg_pred_goal_diff"] > draw_margin)
    ]
    print(f"\nSevere sign-flip disagreements (clf says Home/Away, regressor goal_diff sign disagrees outright): {len(sign_flip_cases)}")
    for case in sign_flip_cases[:15]:
        print(case)


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    check_agreement(
        feature_matrix_path=str(ROOT / "data/processed/feature_matrix.csv"),
        classifier_path=str(ROOT / "backend/models/artifacts/classifier.json"),
        regressor_path=str(ROOT / "backend/models/artifacts/regressor.json"),
    )