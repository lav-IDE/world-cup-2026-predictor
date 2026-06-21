import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from backend.models.data_split import prepare_data


def train_classifier(X_train, y_train) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model: xgb.XGBClassifier, X_test, y_test) -> None:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {loss:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted, order=[Draw, Home, Away]):")
    print(cm)


def save_classifier(model: xgb.XGBClassifier, path: str) -> None:
    model.save_model(path)
    print(f"Saved classifier to {path}")


def run(feature_matrix_path: str, output_path: str) -> None:
    data = prepare_data(feature_matrix_path)

    model = train_classifier(data["X_train"], data["y_clf_train"])
    evaluate_classifier(model, data["X_test"], data["y_clf_test"])
    save_classifier(model, output_path)