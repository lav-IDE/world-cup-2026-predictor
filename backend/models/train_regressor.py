import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from backend.models.data_split import prepare_data


def train_regressor(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model: xgb.XGBRegressor, X_test, y_test) -> None:
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


def save_regressor(model: xgb.XGBRegressor, path: str) -> None:
    model.save_model(path)
    print(f"Saved regressor to {path}")


def run(feature_matrix_path: str, output_path: str) -> None:
    data = prepare_data(feature_matrix_path)

    model = train_regressor(data["X_train"], data["y_reg_train"])
    evaluate_regressor(model, data["X_test"], data["y_reg_test"])
    save_regressor(model, output_path)