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


def train_score_regressor(X_train, y_train) -> xgb.XGBRegressor:

    model = xgb.XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_score_regressor(model: xgb.XGBRegressor, X_test, y_test, label: str) -> None:
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"{label} MAE: {mae:.4f}")
    print(f"{label} RMSE: {rmse:.4f}")


def cross_check_against_goal_diff(
    goal_diff_model: xgb.XGBRegressor,
    home_score_model: xgb.XGBRegressor,
    away_score_model: xgb.XGBRegressor,
    X_test,
) -> None:

    predicted_goal_diff_direct = goal_diff_model.predict(X_test)
    predicted_home = home_score_model.predict(X_test)
    predicted_away = away_score_model.predict(X_test)
    predicted_goal_diff_derived = predicted_home - predicted_away

    mae_between_models = mean_absolute_error(
        predicted_goal_diff_direct, predicted_goal_diff_derived
    )
    mean_bias = (predicted_goal_diff_derived - predicted_goal_diff_direct).mean()

    print(f"Cross-check MAE: {mae_between_models:.4f}")
    print(f"Cross-check bias: {mean_bias:.4f}")


def run(feature_matrix_path: str, goal_diff_output_path: str,
        home_score_output_path: str, away_score_output_path: str) -> None:
    data = prepare_data(feature_matrix_path)

    goal_diff_model = train_regressor(data["X_train"], data["y_reg_train"])
    evaluate_regressor(goal_diff_model, data["X_test"], data["y_reg_test"])
    save_regressor(goal_diff_model, goal_diff_output_path)

    home_score_model = train_score_regressor(
        data["X_train"], data["y_home_score_train"]
    )
    evaluate_score_regressor(
        home_score_model, data["X_test"], data["y_home_score_test"], "home_score"
    )
    save_regressor(home_score_model, home_score_output_path)

    away_score_model = train_score_regressor(
        data["X_train"], data["y_away_score_train"]
    )
    evaluate_score_regressor(
        away_score_model, data["X_test"], data["y_away_score_test"], "away_score"
    )
    save_regressor(away_score_model, away_score_output_path)

    cross_check_against_goal_diff(
        goal_diff_model, home_score_model, away_score_model, data["X_test"]
    )