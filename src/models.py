from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None

try:
    from .data_loader import (
        RANDOM_STATE,
        TARGET_COLUMN,
        correlation_with_target,
        load_data,
        make_preprocessor,
        missing_value_report,
        save_processed_dataset,
        split_features_target,
        train_test_split_data,
    )
except ImportError:
    from data_loader import (
        RANDOM_STATE,
        TARGET_COLUMN,
        correlation_with_target,
        load_data,
        make_preprocessor,
        missing_value_report,
        save_processed_dataset,
        split_features_target,
        train_test_split_data,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por-v1.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student-por-v1-processed.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def build_models(preprocessor) -> dict:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", Ridge()),
                ]
            ),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Decision Tree Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    if root_mean_squared_error is None:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    else:
        rmse = root_mean_squared_error(y_test, y_pred)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse,
        "R-squared": r2_score(y_test, y_pred),
    }


def train_all_models(return_models: bool = False):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(RAW_DATA_PATH)
    print("Kich thuoc du lieu:", df.shape)
    print("\nGia tri thieu theo cot:")
    print(missing_value_report(df))

    save_processed_dataset(df, PROCESSED_DATA_PATH)

    corr = correlation_with_target(df)
    corr.to_csv(REPORT_DIR / "correlation_with_G3.csv", header=["correlation"])
    print("\nTop 15 bien tuong quan manh nhat voi G3:")
    print(corr.head(15))

    X, y = split_features_target(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    preprocessor = make_preprocessor(X_train)
    models = build_models(preprocessor)

    results = []
    fitted_models = {}
    ridge_best_alpha = None
    for model_name, model in models.items():
        print(f"\nDang huan luyen: {model_name}")
        model.fit(X_train, y_train)

        best_alpha = None
        fitted_model = model
        if isinstance(model, GridSearchCV):
            best_alpha = model.best_params_["model__alpha"]
            ridge_best_alpha = best_alpha
            fitted_model = model.best_estimator_

        fitted_models[model_name] = fitted_model

        metrics = evaluate_model(model, X_test, y_test)
        results.append({"Model": model_name, **metrics})

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.loc[0, "Model"]
    print("\nBang so sanh mo hinh tren tap Test:")
    print(results_df.to_string(index=False))
    print(f"\nMo hinh tot nhat theo RMSE: {best_model_name}")
    if ridge_best_alpha is not None:
        print(f"Best alpha cua Ridge Regression: {ridge_best_alpha}")
    if return_models:
        return results_df, fitted_models, X_test, y_test, best_model_name
    return results_df


def main() -> None:
    _, fitted_models, X_test, y_test, best_model_name = train_all_models(return_models=True)
    try:
        from .visualize_models import generate_visualizations
    except ImportError:
        from visualize_models import generate_visualizations

    generate_visualizations(fitted_models, X_test, y_test, best_model_name)

if __name__ == "__main__":
    main()
